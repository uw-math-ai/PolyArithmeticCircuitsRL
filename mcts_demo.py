#!/usr/bin/env python3
"""Visual Monte Carlo Tree Search demo for the polynomial circuit builder."""

import argparse
import copy
import json
import math
import random
import re
import sys
from collections import deque
from dataclasses import dataclass
from itertools import count
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import networkx as nx
import sympy as sp


# ---------------------------------------------------------------------------
# Action encoding utilities (mirrors src/PPO RL/utils.py)
# ---------------------------------------------------------------------------
def encode_action(operation: str, node1_id: int, node2_id: int, max_nodes: int) -> int:
    if node1_id > node2_id:
        node1_id, node2_id = node2_id, node1_id
    offset = node1_id * max_nodes - (node1_id * (node1_id - 1)) // 2
    pair_idx = offset + (node2_id - node1_id)
    op_idx = 1 if operation == "multiply" else 0
    return pair_idx * 2 + op_idx


def decode_action(action_idx: int, max_nodes: int) -> Tuple[str, int, int]:
    op_idx = action_idx % 2
    pair_idx = action_idx // 2
    operation = "multiply" if op_idx == 1 else "add"
    node1_id = 0
    for i in range(max_nodes):
        offset = i * max_nodes - (i * (i - 1)) // 2
        if offset > pair_idx:
            break
        node1_id = i
    offset_for_node1 = node1_id * max_nodes - (node1_id * (node1_id - 1)) // 2
    node2_id = pair_idx - offset_for_node1 + node1_id
    return operation, node1_id, int(node2_id)


# ---------------------------------------------------------------------------
# Configuration and game state
# ---------------------------------------------------------------------------
@dataclass
class DemoConfig:
    n_variables: int = 2
    max_complexity: int = 6
    display_symbols: Optional[Sequence[str]] = None

    DEFAULT_SYMBOLS: Tuple[str, ...] = ("x", "y", "z", "u", "v", "w", "p", "q", "r", "s")

    def __post_init__(self) -> None:
        if self.display_symbols is None:
            self.display_symbols = self.DEFAULT_SYMBOLS
        if len(self.display_symbols) < self.n_variables:
            raise ValueError("Not enough display symbols provided for the configured variables")
        self.display_symbols = tuple(self.display_symbols[: self.n_variables])
        self.internal_symbols = [sp.Symbol(f"x{i}") for i in range(self.n_variables)]
        self.symbol_map = {self.internal_symbols[i]: self.display_symbols[i] for i in range(self.n_variables)}

    @property
    def max_nodes(self) -> int:
        return self.n_variables + self.max_complexity + 1

    def format_polynomial(self, expr: Optional[sp.Expr]) -> str:
        if expr is None:
            return "Start"
        expanded = sp.expand(expr)
        text = str(expanded)

        def replace_symbol(match: re.Match[str]) -> str:
            idx = int(match.group(1))
            if idx < len(self.display_symbols):
                return self.display_symbols[idx]
            return match.group(0)

        text = re.sub(r"x(\d+)", replace_symbol, text)
        text = text.replace("**", "^")
        text = text.replace("*", "")
        return text


class CircuitGameState:
    """Lightweight state clone of PPO Game for MCTS planning."""

    def __init__(self, target_poly: sp.Expr, config: DemoConfig):
        self.target_poly = sp.expand(target_poly)
        self.config = config
        self.symbols = config.internal_symbols
        self.actions: List[Tuple[str, int, int]] = []
        self.polynomials: List[sp.Expr] = []
        for i, sym in enumerate(self.symbols):
            self.actions.append(("input", i, -1))
            self.polynomials.append(sym)
        self.actions.append(("constant", -1, -1))
        self.polynomials.append(sp.Integer(1))
        self.max_nodes = config.max_nodes
        self.max_steps = config.max_complexity
        self.current_step = 0
        self.history: List[Dict[str, object]] = []
        self._target_terms = self.target_poly.as_coefficients_dict()

    def clone(self) -> "CircuitGameState":
        new_state = CircuitGameState.__new__(CircuitGameState)
        new_state.target_poly = self.target_poly
        new_state.config = self.config
        new_state.symbols = self.symbols
        new_state.actions = list(self.actions)
        new_state.polynomials = list(self.polynomials)
        new_state.max_nodes = self.max_nodes
        new_state.max_steps = self.max_steps
        new_state.current_step = self.current_step
        new_state.history = copy.deepcopy(self.history)
        new_state._target_terms = self._target_terms
        return new_state

    # --- Core game logic (mirrors src/PPO RL/State.py) ---
    def available_actions(self) -> List[int]:
        if self.current_step >= self.max_steps:
            return []
        actions: List[int] = []
        n_nodes = len(self.polynomials)
        for i in range(n_nodes):
            for j in range(i, n_nodes):
                actions.append(encode_action("add", i, j, self.max_nodes))
                actions.append(encode_action("multiply", i, j, self.max_nodes))
        return actions

    def apply_action(self, action_idx: int) -> None:
        operation, node1_id, node2_id = decode_action(action_idx, self.max_nodes)
        if node1_id >= len(self.polynomials) or node2_id >= len(self.polynomials):
            self.current_step = self.max_steps
            return
        poly1 = self.polynomials[node1_id]
        poly2 = self.polynomials[node2_id]
        if operation == "add":
            new_poly = sp.expand(poly1 + poly2)
        else:
            new_poly = sp.expand(poly1 * poly2)
        self.actions.append((operation, node1_id, node2_id))
        self.polynomials.append(new_poly)
        self.current_step += 1
        self.history.append(
            {
                "action_idx": action_idx,
                "operation": operation,
                "operands": (node1_id, node2_id),
                "operand_polys": [
                    self.config.format_polynomial(poly1),
                    self.config.format_polynomial(poly2),
                ],
                "result_expr": new_poly,
                "result_poly": self.config.format_polynomial(new_poly),
            }
        )

    def matches_target(self) -> bool:
        if self.current_step == 0:
            return False
        return sp.simplify(self.polynomials[-1] - self.target_poly) == 0

    def is_terminal(self) -> bool:
        return self.matches_target() or self.current_step >= self.max_steps

    def similarity_score(self) -> float:
        if self.current_step == 0:
            return 0.0
        try:
            current_terms = self.polynomials[-1].as_coefficients_dict()
        except Exception:
            return 0.0
        total = max(len(self._target_terms), 1)
        matches = sum(
            1 for term, coeff in self._target_terms.items() if current_terms.get(term) == coeff
        )
        return matches / total

    def evaluate_reward(self) -> float:
        if self.matches_target():
            return 1.0
        return self.similarity_score()

    def last_polynomial(self) -> Optional[sp.Expr]:
        if self.current_step == 0:
            return None
        return self.polynomials[-1]

    def last_polynomial_str(self, max_len: int = 40) -> str:
        expr = self.last_polynomial()
        text = self.config.format_polynomial(expr)
        return text if len(text) <= max_len else text[: max_len - 1] + "…"

    def describe_action(self, action_idx: int) -> str:
        operation, node1_id, node2_id = decode_action(action_idx, self.max_nodes)
        poly1 = self.config.format_polynomial(self.polynomials[node1_id])
        poly2 = self.config.format_polynomial(self.polynomials[node2_id])
        op_symbol = "×" if operation == "multiply" else "+"
        return f"{operation}({node1_id},{node2_id}): {poly1} {op_symbol} {poly2}"


# ---------------------------------------------------------------------------
# MCTS core
# ---------------------------------------------------------------------------
class MCTSNode:
    _ids = count()

    def __init__(self, state: CircuitGameState, parent: Optional["MCTSNode"], action_taken: Optional[int]):
        self.state = state
        self.parent = parent
        self.action_taken = action_taken
        self.children: Dict[int, MCTSNode] = {}
        self.untried_actions = state.available_actions()
        random.shuffle(self.untried_actions)
        self.visit_count = 0
        self.total_value = 0.0
        self.uid = next(MCTSNode._ids)

    def is_fully_expanded(self) -> bool:
        return not self.untried_actions

    def best_child(self, exploration_constant: float) -> "MCTSNode":
        best_score = -float("inf")
        best = None
        for child in self.children.values():
            if child.visit_count == 0:
                score = float("inf")
            else:
                exploit = child.total_value / child.visit_count
                explore = math.sqrt(math.log(self.visit_count) / child.visit_count)
                score = exploit + exploration_constant * explore
            if score > best_score:
                best_score = score
                best = child
        assert best is not None
        return best

    def add_child(self, action_idx: int, child_state: CircuitGameState) -> "MCTSNode":
        child = MCTSNode(child_state, parent=self, action_taken=action_idx)
        self.children[action_idx] = child
        return child

    def update(self, reward: float) -> None:
        self.visit_count += 1
        self.total_value += reward

    @property
    def mean_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count


class MCTS:
    def __init__(self, root_state: CircuitGameState, exploration_constant: float = 1.4):
        self.root = MCTSNode(root_state, parent=None, action_taken=None)
        self.exploration_constant = exploration_constant

    def run(self, iterations: int) -> None:
        for _ in range(iterations):
            node = self._select(self.root)
            reward = self._rollout_from(node)
            self._backpropagate(node, reward)

    def _select(self, node: MCTSNode) -> MCTSNode:
        current = node
        while not current.state.is_terminal():
            if current.untried_actions:
                return self._expand(current)
            current = current.best_child(self.exploration_constant)
        return current

    def _expand(self, node: MCTSNode) -> MCTSNode:
        action_idx = node.untried_actions.pop()
        next_state = node.state.clone()
        next_state.apply_action(action_idx)
        child = node.add_child(action_idx, next_state)
        return child

    def _rollout_from(self, node: MCTSNode) -> float:
        rollout_state = node.state.clone()
        while not rollout_state.is_terminal():
            actions = rollout_state.available_actions()
            if not actions:
                break
            action_idx = random.choice(actions)
            rollout_state.apply_action(action_idx)
        return rollout_state.evaluate_reward()

    def _backpropagate(self, node: MCTSNode, reward: float) -> None:
        current: Optional[MCTSNode] = node
        while current is not None:
            current.update(reward)
            current = current.parent

    # --- Analysis helpers ---
    def extract_best_path(self) -> Tuple[List[MCTSNode], List[Tuple[int, int]], MCTSNode]:
        path_nodes = [self.root]
        path_edges: List[Tuple[int, int]] = []
        current = self.root
        while current.children:
            best_child = max(current.children.values(), key=lambda c: c.visit_count)
            path_nodes.append(best_child)
            path_edges.append((current.uid, best_child.uid))
            current = best_child
            if current.state.is_terminal():
                break
        return path_nodes, path_edges, current

    def build_tree_graph(self) -> nx.DiGraph:
        graph = nx.DiGraph()
        queue: deque[MCTSNode] = deque([self.root])
        seen = set()
        while queue:
            node = queue.popleft()
            if node.uid in seen:
                continue
            seen.add(node.uid)
            last_expr = node.state.last_polynomial()
            full_expr = node.state.config.format_polynomial(last_expr)
            label = f"{node.state.last_polynomial_str()}\nN={node.visit_count} Q={node.mean_value:.2f}"
            graph.add_node(node.uid, label=label, full_expr=full_expr)
            for action_idx, child in node.children.items():
                edge_label = node.state.describe_action(action_idx)
                graph.add_edge(node.uid, child.uid, label=edge_label)
                queue.append(child)
        return graph


def visualize_tree_html(
        graph: nx.DiGraph,
        root_id: int,
        best_nodes: Sequence[int],
        best_edges: Sequence[Tuple[int, int]],
        output_path: Path,
        summary: Dict[str, object],
) -> None:
    best_nodes_set = set(best_nodes)
    best_edges_set = set(best_edges)

    vis_nodes = []
    vis_edges = []

    for node_id, data in graph.nodes(data=True):
        label = data.get("label", "")
        full_expr = data.get("full_expr", label)
        vis_nodes.append(
            {
                "id": node_id,
                "label": label.replace("\n", "<br>") if label else "",
                "title": full_expr,
                "color": "#ff6f69" if node_id in best_nodes_set else "#97c2fc",
                "font": {"multi": "html"},
            }
        )

    for u, v, data in graph.edges(data=True):
        vis_edges.append(
            {
                "from": u,
                "to": v,
                "label": data.get("label", ""),
                "color": {"color": "#ff6f69" if (u, v) in best_edges_set else "#6c757d"},
                "arrows": "to",
            }
        )

    nodes_json = json.dumps(vis_nodes)
    edges_json = json.dumps(vis_edges)
    summary_json = json.dumps(summary)

    html_content = f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
    <meta charset=\"utf-8\" />
    <title>MCTS Demo Tree</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 0; padding: 0; background: #f7f7f9; }}
        header {{ background: #343a40; color: #fff; padding: 1rem 1.5rem; }}
        header h1 {{ margin: 0; font-size: 1.6rem; letter-spacing: 0.02em; }}
        .container {{ padding: 1.75rem 1.5rem 2rem; max-width: 1080px; margin: 0 auto; }}
        #network {{ height: 680px; border: 1px solid #ced4da; background: #fff; border-radius: 6px; box-shadow: 0 2px 6px rgba(0,0,0,0.05); }}
        .search-bar {{ margin: 1rem 0; display: flex; gap: 0.5rem; align-items: center; }}
        .search-bar input {{ flex: 1; padding: 0.55rem 0.75rem; border: 1px solid #ced4da; border-radius: 4px; font-size: 0.95rem; }}
        .search-bar button {{ padding: 0.55rem 1.1rem; border: none; border-radius: 4px; background: #007bff; color: #fff; cursor: pointer; font-size: 0.95rem; }}
        .search-bar button:hover {{ background: #0069d9; }}
        .summary, .best-path {{ background: #fff; border-radius: 6px; border: 1px solid #dee2e6; padding: 1.1rem 1.25rem; margin-top: 1.5rem; box-shadow: 0 2px 6px rgba(0,0,0,0.03); }}
        .summary h2, .best-path h2 {{ margin-top: 0; }}
        .best-path ol {{ padding-left: 1.25rem; margin-bottom: 0; }}
        .best-path li {{ margin-bottom: 0.45rem; line-height: 1.4; }}
        .matches {{ margin-left: 0.75rem; color: #6c757d; font-size: 0.9rem; min-width: 90px; }}
        .note {{ margin-top: 1rem; padding: 0.75rem 1rem; background: #e7f4ff; border-left: 4px solid #1d6fb8; border-radius: 4px; color: #134b79; font-size: 0.95rem; box-shadow: inset 0 0 0 rgba(0,0,0,0); }}
        .note strong {{ display: inline-block; margin-right: 0.35rem; }}
    </style>
    <script src=\"https://unpkg.com/vis-network/standalone/umd/vis-network.min.js\"></script>
</head>
<body>
    <header>
        <h1>MCTS Tree Explorer</h1>
    </header>
    <div class=\"container\">
        <div class=\"summary\" id=\"summary\"></div>
        <div class=\"note\">
            <strong>Legend:</strong> N records how many simulations visited a node. Q is the average reward collected from rollouts that passed through the node (higher is better).
        </div>
        <div class=\"search-bar\">
            <input type=\"text\" id=\"searchInput\" placeholder=\"Search node text or tooltip...\" />
            <button id=\"clearBtn\">Clear</button>
            <span class=\"matches\" id=\"matchCount\"></span>
        </div>
        <div id=\"network\"></div>
        <div class=\"best-path\" id=\"bestPath\"></div>
    </div>
    <script>
        const nodesData = new vis.DataSet({nodes_json});
        const edgesData = new vis.DataSet({edges_json});
        const summaryData = {summary_json};

        const container = document.getElementById('network');
        const data = {{ nodes: nodesData, edges: edgesData }};
        const options = {{
            layout: {{
                hierarchical: {{
                    enabled: true,
                    direction: 'UD',
                    sortMethod: 'hubsize',
                    levelSeparation: 140,
                    nodeSpacing: 180
                }}
            }},
            edges: {{ smooth: true }},
            physics: false,
        }};

        const network = new vis.Network(container, data, options);

        function renderSummary() {{
            const summaryEl = document.getElementById('summary');
            const target = summaryData.target_polynomial;
            const iterations = summaryData.iterations;
            const matches = summaryData.final_match ? 'Yes' : 'No';
            const meanValue = summaryData.final_mean_value.toFixed(3);
            const visits = summaryData.final_visits;
            const finalPoly = summaryData.final_polynomial;
            summaryEl.innerHTML = `
                <h2>Run Summary</h2>
                <p><strong>Target polynomial:</strong> ${{target}}</p>
                <p><strong>MCTS iterations:</strong> ${{iterations}}</p>
                <p><strong>Best leaf matches target:</strong> ${{matches}}</p>
                <p><strong>Best leaf mean value:</strong> ${{meanValue}}</p>
                <p><strong>Best leaf visits:</strong> ${{visits}}</p>
                <p><strong>Best leaf polynomial:</strong> ${{finalPoly}}</p>
            `;
        }}

        function renderBestPath() {{
            const container = document.getElementById('bestPath');
            const steps = summaryData.best_path_steps || [];
            if (!steps.length) {{
                container.innerHTML = '<h2>Best Path</h2><p>No actions recorded.</p>';
                return;
            }}
            const items = steps.map((step, idx) => {{
                const symbol = step.operation === 'multiply' ? '×' : '+';
                const left = step.operand_polys[0] + ' ' + symbol + ' ' + step.operand_polys[1];
                return `<li><strong>Step ${{idx + 1}}:</strong> ${{step.operation}}(${{step.operands[0]}}, ${{step.operands[1]}}) — ${{left}} → ${{step.result_poly}}</li>`;
            }}).join('');
            container.innerHTML = `<h2>Best Path</h2><ol>${{items}}</ol>`;
        }}

        function setupSearch() {{
            const input = document.getElementById('searchInput');
            const clearBtn = document.getElementById('clearBtn');
            const matchLabel = document.getElementById('matchCount');
            const originalColors = nodesData.get().reduce((acc, node) => {{
                acc[node.id] = node.color;
                return acc;
            }}, {{}});

            function applySearch() {{
                const query = input.value.trim().toLowerCase();
                const allNodes = nodesData.get();
                const matches = [];
                const updates = [];

                allNodes.forEach(node => {{
                    const labelText = node.label ? node.label.replace(/<br>/g, ' ') : '';
                    const titleText = node.title || '';
                    const isMatch = query && (labelText.toLowerCase().includes(query) || titleText.toLowerCase().includes(query));
                    updates.push({{ id: node.id, color: isMatch ? '#ffc107' : originalColors[node.id] }});
                    if (isMatch) {{
                        matches.push(node.id);
                    }}
                }});

                nodesData.update(updates);
                network.unselectAll();
                if (matches.length) {{
                    network.selectNodes(matches, false);
                    network.fit({{ nodes: matches, animation: true }});
                    matchLabel.textContent = `${{matches.length}} match${{matches.length === 1 ? '' : 'es' }}`;
                }} else if (!query) {{
                    network.fit({{ animation: true }});
                    matchLabel.textContent = '';
                }} else {{
                    matchLabel.textContent = '0 matches';
                }}
            }}

            input.addEventListener('input', applySearch);
            clearBtn.addEventListener('click', () => {{
                input.value = '';
                applySearch();
            }});
            applySearch();
        }}

        renderSummary();
        renderBestPath();
        setupSearch();
    </script>
</body>
</html>"""

    output_path.write_text(html_content, encoding="utf-8")


# ---------------------------------------------------------------------------
# Target helpers
# ---------------------------------------------------------------------------
def parse_target_polynomial(expr_str: str, config: DemoConfig) -> sp.Expr:
    display_names = config.display_symbols
    display_symbols = [sp.Symbol(name) for name in display_names]
    local_dict = {name: display_symbols[idx] for idx, name in enumerate(display_names)}

    try:
        parsed = sp.sympify(expr_str, locals=local_dict)
    except Exception as exc:  # pragma: no cover - sympy errors are diverse
        raise ValueError(f"Unable to parse target polynomial '{expr_str}': {exc}") from exc

    unexpected = [sym for sym in parsed.free_symbols if sym not in display_symbols]
    if unexpected:
        names = ", ".join(sorted(str(sym) for sym in unexpected))
        allowed = ", ".join(display_names)
        raise ValueError(f"Target polynomial references unsupported symbols: {names}. Allowed symbols: {allowed}")

    subs_map = {display_symbols[idx]: config.internal_symbols[idx] for idx in range(config.n_variables)}
    internal_expr = sp.expand(parsed.xreplace(subs_map))
    return internal_expr


# ---------------------------------------------------------------------------
# Demo runner
# ---------------------------------------------------------------------------
def run_demo(config: DemoConfig, iterations: int, seed: int, target_expr: str, output_path: Path) -> None:
    random.seed(seed)
    target_poly = parse_target_polynomial(target_expr, config)

    print(f"Target polynomial   : {config.format_polynomial(target_poly)}")
    print(f"MCTS iterations     : {iterations}")

    initial_state = CircuitGameState(target_poly=target_poly, config=config)
    planner = MCTS(initial_state)
    planner.run(iterations)

    path_nodes, best_edges, best_leaf = planner.extract_best_path()
    best_node_ids = [node.uid for node in path_nodes]
    tree_graph = planner.build_tree_graph()

    final_match = best_leaf.state.matches_target()
    best_path_steps: List[Dict[str, object]] = []
    for idx, step in enumerate(best_leaf.state.history, start=1):
        op = step["operation"]
        operands = tuple(int(x) for x in step["operands"])
        best_path_steps.append(
            {
                "step": idx,
                "operation": op,
                "operands": list(operands),
                "operand_polys": list(step["operand_polys"]),
                "result_poly": step["result_poly"],
            }
        )

    final_poly_display = config.format_polynomial(best_leaf.state.last_polynomial())

    summary = {
        "target_polynomial": config.format_polynomial(target_poly),
        "iterations": iterations,
        "final_match": bool(final_match),
        "final_mean_value": float(best_leaf.mean_value),
        "final_visits": int(best_leaf.visit_count),
        "final_polynomial": final_poly_display,
        "best_path_steps": best_path_steps,
    }

    visualize_tree_html(tree_graph, planner.root.uid, best_node_ids, best_edges, output_path, summary)
    print(f"Saved tree visualization to {output_path}")

    print("\nBest path discovered:")
    for step in best_path_steps:
        op = step["operation"]
        operands = step["operands"]
        left_a, left_b = step["operand_polys"]
        result_poly = step["result_poly"]
        symbol = "×" if op == "multiply" else "+"
        print(
            f"  Step {step['step']}: {op}({operands[0]}, {operands[1]}) — {left_a} {symbol} {left_b} -> {result_poly}"
        )

    print(f"\nFinal polynomial: {final_poly_display}")
    print(f"Matches target : {'yes' if final_match else 'no'}")
    print(f"Mean value     : {best_leaf.mean_value:.3f}")
    print(f"Visits         : {best_leaf.visit_count}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an interactive MCTS demo on a custom polynomial target.")
    parser.add_argument("--iterations", type=int, default=300, help="Number of MCTS iterations to perform")
    parser.add_argument("--max-complexity", type=int, default=6, help="Maximum number of arithmetic operations allowed when building the circuit")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--output", type=str, default="mcts_tree.html", help="Output path for the interactive tree visualization")
    parser.add_argument("--variables", type=int, default=2, help="Number of base variables (e.g., x, y, z) available to the circuit builder")
    parser.add_argument(
        "--target",
        type=str,
        default="x**2 + 2*x*y + y**2",
        help="Target polynomial expression written with the chosen variable names (default: x**2 + 2*x*y + y**2)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    config = DemoConfig(n_variables=args.variables, max_complexity=args.max_complexity)
    output_path = Path(args.output)
    run_demo(config, args.iterations, args.seed, args.target, output_path)


if __name__ == "__main__":
    main(sys.argv[1:])
