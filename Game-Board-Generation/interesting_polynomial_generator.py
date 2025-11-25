#!/usr/bin/env python3
"""
Generate arithmetic-circuit "game boards" and extract interesting polynomials.

This script merges the capabilities of:
  - build-game-board.py      (graph construction + exports + plotting)
  - copygameboard.py         (multi-variable support)
  - pre-training-data/analyze_paths.py (path multiplicity analysis)

Key pipeline:
  1. Build the DAG of reachable polynomials up to C steps and N variables.
  2. Export sanitized GraphML + JSON + JSONL (nodes / edges).
  3. Run a path-multiplicity analysis to surface "interesting" polynomials,
     i.e. nodes with multiple distinct paths or multiple shortest paths.
  4. Optionally render a layered PNG visualization.

Runtime controls:
  --max-nodes                Hard cap on distinct polynomials in the DAG.
  --max-successors-per-node  Limit per-node expansions to tame blow-up.
  --analysis-max-step        Keep only nodes up to a target step/complexity.

Example:
    python interesting_polynomial_generator.py --steps 4 --num-vars 2 \
        --output-dir Game-Board-Generation/pre-training-data \
        --prefix game_board_C4 --with-labels --max-samples 5 \
        --only-multipath
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import matplotlib.pyplot as plt
import networkx as nx
from sympy import expand, srepr, symbols

MAX_ALLOWED_STEPS = 5


# ---------------------------------------------------------------------------
# Canonical SymPy helpers
# ---------------------------------------------------------------------------

def canon_key(expr) -> str:
    """Canonical, hashable string for a polynomial expression."""
    return srepr(expand(expr))


def pretty_label(expr, max_len: int = 32) -> str:
    """Friendly truncated label for plotting."""
    text = str(expand(expr))
    return text if len(text) <= max_len else (text[: max_len - 1] + "…")


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_game_graph(
    steps: int,
    num_vars: int = 1,
    max_nodes: Optional[int] = None,
    max_successors_per_node: Optional[int] = None,
) -> nx.DiGraph:
    """
    Build the arithmetic game DAG up to `steps` expansions.

    The traversal can be bounded by:
      - `max_nodes`: hard cap on total distinct polynomials.
      - `max_successors_per_node`: deterministic cap on how many operand choices
        each node expands against per step (helps tame blow-up for large graphs).
    """
    G = nx.DiGraph()

    if num_vars <= 0:
        raise ValueError("num_vars must be >= 1")

    start_exprs = (
        [symbols("x")]
        if num_vars == 1
        else list(symbols(f"x0:{num_vars}"))
    )

    start_nodes: list[str] = []
    for expr in start_exprs:
        key = canon_key(expr)
        if key in G:
            continue
        G.add_node(
            key,
            expr=expr,
            key=key,
            step=0,
            label=str(expr),
        )
        start_nodes.append(key)

    levels: Dict[int, list[str]] = {0: start_nodes}
    all_seen_by_step: Dict[int, list[str]] = {0: start_nodes}

    op_cache: Dict[Tuple[str, str, str], Tuple[str, Any]] = {}
    for step in range(steps):
        next_level: list[str] = []
        operand_pool: list[str] = []
        for s in range(step + 1):
            operand_pool.extend(all_seen_by_step.get(s, []))

        for node_id in levels.get(step, []):
            node_expr = G.nodes[node_id]["expr"]
            operands_iterable = operand_pool
            if max_successors_per_node is not None:
                operands_iterable = operand_pool[:max_successors_per_node]

            for operand_id in operands_iterable:
                operand_expr = G.nodes[operand_id]["expr"]

                for op_name, op_fn, edge_label in (
                    ("add", lambda a, b: expand(a + b), "+"),
                    ("mul", lambda a, b: expand(a * b), "*"),
                ):
                    cache_key = (op_name, *sorted((node_id, operand_id)))
                    cached = op_cache.get(cache_key)

                    if cached:
                        expr_key, new_expr = cached
                    else:
                        new_expr = op_fn(node_expr, operand_expr)
                        expr_key = canon_key(new_expr)
                        op_cache[cache_key] = (expr_key, new_expr)

                    if expr_key not in G:
                        G.add_node(
                            expr_key,
                            expr=new_expr,
                            key=expr_key,
                            step=step + 1,
                            label=pretty_label(new_expr),
                        )
                        next_level.append(expr_key)
                    G.add_edge(node_id, expr_key, op=op_name, operand=operand_id, label=edge_label)

                    if max_nodes is not None and G.number_of_nodes() >= max_nodes:
                        break
                if max_nodes is not None and G.number_of_nodes() >= max_nodes:
                    break
            if max_nodes is not None and G.number_of_nodes() >= max_nodes:
                break

        if max_nodes is not None and G.number_of_nodes() >= max_nodes:
            break

        if not next_level:
            break

        levels[step + 1] = next_level
        all_seen_by_step.setdefault(step + 1, [])
        combined = all_seen_by_step.get(step + 1, []) + next_level
        all_seen_by_step[step + 1] = list(dict.fromkeys(combined))

    G.remove_edges_from(nx.selfloop_edges(G))
    return G


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------

def _sanitize_value(value: Any) -> Any:
    """Convert SymPy objects into GraphML/JSON-safe primitives."""
    try:
        if hasattr(value, "free_symbols"):
            return str(expand(value))
    except Exception:
        pass

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def sanitize_graph(G: nx.DiGraph) -> nx.DiGraph:
    """Strip SymPy objects from node/edge attributes for serialization."""
    sanitized = nx.DiGraph()
    sanitized.graph.update({k: _sanitize_value(v) for k, v in G.graph.items()})

    for node_id, data in G.nodes(data=True):
        sanitized.add_node(node_id)
        for key, value in data.items():
            sanitized.nodes[node_id][key] = _sanitize_value(value)

    for u, v, data in G.edges(data=True):
        sanitized.add_edge(u, v)
        for key, value in data.items():
            sanitized.edges[u, v][key] = _sanitize_value(value)

    return sanitized


def save_graph_files(G: nx.DiGraph, base_path: Path) -> Tuple[Path, Path]:
    """Write sanitized GraphML and node-link JSON files."""
    sanitized = sanitize_graph(G)
    graphml_path = base_path.with_suffix(".graphml")
    json_path = base_path.with_suffix(".json")

    graphml_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    nx.write_graphml(sanitized, graphml_path)

    node_link = nx.node_link_data(sanitized)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(node_link, handle, indent=2)

    return graphml_path, json_path


def plot_graph(
    G: nx.DiGraph,
    png_path: Path,
    with_labels: bool = False,
    with_arrows: bool = False,
) -> None:
    """Render a layered layout of the DAG."""
    for _, data in G.nodes(data=True):
        data.setdefault("step", 0)

    pos = nx.multipartite_layout(G, subset_key="step", scale=2.0)

    plt.figure(figsize=(12, 7))
    nx.draw_networkx_nodes(G, pos, node_size=300)

    if with_labels:
        labels = {n: d.get("label", str(n)) for n, d in G.nodes(data=True)}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)

    nx.draw_networkx_edges(
        G,
        pos,
        arrows=with_arrows,
        arrowstyle="-|>" if with_arrows else "-",
        arrowsize=10,
    )

    plt.axis("off")
    plt.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close()


def write_graph_jsonl(G: nx.DiGraph, nodes_path: Path, edges_path: Path) -> None:
    """Dump node and edge metadata to JSONL files."""
    nodes_path.parent.mkdir(parents=True, exist_ok=True)
    edges_path.parent.mkdir(parents=True, exist_ok=True)

    with nodes_path.open("w", encoding="utf-8") as node_file:
        for node_id, data in G.nodes(data=True):
            expr = data.get("expr")
            expr_str = str(expand(expr)) if expr is not None else data.get("expr_str")
            record = {
                "id": node_id,
                "key": data.get("key", node_id),
                "step": data.get("step"),
                "label": data.get("label"),
                "expr_str": expr_str,
            }
            node_file.write(json.dumps(record) + "\n")

    with edges_path.open("w", encoding="utf-8") as edge_file:
        for u, v, data in G.edges(data=True):
            record = {
                "source": u,
                "target": v,
                "op": data.get("op"),
                "label": data.get("label")
                or ("+" if data.get("op") == "add" else "*" if data.get("op") == "mul" else None),
                "operand": data.get("operand"),
            }
            edge_file.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# Path analysis
# ---------------------------------------------------------------------------

@dataclass
class NodeRecord:
    node_id: str
    expr_str: Optional[str]
    step: Optional[int]
    label: Optional[str]


def build_analysis_structures(
    G: nx.DiGraph,
    max_step: Optional[int] = None,
) -> Tuple[Dict[str, NodeRecord], Dict[str, Set[str]], Dict[str, int]]:
    """Convert the NetworkX graph into lightweight adjacency structures."""
    nodes: Dict[str, NodeRecord] = {}
    forward: Dict[str, Set[str]] = defaultdict(set)
    in_degree: Dict[str, int] = defaultdict(int)

    for node_id, data in G.nodes(data=True):
        if max_step is not None and data.get("step") is not None and data["step"] > max_step:
            continue
        expr = data.get("expr")
        expr_str = str(expand(expr)) if expr is not None else data.get("expr_str")
        nodes[node_id] = NodeRecord(
            node_id=node_id,
            expr_str=expr_str,
            step=data.get("step"),
            label=data.get("label"),
        )

    for source, target in G.edges():
        if source not in nodes or target not in nodes:
            continue
        forward[source].add(target)
        in_degree[target] += 1
        in_degree.setdefault(source, 0)

    for node_id in nodes:
        forward.setdefault(node_id, set())
        in_degree.setdefault(node_id, 0)

    return nodes, forward, in_degree


def topological_sort(
    graph: Dict[str, Set[str]],
    in_degree: Dict[str, int],
    nodes: Dict[str, NodeRecord],
) -> List[str]:
    """Deterministic topological ordering grouped by (step, id)."""
    queue: deque[str] = deque(
        sorted(
            (nid for nid, deg in in_degree.items() if deg == 0),
            key=lambda nid: (
                nodes[nid].step if nodes[nid].step is not None else math.inf,
                nid,
            ),
        )
    )

    remaining = dict(in_degree)
    order: List[str] = []

    while queue:
        node_id = queue.popleft()
        order.append(node_id)
        for neighbor in graph[node_id]:
            remaining[neighbor] -= 1
            if remaining[neighbor] == 0:
                queue.append(neighbor)

    if len(order) != len(graph):
        raise ValueError("Graph contains a cycle, analysis aborted.")

    return order


def find_roots(nodes: Dict[str, NodeRecord], in_degree: Dict[str, int]) -> Set[str]:
    """Roots are in-degree zero; fallback to global min step."""
    min_step = math.inf
    for node in nodes.values():
        if node.step is not None:
            min_step = min(min_step, node.step)

    roots: Set[str] = set()
    for node_id, node in nodes.items():
        if in_degree.get(node_id, 0) == 0:
            roots.add(node_id)
        elif node.step is not None and node.step == min_step:
            roots.add(node_id)
    return roots


def collect_shortest_path_samples(
    node_id: str,
    roots: Set[str],
    predecessors: Dict[str, Set[str]],
    limit: int,
    order_hint: Dict[str, int],
) -> List[List[str]]:
    """Enumerate up to `limit` shortest paths ending at `node_id`."""
    if limit <= 0:
        return []

    samples: List[List[str]] = []
    path: List[str] = []

    def dfs(current: str) -> None:
        if len(samples) >= limit:
            return
        path.append(current)
        if current in roots or not predecessors[current]:
            samples.append(list(reversed(path)))
            path.pop()
            return
        next_nodes = sorted(
            predecessors[current],
            key=lambda nid: (order_hint.get(nid, math.inf), nid),
        )
        for nxt in next_nodes:
            dfs(nxt)
            if len(samples) >= limit:
                break
        path.pop()

    dfs(node_id)
    return samples


def analyze_graph(
    G: nx.DiGraph,
    max_samples: int,
    only_multipath: bool,
    max_step: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Run the path multiplicity analysis."""
    nodes, forward, in_degree = build_analysis_structures(G, max_step=max_step)
    order = topological_sort(forward, in_degree, nodes)
    roots = find_roots(nodes, in_degree)

    dist: Dict[str, float] = {node_id: math.inf for node_id in nodes}
    shortest_count: Dict[str, int] = {node_id: 0 for node_id in nodes}
    total_paths: Dict[str, int] = {node_id: 0 for node_id in nodes}
    shortest_preds: Dict[str, Set[str]] = {node_id: set() for node_id in nodes}

    for root in roots:
        dist[root] = 0
        shortest_count[root] = 1
        total_paths[root] = 1

    for node_id in order:
        if total_paths[node_id] == 0:
            continue
        node_dist = dist[node_id]
        node_shortest = shortest_count[node_id]
        node_total = total_paths[node_id]
        for neighbor in forward[node_id]:
            total_paths[neighbor] += node_total
            candidate_dist = node_dist + 1
            if candidate_dist < dist[neighbor]:
                dist[neighbor] = candidate_dist
                shortest_count[neighbor] = node_shortest
                shortest_preds[neighbor] = {node_id}
            elif candidate_dist == dist[neighbor] and candidate_dist != math.inf:
                shortest_count[neighbor] += node_shortest
                shortest_preds[neighbor].add(node_id)

    order_rank = {node_id: idx for idx, node_id in enumerate(order)}
    records: List[Dict[str, Any]] = []
    for node_id in order:
        node = nodes[node_id]
        shortest_length = dist[node_id]
        if math.isinf(shortest_length):
            shortest_length = None
        record: Dict[str, Any] = {
            "id": node.node_id,
            "expr_str": node.expr_str,
            "label": node.label,
            "step": node.step,
            "shortest_length": shortest_length,
            "shortest_path_count": shortest_count[node_id],
            "total_path_count": total_paths[node_id],
        }
        record["multiple_shortest_paths"] = shortest_count[node_id] > 1
        record["multiple_paths"] = total_paths[node_id] > 1

        include_samples = (
            max_samples > 0
            and record["shortest_length"] is not None
            and record["multiple_shortest_paths"]
        )
        if include_samples:
            samples = collect_shortest_path_samples(
                node_id,
                roots,
                shortest_preds,
                max_samples,
                order_rank,
            )
            record["shortest_path_samples"] = [
                {
                    "node_ids": path,
                    "expr_strs": [nodes[nid].expr_str for nid in path],
                }
                for path in samples
            ]

        if only_multipath and not (
            record["multiple_shortest_paths"] or record["multiple_paths"]
        ):
            continue

        records.append(record)

    return records


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate polynomial game boards and surface interesting nodes."
    )
    parser.add_argument(
        "--steps",
        "-C",
        type=int,
        default=MAX_ALLOWED_STEPS,
        help=f"Number of expansion steps (default: {MAX_ALLOWED_STEPS}).",
    )
    parser.add_argument("--num-vars", "-V", type=int, default=1, help="Number of variables to seed.")
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=None,
        help="Hard cap on the number of distinct polynomials generated.",
    )
    parser.add_argument(
        "--max-successors-per-node",
        type=int,
        default=None,
        help="Limit outgoing expansions per node per step to tame combinatorial blow-up.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="Base filename prefix (default: game_board_C<steps>).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory for all generated artifacts.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=5,
        help="Max stored shortest-path samples per node.",
    )
    parser.add_argument(
        "--only-multipath",
        action="store_true",
        help="Only emit nodes with multiple total or shortest paths.",
    )
    parser.add_argument(
        "--analysis-output",
        type=Path,
        default=None,
        help="Optional override for the analysis JSONL path.",
    )
    parser.add_argument(
        "--analysis-max-step",
        type=int,
        default=None,
        help="Only keep analysis records up to this step (matches RL complexity).",
    )
    parser.add_argument(
        "--with-labels",
        action="store_true",
        help="Plot node labels in the PNG visualization.",
    )
    parser.add_argument(
        "--with-arrows",
        action="store_true",
        help="Draw arrowheads in the PNG visualization.",
    )
    parser.add_argument(
        "--skip-plot",
        action="store_true",
        help="Skip PNG generation (useful on headless servers).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)

    effective_steps = min(args.steps, MAX_ALLOWED_STEPS)
    if args.steps > MAX_ALLOWED_STEPS:
        print(
            f"Requested steps {args.steps} exceed maximum of {MAX_ALLOWED_STEPS}; "
            f"capping to {MAX_ALLOWED_STEPS}."
        )

    prefix = args.prefix or f"game_board_C{effective_steps}"
    base_path = args.output_dir / prefix

    print(f"Building game board with C={effective_steps}, N={args.num_vars} ...")
    graph = build_game_graph(
        effective_steps,
        args.num_vars,
        max_nodes=args.max_nodes,
        max_successors_per_node=args.max_successors_per_node,
    )
    print(f"Built DAG: nodes={graph.number_of_nodes()}, edges={graph.number_of_edges()}")

    graphml_path, json_path = save_graph_files(graph, base_path)
    print(f"Saved GraphML to {graphml_path}")
    print(f"Saved node-link JSON to {json_path}")

    nodes_jsonl = base_path.with_suffix(".nodes.jsonl")
    edges_jsonl = base_path.with_suffix(".edges.jsonl")
    write_graph_jsonl(graph, nodes_jsonl, edges_jsonl)
    print(f"Wrote nodes JSONL to {nodes_jsonl}")
    print(f"Wrote edges JSONL to {edges_jsonl}")

    analysis_records = analyze_graph(
        graph,
        max_samples=args.max_samples,
        only_multipath=args.only_multipath,
        max_step=args.analysis_max_step,
    )
    analysis_path = args.analysis_output or base_path.with_suffix(".analysis.jsonl")
    analysis_path.parent.mkdir(parents=True, exist_ok=True)
    with analysis_path.open("w", encoding="utf-8") as handle:
        for record in analysis_records:
            handle.write(json.dumps(record) + "\n")
    print(f"Analysis JSONL saved to {analysis_path}")

    interesting = [
        rec for rec in analysis_records if rec["multiple_shortest_paths"] or rec["multiple_paths"]
    ]
    print(
        f"Found {len(interesting)} interesting polynomials "
        f"({len(analysis_records)} records total)."
    )

    if not args.skip_plot:
        png_path = base_path.with_suffix(".png")
        plot_graph(graph, png_path, with_labels=args.with_labels, with_arrows=args.with_arrows)
        print(f"Saved layered plot to {png_path}")


if __name__ == "__main__":
    main()
