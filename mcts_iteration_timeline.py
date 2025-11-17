#!/usr/bin/env python3
"""Generate an iteration-by-iteration timeline visualization for the MCTS demo."""

import argparse
import math
import random
import sys
from html import escape
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from mcts_demo import (
    CircuitGameState,
    DemoConfig,
    MCTS,
    decode_action,
    parse_target_polynomial,
)


def _format_float(value: float) -> str:
    if math.isinf(value):
        return "inf"
    return f"{value:.3f}"


class TracingMCTS(MCTS):
    def __init__(self, root_state: CircuitGameState, exploration_constant: float = 1.4):
        super().__init__(root_state, exploration_constant)
        self.iteration_logs: List[Dict[str, object]] = []

    def run(self, iterations: int) -> None:  # type: ignore[override]
        for idx in range(1, iterations + 1):
            node, selection_path, decisions = self._select_with_trace()
            reward, rollout_info = self._rollout_with_trace(node)
            backprop_info = self._backpropagate_with_trace(node, reward)
            trace = {
                "iteration": idx,
                "selection_path": selection_path,
                "selection_decisions": decisions,
                "reward": reward,
                "rollout": rollout_info,
                "terminal": node.state.is_terminal(),
                "backpropagation": backprop_info,
            }
            self.iteration_logs.append(trace)

    def _select_with_trace(self):
        node = self.root
        selection_path = [self._node_snapshot(node, "Root state")]
        decisions: List[Dict[str, object]] = []
        while not node.state.is_terminal():
            if node.untried_actions:
                action_idx = node.untried_actions.pop()
                next_state = node.state.clone()
                next_state.apply_action(action_idx)
                child = node.add_child(action_idx, next_state)
                decisions.append(self._decision_snapshot(node, child, action_idx, "expand"))
                node = child
                selection_path.append(self._node_snapshot(node, "Expanded new node"))
                return node, selection_path, decisions
            child = node.best_child(self.exploration_constant)
            decisions.append(self._decision_snapshot(node, child, None, "uct"))
            node = child
            selection_path.append(self._node_snapshot(node, "Selected via UCT"))
        return node, selection_path, decisions

    def _node_snapshot(self, node, note: str) -> Dict[str, object]:
        poly = node.state.config.format_polynomial(node.state.last_polynomial())
        return {
            "node_id": node.uid,
            "polynomial": poly,
            "visits": node.visit_count,
            "total_value": node.total_value,
            "mean_value": node.mean_value,
            "note": note,
        }

    def _decision_snapshot(self, parent, child, action_idx, decision_type: str) -> Dict[str, object]:
        snapshot: Dict[str, object] = {
            "type": decision_type,
            "from": parent.uid,
            "to": child.uid,
            "parent_visits": parent.visit_count,
            "child_visits": child.visit_count,
            "child_total": child.total_value,
            "child_mean": child.mean_value,
        }
        if decision_type == "uct":
            parent_visits = max(parent.visit_count, 1)
            if child.visit_count == 0:
                score = math.inf
            else:
                exploit = child.total_value / child.visit_count
                explore = math.sqrt(math.log(parent_visits) / child.visit_count)
                score = exploit + self.exploration_constant * explore
            snapshot.update({
                "uct_score": score,
                "exploration_constant": self.exploration_constant,
            })
        if action_idx is not None:
            op, lhs, rhs = decode_action(action_idx, parent.state.max_nodes)
            left_poly = parent.state.config.format_polynomial(parent.state.polynomials[lhs])
            right_poly = parent.state.config.format_polynomial(parent.state.polynomials[rhs])
            result_poly = child.state.history[-1]["result_poly"] if child.state.history else ""
            snapshot["action"] = {
                "operation": op,
                "operands": [lhs, rhs],
                "operands_poly": [left_poly, right_poly],
                "result_poly": result_poly,
            }
        return snapshot

    def _rollout_with_trace(self, node):
        rollout_state = node.state.clone()
        steps: List[Dict[str, object]] = []
        while not rollout_state.is_terminal():
            actions = rollout_state.available_actions()
            if not actions:
                break
            action_idx = random.choice(actions)
            op, lhs, rhs = decode_action(action_idx, rollout_state.max_nodes)
            left_poly = rollout_state.config.format_polynomial(rollout_state.polynomials[lhs])
            right_poly = rollout_state.config.format_polynomial(rollout_state.polynomials[rhs])
            rollout_state.apply_action(action_idx)
            result_poly = rollout_state.history[-1]["result_poly"] if rollout_state.history else ""
            steps.append({
                "operation": op,
                "operands": [lhs, rhs],
                "operands_poly": [left_poly, right_poly],
                "result_poly": result_poly,
            })
        reward = rollout_state.evaluate_reward()
        final_poly = rollout_state.config.format_polynomial(rollout_state.last_polynomial())
        return reward, {"steps": steps, "final_polynomial": final_poly}

    def _backpropagate_with_trace(self, node, reward: float) -> List[Dict[str, object]]:
        updates: List[Dict[str, object]] = []
        current = node
        while current is not None:
            before_visits = current.visit_count
            before_value = current.total_value
            current.update(reward)
            updates.append({
                "node_id": current.uid,
                "polynomial": current.state.config.format_polynomial(current.state.last_polynomial()),
                "before_visits": before_visits,
                "before_value": before_value,
                "after_visits": current.visit_count,
                "after_value": current.total_value,
                "mean_value": current.mean_value,
            })
            current = current.parent
        return updates


def build_iteration_section(log: Dict[str, object]) -> str:
    iteration = log["iteration"]
    reward = log["reward"]
    rollout_info = log["rollout"]
    selection_path = log["selection_path"]
    decisions = log["selection_decisions"]
    backprop = log["backpropagation"]

    selection_items = []
    for step in selection_path:
        note = f" — {escape(step['note'])}" if step.get("note") else ""
        poly = escape(step["polynomial"] or "Start")
        selection_items.append(
            f"<li><strong>Node #{step['node_id']}:</strong> {poly} (visits={step['visits']}, mean={_format_float(step['mean_value'])}){note}</li>"
        )
    selection_html = "\n".join(selection_items) if selection_items else "<li>No selection data.</li>"

    decision_items = []
    for entry in decisions:
        if entry["type"] == "uct":
            if math.isinf(entry.get("uct_score", math.inf)):
                desc = (
                    f"Selected child node #{entry['to']} with UCT treated as infinity because it has no visits yet."
                )
            else:
                desc = (
                    f"Selected child node #{entry['to']} using UCT = {_format_float(entry['uct_score'])} "
                    f"(Q={_format_float(entry['child_mean'])}, c={_format_float(entry['exploration_constant'])}, "
                    f"N_parent={entry['parent_visits']}, N_child={entry['child_visits']})."
                )
            decision_items.append(f"<li><strong>UCT step:</strong> {escape(desc)}</li>")
        else:
            action = entry.get("action", {})
            operands = ", ".join(str(x) for x in action.get("operands", []))
            left = escape(action.get("operands_poly", ["", ""])[0]) if action.get("operands_poly") else ""
            right = escape(action.get("operands_poly", ["", ""])[1]) if action.get("operands_poly") else ""
            result = escape(action.get("result_poly", ""))
            op_name = action.get("operation", "")
            operation = escape(op_name)
            symbol = "+" if op_name == "add" else "*"
            decision_items.append(
                "<li><strong>Expansion:</strong> Action "
                f"{operation}({operands}) produced {result} from {left} {symbol} {right}.</li>"
            )
    decision_html = "\n".join(decision_items) if decision_items else "<li>No tree policy decisions (node already terminal).</li>"

    rollout_steps = rollout_info.get("steps", []) if isinstance(rollout_info, dict) else []
    rollout_items = []
    for idx, step in enumerate(rollout_steps, start=1):
        op_name = step.get("operation", "")
        op = escape(op_name)
        operands = ", ".join(str(x) for x in step.get("operands", []))
        left = escape(step.get("operands_poly", ["", ""])[0]) if step.get("operands_poly") else ""
        right = escape(step.get("operands_poly", ["", ""])[1]) if step.get("operands_poly") else ""
        result = escape(step.get("result_poly", ""))
        symbol = "+" if op_name == "add" else "*"
        rollout_items.append(
            f"<li><strong>Step {idx}:</strong> {op}({operands}) => {left} {symbol} {right} = {result}</li>"
        )
    if rollout_items:
        rollout_items.append(
            f"<li><strong>Final rollout polynomial:</strong> {escape(rollout_info.get('final_polynomial', ''))}</li>"
        )
    else:
        final_poly = escape(rollout_info.get("final_polynomial", "")) if isinstance(rollout_info, dict) else ""
        rollout_items.append(
            f"<li>No rollout actions; state already terminal. Final polynomial: {final_poly}</li>"
        )
    rollout_html = "\n".join(rollout_items)

    backprop_items = []
    for entry in backprop:
        before_v = entry["before_visits"]
        after_v = entry["after_visits"]
        before_val = _format_float(entry["before_value"])
        after_val = _format_float(entry["after_value"])
        mean_val = _format_float(entry["mean_value"])
        poly = escape(entry["polynomial"] or "Start")
        backprop_items.append(
            f"<li><strong>Node #{entry['node_id']}:</strong> visits {before_v}→{after_v}, total value {before_val}→{after_val}, mean={mean_val} ({poly})</li>"
        )
    backprop_html = "\n".join(backprop_items)

    reward_text = _format_float(reward)
    terminal_note = " (terminal before rollout)" if log.get("terminal") else ""

    return f"""
<div class=\"timeline-item\">
  <div class=\"timeline-marker\"></div>
  <div class=\"timeline-content\">
    <details>
      <summary>Iteration {iteration}: reward {reward_text}{terminal_note}</summary>
      <div class=\"stage\">
        <h4>Selection (Tree Policy / UCT)</h4>
        <ul class=\"stage-list\">
{selection_html}
        </ul>
        <h5>Decisions</h5>
        <ul class=\"stage-list\">
{decision_html}
        </ul>
      </div>
      <div class=\"stage\">
        <h4>Simulation (Rollout)</h4>
        <ul class=\"stage-list\">
{rollout_html}
        </ul>
      </div>
      <div class=\"stage\">
        <h4>Backpropagation</h4>
        <ul class=\"stage-list\">
{backprop_html}
        </ul>
      </div>
    </details>
  </div>
</div>"""


def build_timeline_html(logs: List[Dict[str, object]]) -> str:
    sections = [build_iteration_section(log) for log in logs]
    return "\n".join(sections)


def build_html_document(summary: Dict[str, object], logs: List[Dict[str, object]]) -> str:
    timeline_html = build_timeline_html(logs)
    target = escape(summary["target_polynomial"])
    iterations = summary["iterations"]
    final_poly = escape(summary["final_polynomial"])
    final_match = "Yes" if summary["final_match"] else "No"
    final_mean = _format_float(summary["final_mean_value"])
    final_visits = summary["final_visits"]

    best_path_items = []
    for step in summary["best_path_steps"]:
        op_name = step["operation"]
        op = escape(op_name)
        operands = ", ".join(str(x) for x in step["operands"])
        operands_poly = step["operand_polys"]
        left = escape(operands_poly[0]) if operands_poly else ""
        right = escape(operands_poly[1]) if operands_poly else ""
        result = escape(step["result_poly"])
        symbol = "+" if op_name == "add" else "*"
        best_path_items.append(
            f"<li><strong>Step {step['step']}:</strong> {op}({operands}) => {left} {symbol} {right} = {result}</li>"
        )
    best_path_html = "\n".join(best_path_items) if best_path_items else "<li>No actions recorded.</li>"

    return f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <title>MCTS Iteration Timeline</title>
  <style>
    body {{ font-family: Arial, sans-serif; background: #f5f6fa; margin: 0; }}
    header {{ background: #1f2933; color: #fff; padding: 1.5rem 2rem; }}
    header h1 {{ margin: 0; font-size: 1.8rem; }}
    .container {{ max-width: 1100px; margin: 0 auto; padding: 2rem; }}
    .summary-card {{ background: #fff; border-radius: 6px; padding: 1.25rem 1.5rem; box-shadow: 0 2px 6px rgba(15,23,42,0.1); }}
    .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 1rem; margin-top: 1rem; }}
    .summary-item {{ background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 4px; padding: 0.9rem 1rem; }}
    .note {{ margin-top: 1.5rem; background: #e8f1ff; border-left: 4px solid #3b82f6; padding: 0.9rem 1.1rem; border-radius: 4px; color: #1f2937; }}
    .timeline {{ position: relative; margin-top: 2.5rem; padding-left: 2rem; }}
    .timeline::before {{ content: ""; position: absolute; left: 18px; top: 0; bottom: 0; width: 2px; background: #cbd5e1; }}
    .timeline-item {{ position: relative; margin-bottom: 1.8rem; }}
    .timeline-marker {{ position: absolute; left: -3px; top: 0.5rem; width: 14px; height: 14px; border-radius: 50%; background: #3b82f6; border: 2px solid #f5f6fa; box-shadow: 0 0 0 4px #f5f6fa; }}
    .timeline-content {{ margin-left: 2rem; background: #fff; border-radius: 6px; padding: 1rem 1.25rem; box-shadow: 0 2px 6px rgba(15,23,42,0.08); }}
    details {{ cursor: pointer; }}
    summary {{ font-weight: 600; font-size: 1.05rem; color: #0f172a; }}
    summary::marker {{ color: #64748b; }}
    .stage {{ margin-top: 1.1rem; }}
    .stage h4 {{ margin: 0 0 0.6rem 0; color: #1f2937; }}
    .stage h5 {{ margin: 0.8rem 0 0.45rem 0; font-size: 0.95rem; color: #334155; }}
    .stage-list {{ margin: 0; padding-left: 1.2rem; color: #334155; line-height: 1.45; }}
    .stage-list li {{ margin-bottom: 0.45rem; }}
    .best-path {{ margin-top: 2.5rem; background: #fff; border-radius: 6px; padding: 1.25rem 1.5rem; box-shadow: 0 2px 6px rgba(15,23,42,0.08); }}
    .best-path h2 {{ margin-top: 0; }}
    .best-path ul {{ padding-left: 1.2rem; line-height: 1.45; }}
    .footer-note {{ margin-top: 1.5rem; font-size: 0.9rem; color: #475569; }}
  </style>
</head>
<body>
  <header>
    <h1>MCTS Iteration Timeline</h1>
  </header>
  <div class=\"container\">
    <div class=\"summary-card\">
      <h2>Run Summary</h2>
      <div class=\"summary-grid\">
        <div class=\"summary-item\"><strong>Target:</strong><br />{target}</div>
        <div class=\"summary-item\"><strong>Iterations:</strong><br />{iterations}</div>
        <div class=\"summary-item\"><strong>Best leaf matches target:</strong><br />{final_match}</div>
        <div class=\"summary-item\"><strong>Best leaf mean value:</strong><br />{final_mean}</div>
        <div class=\"summary-item\"><strong>Best leaf visits:</strong><br />{final_visits}</div>
        <div class=\"summary-item\"><strong>Best leaf polynomial:</strong><br />{final_poly}</div>
      </div>
      <div class=\"note\">
        The tree policy relies on the Upper Confidence Bound for Trees (UCT):
        <code>UCT = Q + c * sqrt(ln(N_parent) / N_child)</code>, where <code>Q</code> is the mean value of a child,
        <code>c</code> is the exploration constant, and <code>N</code> counts visits. Each iteration follows four phases:
        selection via UCT, expansion of a new action, a random rollout (simulation), and value backpropagation.
      </div>
    </div>
    <div class=\"timeline\">
{timeline_html}
    </div>
    <div class=\"best-path\">
      <h2>Best Path After Search</h2>
      <ul>
{best_path_html}
      </ul>
    </div>
    <p class=\"footer-note\">Use the disclosure triangles to expand or collapse iteration details. The timeline shows every formula the planner evaluated during the run without overlapping text.</p>
  </div>
</body>
</html>"""


def run_timeline_demo(
    config: DemoConfig,
    iterations: int,
    seed: int,
    target_expr: str,
    output_path: Path,
) -> None:
    random.seed(seed)
    target_poly = parse_target_polynomial(target_expr, config)
    initial_state = CircuitGameState(target_poly=target_poly, config=config)

    tracer = TracingMCTS(initial_state)
    tracer.run(iterations)

    _, _, best_leaf = tracer.extract_best_path()

    best_path_steps: List[Dict[str, object]] = []
    for idx, step in enumerate(best_leaf.state.history, start=1):
        best_path_steps.append(
            {
                "step": idx,
                "operation": step["operation"],
                "operands": list(step["operands"]),
                "operand_polys": list(step["operand_polys"]),
                "result_poly": step["result_poly"],
            }
        )

    final_poly_display = config.format_polynomial(best_leaf.state.last_polynomial())
    summary = {
        "target_polynomial": config.format_polynomial(target_poly),
        "iterations": iterations,
        "final_match": bool(best_leaf.state.matches_target()),
        "final_mean_value": float(best_leaf.mean_value),
        "final_visits": int(best_leaf.visit_count),
        "final_polynomial": final_poly_display,
        "best_path_steps": best_path_steps,
    }

    html_content = build_html_document(summary, tracer.iteration_logs)
    output_path.write_text(html_content, encoding="utf-8")

    print(f"Target polynomial   : {config.format_polynomial(target_poly)}")
    print(f"MCTS iterations     : {iterations}")
    print(f"Timeline saved to   : {output_path}")
    print(f"Best leaf polynomial: {final_poly_display}")
    print(f"Matches target      : {'yes' if summary['final_match'] else 'no'}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the MCTS demo and produce an iteration-by-iteration timeline visualization.",
    )
    parser.add_argument("--iterations", type=int, default=20, help="Number of MCTS iterations to perform")
    parser.add_argument(
        "--max-complexity",
        type=int,
        default=6,
        help="Maximum number of arithmetic operations allowed when building the circuit",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument(
        "--output",
        type=str,
        default="mcts_timeline.html",
        help="Output path for the iteration timeline visualization",
    )
    parser.add_argument("--variables", type=int, default=2, help="Number of base variables available to the circuit builder")
    parser.add_argument(
        "--target",
        type=str,
        default="x**2 + 2*x*y + y**2",
        help="Target polynomial expression written with the chosen variable names",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    config = DemoConfig(n_variables=args.variables, max_complexity=args.max_complexity)
    output_path = Path(args.output)
    run_timeline_demo(config, args.iterations, args.seed, args.target, output_path)


if __name__ == "__main__":
    main(sys.argv[1:])
