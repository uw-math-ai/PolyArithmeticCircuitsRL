"""
Utility helpers for turning arithmetic-circuit actions into visual artifacts.

The functions here are intentionally independent of Streamlit so they can be
tested and reused from notebooks, scripts, or the demo app.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import sympy

Action = Tuple[str, int | None, int | None]


@dataclass
class CircuitGraph:
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]


NODE_STYLES = {
    "input": {"color": "#2563eb", "shape": "box"},
    "constant": {"color": "#6b7280", "shape": "ellipse"},
    "add": {"color": "#22c55e", "shape": "dot"},
    "multiply": {"color": "#f59e0b", "shape": "dot"},
}


def _poly_to_str(expr: sympy.Expr) -> str:
    try:
        return str(sympy.simplify(expr))
    except Exception:
        return str(expr)


def circuit_to_graph_data(actions: List[Action], polynomials: List[sympy.Expr]) -> CircuitGraph:
    """
    Convert the action/polynomial lists into a simple graph description that
    downstream renderers (PyVis, Graphviz, etc.) can consume.
    """
    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []

    for idx, (op, in1, in2) in enumerate(actions):
        poly_str = _poly_to_str(polynomials[idx]) if idx < len(polynomials) else op
        label = f"n{idx}: {poly_str}"
        title = {
            "input": f"Input variable x{in1}",
            "constant": "Constant 1",
            "add": "Addition node",
            "multiply": "Multiplication node",
        }.get(op, op)

        nodes.append(
            {
                "id": idx,
                "label": label,
                "polynomial": poly_str,
                "type": op,
                "title": title,
            }
        )

        if op in ("add", "multiply"):
            if in1 is not None and in1 >= 0:
                edges.append({"source": in1, "target": idx, "type": op})
            if in2 is not None and in2 >= 0:
                edges.append({"source": in2, "target": idx, "type": op})

    return CircuitGraph(nodes=nodes, edges=edges)


def render_circuit_html(
    actions: List[Action],
    polynomials: List[sympy.Expr],
    height: str = "640px",
    width: str = "100%",
) -> str:
    """
    Build an interactive HTML snippet for the circuit graph using PyVis.
    """
    try:
        from pyvis.network import Network
    except ImportError as exc:
        raise RuntimeError(
            "pyvis is required to render circuit HTML. Install with `pip install pyvis`."
        ) from exc

    graph = circuit_to_graph_data(actions, polynomials)
    net = Network(height=height, width=width, directed=True)

    for node in graph.nodes:
        style = NODE_STYLES.get(node["type"], {})
        net.add_node(
            node["id"],
            label=node["label"],
            title=node["title"],
            color=style.get("color"),
            shape=style.get("shape", "ellipse"),
        )

    for edge in graph.edges:
        color = "#a3a3a3" if edge["type"] == "add" else "#fb7185"
        net.add_edge(edge["source"], edge["target"], arrows="to", color=color)

    # We rely on PyVis' built-in HTML generator and return the string for embedding.
    net.generate_html()
    return net.html


def summarize_actions(actions: List[Action], polynomials: List[sympy.Expr]) -> List[Dict[str, Any]]:
    """
    Produce a concise, table-friendly summary of each step in the circuit.
    """
    summary: List[Dict[str, Any]] = []
    for idx, (op, in1, in2) in enumerate(actions):
        inputs = "-"
        if op == "input":
            inputs = f"x{in1}"
        elif op == "constant":
            inputs = "1"
        elif op in ("add", "multiply"):
            inputs = f"n{in1}, n{in2}"

        poly_str = _poly_to_str(polynomials[idx]) if idx < len(polynomials) else ""
        summary.append(
            {
                "step": idx,
                "op": op,
                "inputs": inputs,
                "polynomial": poly_str,
            }
        )
    return summary
