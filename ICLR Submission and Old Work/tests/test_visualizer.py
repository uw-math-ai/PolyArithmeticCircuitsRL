import os
import sys

import pytest
import sympy as sp

# Ensure project root is on the path for direct test invocation
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from visualization.circuit_visualizer import (
    circuit_to_graph_data,
    render_circuit_html,
    summarize_actions,
)


@pytest.fixture()
def sample_circuit():
    x0, x1 = sp.symbols("x0:2")
    actions = [
        ("input", 0, -1),      # n0 -> x0
        ("input", 1, -1),      # n1 -> x1
        ("constant", -1, -1),  # n2 -> 1
        ("add", 0, 1),         # n3 -> x0 + x1
        ("multiply", 3, 0),    # n4 -> (x0 + x1) * x0
    ]
    polynomials = [
        x0,
        x1,
        sp.Integer(1),
        sp.expand(x0 + x1),
        sp.expand((x0 + x1) * x0),
    ]
    return actions, polynomials


def test_circuit_to_graph_data_contains_edges(sample_circuit):
    actions, polynomials = sample_circuit
    graph = circuit_to_graph_data(actions, polynomials)

    assert len(graph.nodes) == len(actions)
    assert {"source": 0, "target": 3, "type": "add"} in graph.edges
    assert {"source": 3, "target": 4, "type": "multiply"} in graph.edges
    assert graph.nodes[3]["polynomial"] == "x0 + x1"


def test_summarize_actions_formats_rows(sample_circuit):
    actions, polynomials = sample_circuit
    rows = summarize_actions(actions, polynomials)

    assert rows[0]["inputs"] == "x0"
    assert rows[2]["inputs"] == "1"
    assert rows[-1]["op"] == "multiply"
    assert rows[-1]["polynomial"] == "x0*(x0 + x1)"


def test_render_circuit_html_returns_html(sample_circuit):
    pytest.importorskip("pyvis")
    actions, polynomials = sample_circuit
    html = render_circuit_html(actions, polynomials, height="200px")

    assert "<html" in html.lower()
    assert "x0 + x1" in html
