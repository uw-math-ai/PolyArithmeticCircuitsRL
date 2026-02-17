#!/usr/bin/env python3
"""
Build and plot the single-variable polynomial "game board" DAG.

Start state is x. From a current polynomial P, you may create a successor
by either P + S or P * S where S is ANY previously seen polynomial (in the
global DAG up to that point). We deduplicate states by canonical expansion.
The graph is acyclic and layered by "step" (distance from start).

Outputs (prefix defaults to game_board_C<steps>):
  - <prefix>.graphml : GraphML (sanitized: only primitives/strings)
  - <prefix>.json    : node-link JSON (sanitized)
  - <prefix>.png     : layered plot (labels/arrows optional)

Examples:
  python build_game_board.py --steps 3
  python build_game_board.py --steps 3 --with-labels --with-arrows
  python build_game_board.py --steps 4 --prefix my_board
"""

from __future__ import annotations

import argparse
import json
from typing import Tuple, Dict, Any

import networkx as nx
import matplotlib.pyplot as plt
from sympy import symbols, expand, srepr

# ------------------------------
# SymPy basics and canon helpers
# ------------------------------

def canon_key(expr) -> str:
    """Canonical, hashable string for a polynomial expression."""
    # Expand and use SymPy's S-expression repr for stability.
    return srepr(expand(expr))

def pretty_label(expr, max_len: int = 32) -> str:
    """Short label for plotting."""
    s = str(expand(expr))
    return s if len(s) <= max_len else (s[: max_len - 1] + "…")

# ------------------------------
# Graph construction
# ------------------------------
def build_game_graph(C: int, num_vars: int = 1) -> nx.DiGraph:
    """
    Build the game DAG up to C steps.
    Nodes use their canonical key as id. Node attrs:
        - expr  : SymPy expression (IN-MEMORY ONLY; not export-safe)
        - key   : canonical string (same as node id)
        - step  : int, distance from start
        - label : short string version of expr
    Edge attrs:
        - op      : 'add' | 'mul'
        - operand : node id (string) of the operand S
    """
    G = nx.DiGraph()

    # Seed
    if num_vars == 1:
        start_exprs = [symbols('x')]
    else:
        start_exprs = list(symbols(f'x0:{num_vars}'))

    start_nodes = []
    for expr in start_exprs:
        key = canon_key(expr)
        if key not in G:
            G.add_node(key, expr=expr, key=key, step=0, label=str(expr))
            start_nodes.append(key)

    # Layered BFS by "step"
    levels: Dict[int, list[str]] = {0: start_nodes}
    all_seen_by_step: Dict[int, list[str]] = {0: start_nodes}

    for t in range(C):
        next_level: list[str] = []

        # Operands available at this step: everything seen up to step t
        operand_pool: list[str] = []
        for s in range(t + 1):
            operand_pool.extend(all_seen_by_step.get(s, []))

        # Expand successors for current layer
        for u in levels.get(t, []):
            u_expr = G.nodes[u]['expr']
            for w in operand_pool:
                w_expr = G.nodes[w]['expr']

                # Addition
                add_expr = expand(u_expr + w_expr)
                add_key = canon_key(add_expr)
                if add_key not in G:
                    G.add_node(add_key, expr=add_expr, key=add_key, step=t+1, label=pretty_label(add_expr))
                    next_level.append(add_key)
                G.add_edge(u, add_key, op='add', operand=w)

                # Multiplication
                mul_expr = expand(u_expr * w_expr)
                mul_key = canon_key(mul_expr)
                if mul_key not in G:
                    G.add_node(mul_key, expr=mul_expr, key=mul_key, step=t+1, label=pretty_label(mul_expr))
                    next_level.append(mul_key)
                G.add_edge(u, mul_key, op='mul', operand=w)

        if next_level:
            levels[t+1] = next_level
            all_seen_by_step.setdefault(t+1, [])
            # Dedup next-level record
            all_seen_by_step[t+1] = list(set(all_seen_by_step.get(t+1, []) + next_level))
        else:
            # Reached a fixed point (rare here, but safe to short-circuit)
            break

    # Defensive: remove self-loops if any arise
    G.remove_edges_from(nx.selfloop_edges(G))
    return G

# ------------------------------
# Export: sanitize for GraphML/JSON
# ------------------------------
def _sanitize_value(v: Any) -> Any:
    """
    Convert values to GraphML/JSON-safe primitives.
    - SymPy expr => string(str(expand(expr)))
    - Everything else: keep ints/floats/bools/None/strings; coerce others to str
    """
    # SymPy expressions have .free_symbols attribute; safer to just stringify unknowns
    try:
        # detect SymPy by duck-typing: anything with 'free_symbols' and 'as_coefficients_dict'
        if hasattr(v, 'free_symbols') or v.__class__.__module__.startswith('sympy'):
            from sympy import expand
            return str(expand(v))
    except Exception:
        pass

    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    return str(v)

def sanitize_graph(G: nx.DiGraph) -> nx.DiGraph:
    """
    Return a copy of G where all node/edge attributes are GraphML/JSON-safe.
    Keeps node ids identical (already strings).
    """
    H = nx.DiGraph()
    # Graph-level attrs (rarely used): sanitize too
    H.graph.update({k: _sanitize_value(v) for k, v in G.graph.items()})

    for n, d in G.nodes(data=True):
        H.add_node(n)
        # Strip or convert SymPy attrs to strings
        for k, v in d.items():
            H.nodes[n][k] = _sanitize_value(v)

    for u, v, d in G.edges(data=True):
        H.add_edge(u, v)
        for k, v2 in d.items():
            H.edges[u, v][k] = _sanitize_value(v2)

    return H

def save_graph_files(G: nx.DiGraph, prefix: str) -> Tuple[str, str]:
    """
    Save sanitized GraphML and node-link JSON so NetworkX and downstream tools
    don’t choke on SymPy objects.
    """
    H = sanitize_graph(G)

    graphml_path = f"{prefix}.graphml"
    json_path = f"{prefix}.json"

    # GraphML
    nx.write_graphml(H, graphml_path)

    # JSON (node-link)
    data = nx.node_link_data(H)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    return graphml_path, json_path

# ------------------------------
# Plotting
# ------------------------------
def plot_graph(G: nx.DiGraph, prefix: str, with_labels: bool = False, with_arrows: bool = False) -> str:
    """
    Layered plot by 'step' using NetworkX multipartite layout.
    For larger graphs, prefer no labels and no arrows for speed and clarity.
    """
    # Ensure 'step' exists
    for _, d in G.nodes(data=True):
        d.setdefault('step', 0)

    pos = nx.multipartite_layout(G, subset_key="step", scale=2.0)

    plt.figure(figsize=(12, 7))
    nx.draw_networkx_nodes(G, pos, node_size=300)

    if with_labels:
        labels = {n: d.get('label', str(n)) for n, d in G.nodes(data=True)}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)

    nx.draw_networkx_edges(
        G, pos,
        arrows=with_arrows,
        arrowstyle='-|>' if with_arrows else '-',
        arrowsize=10 if with_arrows else 10
    )

    # Edge labels (op) are often cluttered; uncomment if needed for small graphs.
    # edge_labels = {(u, v): d.get('op', '') for u, v, d in G.edges(data=True)}
    # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)

    plt.axis('off')
    plt.tight_layout()
    out_png = f"{prefix}.png"
    plt.savefig(out_png, dpi=200, bbox_inches='tight')
    print(f"Saved plot to {out_png}")
    return out_png

# ------------------------------
# CLI
# ------------------------------
def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--steps', '-C', type=int, required=True,
                   help='Max number of steps from the start node.')
    p.add_argument('--num-vars', '-V', type=int, default=1,
                     help='Number of variables to use (default: 1).')
    p.add_argument('--prefix', type=str, default=None,
                   help='Output prefix (default: game_board_C<steps>).')
    p.add_argument('--with-labels', action='store_true',
                   help='Draw node labels (polynomial strings).')
    p.add_argument('--with-arrows', action='store_true',
                   help='Draw arrows on edges (slower).')
    args = p.parse_args(argv)

    C = args.steps
    prefix = args.prefix or f"game_board_C{C}"

    G = build_game_graph(C, args.num_vars)
    print(f"Built DAG: nodes={G.number_of_nodes()}, edges={G.number_of_edges()}")

    graphml_path, json_path = save_graph_files(G, prefix)
    print(f"Saved GraphML: {graphml_path}")
    print(f"Saved JSON   : {json_path}")

    plot_graph(G, prefix, with_labels=args.with_labels, with_arrows=args.with_arrows)

if __name__ == '__main__':
    main()