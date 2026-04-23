"""Graph enumeration and path-multiplicity analysis for interesting polynomials.

Ported from Game-Board-Generation/interesting_polynomial_generator.py,
stripped of all I/O, plotting, and CLI code. Used by
GenerativeInterestingPolynomialSampler to auto-generate interesting
polynomials at training time.
"""

from __future__ import annotations

import math
import time
import warnings
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
from sympy import Poly as SympyPoly, expand, srepr, symbols


# ---------------------------------------------------------------------------
# Canonical SymPy helpers
# ---------------------------------------------------------------------------

def canon_key(expr) -> str:
    """Canonical, hashable string for a polynomial expression."""
    return srepr(expand(expr))


def estimate_naive_ops(expr, num_vars: int = 2) -> int:
    """Estimate operations for naive term-by-term polynomial construction.

    For each monomial with total degree d: max(0, d-1) multiplications.
    To combine n terms: n-1 additions.
    """
    expanded = expand(expr)
    var_syms = (
        [symbols("x")]
        if num_vars == 1
        else list(symbols(f"x0:{num_vars}"))
    )
    try:
        sp = SympyPoly(expanded, *var_syms)
    except Exception:
        return 0
    terms = sp.as_dict()
    if not terms:
        return 0
    mul_ops = sum(max(0, sum(monom) - 1) for monom in terms)
    add_ops = max(0, len(terms) - 1)
    return mul_ops + add_ops


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_game_graph(
    steps: int,
    num_vars: int = 1,
    max_nodes: Optional[int] = None,
    max_successors_per_node: Optional[int] = None,
    max_seconds: Optional[float] = None,
) -> nx.DiGraph:
    """Build the arithmetic game DAG up to *steps* expansions.

    Parameters
    ----------
    steps : int
        Number of expansion steps (complexity levels).
    num_vars : int
        Number of polynomial variables.
    max_nodes : int | None
        Hard cap on total distinct polynomials in the DAG.
    max_successors_per_node : int | None
        Per-node expansion cap to tame blow-up.
    max_seconds : float | None
        Optional wall-clock budget in seconds for graph construction.
    """
    G = nx.DiGraph()
    start_time = time.perf_counter()
    deadline = None if max_seconds is None else start_time + max_seconds
    timed_out = False

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
        expr_str = str(expr)
        G.add_node(key, expr=expr, expr_str=expr_str, key=key, step=0, label=expr_str)
        start_nodes.append(key)

    levels: Dict[int, list[str]] = {0: start_nodes}
    all_seen_by_step: Dict[int, list[str]] = {0: start_nodes}

    op_cache: Dict[Tuple[str, str, str], Tuple[str, Any]] = {}
    for step in range(steps):
        if deadline is not None and time.perf_counter() >= deadline:
            timed_out = True
            break
        next_level: list[str] = []
        operand_pool: list[str] = []
        for s in range(step + 1):
            operand_pool.extend(all_seen_by_step.get(s, []))

        for node_id in levels.get(step, []):
            if deadline is not None and time.perf_counter() >= deadline:
                timed_out = True
                break
            node_expr = G.nodes[node_id]["expr"]
            operands_iterable = operand_pool
            if max_successors_per_node is not None:
                operands_iterable = operand_pool[:max_successors_per_node]

            for operand_id in operands_iterable:
                if deadline is not None and time.perf_counter() >= deadline:
                    timed_out = True
                    break
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
                        # op_fn wraps `expand(...)`, so new_expr is already canonical-expanded;
                        # downstream readers of `expr` / `expr_str` must not re-expand.
                        new_expr = op_fn(node_expr, operand_expr)
                        expr_key = canon_key(new_expr)
                        op_cache[cache_key] = (expr_key, new_expr)

                    if expr_key not in G:
                        expr_str = str(new_expr)
                        G.add_node(
                            expr_key,
                            expr=new_expr,
                            expr_str=expr_str,
                            key=expr_key,
                            step=step + 1,
                            label=expr_str,
                        )
                        next_level.append(expr_key)
                    G.add_edge(
                        node_id, expr_key,
                        op=op_name, operand=operand_id, label=edge_label,
                    )

                    if max_nodes is not None and G.number_of_nodes() >= max_nodes:
                        break
                if max_nodes is not None and G.number_of_nodes() >= max_nodes:
                    break
            if max_nodes is not None and G.number_of_nodes() >= max_nodes:
                break
            if timed_out:
                break

        if max_nodes is not None and G.number_of_nodes() >= max_nodes:
            break
        if timed_out:
            break

        if not next_level:
            break

        levels[step + 1] = next_level
        all_seen_by_step.setdefault(step + 1, [])
        combined = all_seen_by_step.get(step + 1, []) + next_level
        all_seen_by_step[step + 1] = list(dict.fromkeys(combined))

    G.remove_edges_from(nx.selfloop_edges(G))
    if timed_out:
        warnings.warn(
            (
                "build_game_graph exceeded max_seconds="
                f"{max_seconds}; returning partial graph with {G.number_of_nodes()} nodes"
            ),
            stacklevel=2,
        )
    return G


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
        # `expr` is already canonical-expanded at construction time (see build_game_graph).
        expr_str = data.get("expr_str") or (str(expr) if expr is not None else None)
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


def analyze_graph(
    G: nx.DiGraph,
    only_multipath: bool = False,
    only_shortcut: bool = True,
    min_shortcut_gap: int = 2,
    max_step: Optional[int] = None,
    num_vars: int = 2,
) -> Tuple[List[Dict[str, Any]], Dict[str, float], Set[str]]:
    """Run path-multiplicity analysis, returning records for interesting nodes.

    Parameters
    ----------
    G : nx.DiGraph
        Game-board graph from :func:`build_game_graph`.
    only_multipath : bool
        If True, only return nodes with multiple total or shortest paths.
    only_shortcut : bool
        If True, only return nodes where naive_ops - shortest_length >= min_shortcut_gap.
    min_shortcut_gap : int
        Minimum gap between naive and optimal cost to qualify as a shortcut.
    max_step : int | None
        Filter to nodes at or below this step.
    num_vars : int
        Number of polynomial variables (for naive cost estimation).

    Returns
    -------
    records : list of dict
        Per-node analysis records (filtered by multipath/shortcut criteria).
    dist : dict
        Shortest distance from roots for every node in the graph.
    roots : set
        Root node IDs (in-degree zero / base variables).
    """
    nodes, forward, in_degree = build_analysis_structures(G, max_step=max_step)
    order = topological_sort(forward, in_degree, nodes)
    roots = find_roots(nodes, in_degree)

    dist: Dict[str, float] = {node_id: math.inf for node_id in nodes}
    shortest_count: Dict[str, int] = {node_id: 0 for node_id in nodes}
    total_paths: Dict[str, int] = {node_id: 0 for node_id in nodes}

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
            elif candidate_dist == dist[neighbor] and candidate_dist != math.inf:
                shortest_count[neighbor] += node_shortest

    records: List[Dict[str, Any]] = []
    for node_id in order:
        node = nodes[node_id]
        shortest_length = dist[node_id]
        if math.isinf(shortest_length):
            shortest_length = None

        # Compute naive cost and shortcut gap
        expr = G.nodes[node_id].get("expr") if node_id in G else None
        naive_ops = estimate_naive_ops(expr, num_vars) if expr is not None else 0
        shortcut_gap = (naive_ops - shortest_length) if shortest_length is not None else 0

        record: Dict[str, Any] = {
            "id": node.node_id,
            "expr_str": node.expr_str,
            "label": node.label,
            "step": node.step,
            "shortest_length": shortest_length,
            "shortest_path_count": shortest_count[node_id],
            "total_path_count": total_paths[node_id],
            "multiple_shortest_paths": shortest_count[node_id] > 1,
            "multiple_paths": total_paths[node_id] > 1,
            "naive_ops": naive_ops,
            "shortcut_gap": shortcut_gap,
            "has_shortcut": shortcut_gap >= min_shortcut_gap,
        }

        # Apply filters
        if only_shortcut and not record["has_shortcut"]:
            continue
        if only_multipath and not (
            record["multiple_shortest_paths"] or record["multiple_paths"]
        ):
            continue

        records.append(record)

    return records, dist, roots


def shortest_path_intermediates(
    G: nx.DiGraph,
    target_id: str,
    dist: Dict[str, float],
    roots: Set[str],
) -> Set[str]:
    """Return DAG node IDs on shortest paths from roots to *target_id*.

    Walks backward from the target, following only edges where
    ``dist[predecessor] + 1 == dist[node]`` (i.e. edges on a shortest
    path).  Returns intermediate node IDs, excluding roots and the
    target itself.
    """
    visited: Set[str] = set()
    queue: deque[str] = deque([target_id])
    while queue:
        v = queue.popleft()
        if v in visited:
            continue
        visited.add(v)
        for u in G.predecessors(v):
            if u in dist and dist[u] + 1 == dist[v]:
                queue.append(u)
    return visited - roots - {target_id}
