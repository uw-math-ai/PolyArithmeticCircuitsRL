"""Graph enumeration and path-multiplicity analysis for interesting polynomials.

Ported from Game-Board-Generation/interesting_polynomial_generator.py,
stripped of all I/O, plotting, and CLI code. Used by
GenerativeInterestingPolynomialSampler to auto-generate interesting
polynomials at training time.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
from sympy import expand, srepr, symbols


# ---------------------------------------------------------------------------
# Canonical SymPy helpers
# ---------------------------------------------------------------------------

def canon_key(expr) -> str:
    """Canonical, hashable string for a polynomial expression."""
    return srepr(expand(expr))


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_game_graph(
    steps: int,
    num_vars: int = 1,
    max_nodes: Optional[int] = None,
    max_successors_per_node: Optional[int] = None,
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
        G.add_node(key, expr=expr, key=key, step=0, label=str(expr))
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
                            label=str(expand(new_expr)),
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


def analyze_graph(
    G: nx.DiGraph,
    only_multipath: bool = True,
    max_step: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Run path-multiplicity analysis, returning records for interesting nodes.

    Parameters
    ----------
    G : nx.DiGraph
        Game-board graph from :func:`build_game_graph`.
    only_multipath : bool
        If True, only return nodes with multiple total or shortest paths.
    max_step : int | None
        Filter to nodes at or below this step.
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
        }

        if only_multipath and not (
            record["multiple_shortest_paths"] or record["multiple_paths"]
        ):
            continue

        records.append(record)

    return records
