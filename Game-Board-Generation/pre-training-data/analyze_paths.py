#!/usr/bin/env python3
"""Analyze circuit graphs to find polynomials with multiple paths."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Optional, Sequence, Set


@dataclass
class Node:
    """Holds the minimal metadata we need for each circuit node."""
    node_id: str
    expr_str: Optional[str]
    step: Optional[int]
    label: Optional[str]


def parse_args() -> argparse.Namespace:
    """Build and parse CLI arguments controlling file paths and sampling options."""
    parser = argparse.ArgumentParser(
        description="Detect nodes that admit multiple shortest paths in a circuit DAG."
    )
    parser.add_argument(
        "--nodes",
        type=Path,
        required=True,
        help="Path to JSONL file describing nodes.",
    )
    parser.add_argument(
        "--edges",
        type=Path,
        required=True,
        help="Path to JSONL file describing edges.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to write analysis JSONL output.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=5,
        help="Maximum number of shortest-path samples to store per node.",
    )
    parser.add_argument(
        "--only-multipath",
        action="store_true",
        help="Only emit nodes that have multiple total paths or multiple shortest paths.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> Iterable[dict]:
    """Yield dictionaries for each non-empty JSONL line at the given path."""
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def build_nodes(path: Path) -> Dict[str, Node]:
    """Load node definitions and index them by identifier, filling optional fields when missing."""
    nodes: Dict[str, Node] = {}
    for entry in read_jsonl(path):
        node_id = entry.get("id") or entry.get("key")
        if not node_id:
            continue
        nodes[node_id] = Node(
            node_id=node_id,
            expr_str=entry.get("expr_str"),
            step=entry.get("step"),
            label=entry.get("label"),
        )
    return nodes


def build_graph(
    path: Path, nodes: Dict[str, Node]
) -> tuple[Dict[str, Set[str]], Dict[str, Set[str]], Dict[str, int]]:
    """Construct adjacency and reverse-adjacency maps plus in-degree counts from edge definitions."""
    forward: Dict[str, Set[str]] = defaultdict(set)
    reverse: Dict[str, Set[str]] = defaultdict(set)
    in_degree: Dict[str, int] = defaultdict(int)

    for entry in read_jsonl(path):
        source = entry.get("source")
        target = entry.get("target")
        if source is None or target is None:
            continue

        if source not in nodes:
            nodes[source] = Node(node_id=source, expr_str=None, step=None, label=None)
        if target not in nodes:
            nodes[target] = Node(node_id=target, expr_str=None, step=None, label=None)

        if target not in forward[source]:
            forward[source].add(target)
            reverse[target].add(source)
            in_degree[target] += 1
            in_degree.setdefault(source, 0)

    for node_id in nodes:
        forward.setdefault(node_id, set())
        reverse.setdefault(node_id, set())
        in_degree.setdefault(node_id, 0)

    return forward, reverse, in_degree


def topological_sort(
    graph: Dict[str, Set[str]], in_degree: Dict[str, int], nodes: Dict[str, Node]
) -> List[str]:
    """Return a deterministic topological order or raise if the graph is not a DAG."""
    queue: Deque[str] = deque(
        sorted(
            (node_id for node_id, deg in in_degree.items() if deg == 0),
            key=lambda nid: (
                nodes[nid].step if nodes[nid].step is not None else math.inf,
                nid,
            ),
        )
    )
    remaining_in_deg = dict(in_degree)
    order: List[str] = []

    while queue:
        node_id = queue.popleft()
        order.append(node_id)
        for neighbor in graph[node_id]:
            remaining_in_deg[neighbor] -= 1
            if remaining_in_deg[neighbor] == 0:
                queue.append(neighbor)

    if len(order) != len(graph):
        raise ValueError("Graph contains a cycle or disconnected component preventing DAG analysis.")

    return order


def find_roots(nodes: Dict[str, Node], in_degree: Dict[str, int]) -> Set[str]:
    """Identify starting nodes using in-degree zero or minimal step value as a fallback."""
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
    """Backtrack through stored predecessors to enumerate up to the requested shortest paths."""
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


def analyze(
    nodes_path: Path,
    edges_path: Path,
    output_path: Path,
    max_samples: int,
    only_multipath: bool,
) -> None:
    """Run the full analysis pipeline and stream JSONL output describing path multiplicities."""
    nodes = build_nodes(nodes_path)
    graph, _, in_degree = build_graph(edges_path, nodes)
    order = topological_sort(graph, in_degree, nodes)
    roots = find_roots(nodes, in_degree)

    dist: Dict[str, int] = {node_id: math.inf for node_id in nodes}
    shortest_count: Dict[str, int] = {node_id: 0 for node_id in nodes}
    total_paths: Dict[str, int] = {node_id: 0 for node_id in nodes}
    shortest_preds: Dict[str, Set[str]] = {node_id: set() for node_id in nodes}

    for root in roots:
        dist[root] = 0
        shortest_count[root] = 1
        total_paths[root] = 1

    for node_id in order:
        node_dist = dist[node_id]
        node_shortest = shortest_count[node_id]
        node_total = total_paths[node_id]
        if node_total == 0:
            continue
        for neighbor in graph[node_id]:
            # Total path count update always uses all paths through the predecessor.
            total_paths[neighbor] += node_total

            candidate_dist = node_dist + 1
            if candidate_dist < dist[neighbor]:
                dist[neighbor] = candidate_dist
                shortest_count[neighbor] = node_shortest
                shortest_preds[neighbor] = {node_id}
            elif candidate_dist == dist[neighbor] and candidate_dist != math.inf:
                shortest_count[neighbor] += node_shortest
                shortest_preds[neighbor].add(node_id)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    order_rank = {node_id: idx for idx, node_id in enumerate(order)}
    with output_path.open("w", encoding="utf-8") as handle:
        for node_id in order:
            node = nodes[node_id]
            shortest_length = dist[node_id]
            if math.isinf(shortest_length):
                shortest_length = None
            record = {
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

            handle.write(json.dumps(record) + "\n")


def main() -> None:
    """Entry point wiring CLI arguments into the analysis routine."""
    args = parse_args()
    analyze(
        nodes_path=args.nodes,
        edges_path=args.edges,
        output_path=args.output,
        max_samples=args.max_samples,
        only_multipath=args.only_multipath,
    )


if __name__ == "__main__":
    main()