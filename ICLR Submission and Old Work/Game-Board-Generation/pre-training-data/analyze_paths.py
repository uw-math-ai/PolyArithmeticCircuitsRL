#!/usr/bin/env python3
"""
Legacy CLI wrapper that reuses interesting_polynomial_generator for analysis.

This script preserves the original interface:
    python analyze_paths.py --nodes path.nodes.jsonl --edges path.edges.jsonl \
        --output path.analysis.jsonl --max-samples 5 --only-multipath
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import networkx as nx

from interesting_polynomial_generator import analyze_graph


def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def build_graph(nodes_path: Path, edges_path: Path) -> nx.DiGraph:
    graph = nx.DiGraph()
    for record in read_jsonl(nodes_path):
        node_id = record.get("id") or record.get("key")
        if not node_id:
            continue
        graph.add_node(
            node_id,
            key=record.get("key", node_id),
            step=record.get("step"),
            label=record.get("label"),
            expr_str=record.get("expr_str"),
        )
    for record in read_jsonl(edges_path):
        source = record.get("source")
        target = record.get("target")
        if source is None or target is None:
            continue
        graph.add_edge(
            source,
            target,
            op=record.get("op"),
            label=record.get("label"),
            operand=record.get("operand"),
        )
    return graph


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze path multiplicities.")
    parser.add_argument("--nodes", type=Path, required=True, help="Nodes JSONL file.")
    parser.add_argument("--edges", type=Path, required=True, help="Edges JSONL file.")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL path.")
    parser.add_argument("--max-samples", type=int, default=5, help="Max stored shortest-path samples.")
    parser.add_argument(
        "--only-multipath",
        action="store_true",
        help="Only emit nodes that have multiple paths.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    graph = build_graph(args.nodes, args.edges)
    records = analyze_graph(graph, args.max_samples, args.only_multipath)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
