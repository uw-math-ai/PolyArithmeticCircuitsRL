#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple


@dataclass
class Node:
    node_id: str
    expr_str: str
    step: int | None


def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_board(nodes_path: Path, edges_path: Path) -> tuple[Dict[str, Node], Dict[str, Set[str]], Dict[str, int]]:
    nodes: Dict[str, Node] = {}
    for rec in read_jsonl(nodes_path):
        node_id = rec.get("id") or rec.get("key")
        if not node_id:
            continue
        step = rec.get("step")
        if step is not None:
            try:
                step = int(step)
            except Exception:
                step = None
        nodes[node_id] = Node(
            node_id=node_id,
            expr_str=str(rec.get("expr_str") or rec.get("label") or node_id),
            step=step,
        )

    forward: Dict[str, Set[str]] = {nid: set() for nid in nodes}
    in_degree: Dict[str, int] = {nid: 0 for nid in nodes}

    for rec in read_jsonl(edges_path):
        src = rec.get("source")
        tgt = rec.get("target")
        if src not in nodes or tgt not in nodes:
            continue
        if tgt not in forward[src]:
            forward[src].add(tgt)
            in_degree[tgt] += 1

    return nodes, forward, in_degree


def get_roots(nodes: Dict[str, Node], in_degree: Dict[str, int]) -> Set[str]:
    roots = {nid for nid, deg in in_degree.items() if deg == 0}
    if roots:
        return roots

    min_step = math.inf
    for node in nodes.values():
        if node.step is not None:
            min_step = min(min_step, node.step)

    if math.isinf(min_step):
        return set(nodes)
    return {nid for nid, node in nodes.items() if node.step == min_step}


def topological_order(nodes: Dict[str, Node], forward: Dict[str, Set[str]], in_degree: Dict[str, int]) -> List[str]:
    remaining = dict(in_degree)
    queue = deque(
        sorted(
            (nid for nid, deg in remaining.items() if deg == 0),
            key=lambda nid: (
                nodes[nid].step if nodes[nid].step is not None else math.inf,
                nid,
            ),
        )
    )

    out: List[str] = []
    while queue:
        node_id = queue.popleft()
        out.append(node_id)
        for nxt in forward[node_id]:
            remaining[nxt] -= 1
            if remaining[nxt] == 0:
                queue.append(nxt)

    if len(out) != len(nodes):
        raise ValueError("Graph contains a cycle; expected DAG game board.")
    return out


def shortest_path_samples(
    node_id: str,
    roots: Set[str],
    shortest_preds: Dict[str, Set[str]],
    limit: int,
    order_rank: Dict[str, int],
) -> List[List[str]]:
    if limit <= 0:
        return []
    samples: List[List[str]] = []
    cur: List[str] = []

    def dfs(nid: str) -> None:
        if len(samples) >= limit:
            return
        cur.append(nid)
        preds = shortest_preds.get(nid, set())
        if nid in roots or not preds:
            samples.append(list(reversed(cur)))
            cur.pop()
            return
        for p in sorted(preds, key=lambda x: (order_rank.get(x, math.inf), x)):
            dfs(p)
            if len(samples) >= limit:
                break
        cur.pop()

    dfs(node_id)
    return samples


def analyze_board(name: str, nodes_path: Path, edges_path: Path) -> tuple[dict, list[dict]]:
    nodes, forward, in_degree = load_board(nodes_path, edges_path)
    order = topological_order(nodes, forward, in_degree)
    roots = get_roots(nodes, in_degree)

    dist: Dict[str, float] = {nid: math.inf for nid in nodes}
    shortest_count: Dict[str, int] = {nid: 0 for nid in nodes}
    total_paths: Dict[str, int] = {nid: 0 for nid in nodes}
    shortest_preds: Dict[str, Set[str]] = {nid: set() for nid in nodes}

    for r in roots:
        dist[r] = 0
        shortest_count[r] = 1
        total_paths[r] = 1

    for nid in order:
        if total_paths[nid] == 0:
            continue
        nd = dist[nid]
        ns = shortest_count[nid]
        nt = total_paths[nid]
        for nxt in forward[nid]:
            total_paths[nxt] += nt
            cand = nd + 1
            if cand < dist[nxt]:
                dist[nxt] = cand
                shortest_count[nxt] = ns
                shortest_preds[nxt] = {nid}
            elif cand == dist[nxt] and cand != math.inf:
                shortest_count[nxt] += ns
                shortest_preds[nxt].add(nid)

    order_rank = {nid: i for i, nid in enumerate(order)}
    step_counts = Counter()
    shortest_depth_counts = Counter()
    shortest_count_buckets = Counter()
    mismatched_depth = 0

    per_node: list[dict] = []
    for nid in order:
        node = nodes[nid]
        step = node.step
        shortest_length = None if math.isinf(dist[nid]) else int(dist[nid])

        if step is not None:
            step_counts[step] += 1
        if shortest_length is not None:
            shortest_depth_counts[shortest_length] += 1
        if step is not None and shortest_length is not None and step != shortest_length:
            mismatched_depth += 1

        k = shortest_count[nid]
        if k <= 1:
            shortest_count_buckets["1"] += 1
        elif k == 2:
            shortest_count_buckets["2"] += 1
        elif 3 <= k <= 5:
            shortest_count_buckets["3-5"] += 1
        elif 6 <= k <= 10:
            shortest_count_buckets["6-10"] += 1
        else:
            shortest_count_buckets[">10"] += 1

        sample = []
        if shortest_length is not None:
            sample_paths = shortest_path_samples(
                nid,
                roots,
                shortest_preds,
                limit=1,
                order_rank=order_rank,
            )
            if sample_paths:
                sample = [nodes[s].expr_str for s in sample_paths[0]]

        per_node.append(
            {
                "id": nid,
                "expr_str": node.expr_str,
                "step": step,
                "shortest_length": shortest_length,
                "shortest_path_count": shortest_count[nid],
                "total_path_count": total_paths[nid],
                "multiple_optimal_circuits": shortest_count[nid] > 1,
                "canonical_shortest_path_expr": sample,
            }
        )

    multi_opt_nodes = [r for r in per_node if r["multiple_optimal_circuits"]]
    top_multi_opt = sorted(
        multi_opt_nodes,
        key=lambda r: (r["shortest_path_count"], r["total_path_count"]),
        reverse=True,
    )[:15]

    summary = {
        "board_name": name,
        "nodes_path": str(nodes_path),
        "edges_path": str(edges_path),
        "num_nodes": len(nodes),
        "num_edges": sum(len(v) for v in forward.values()),
        "num_roots": len(roots),
        "depth_distribution_by_step": dict(sorted(step_counts.items())),
        "shortest_depth_distribution": dict(sorted(shortest_depth_counts.items())),
        "step_vs_shortest_depth_mismatches": mismatched_depth,
        "nodes_with_multiple_optimal_circuits": len(multi_opt_nodes),
        "fraction_with_multiple_optimal_circuits": (
            len(multi_opt_nodes) / len(nodes) if nodes else 0.0
        ),
        "optimal_circuit_count_buckets": dict(shortest_count_buckets),
        "max_optimal_circuit_count": max((r["shortest_path_count"] for r in per_node), default=0),
        "max_total_circuit_count": max((r["total_path_count"] for r in per_node), default=0),
        "top_nodes_by_optimal_circuit_count": [
            {
                "expr_str": r["expr_str"],
                "step": r["step"],
                "shortest_path_count": r["shortest_path_count"],
                "total_path_count": r["total_path_count"],
                "example_shortest_path": r["canonical_shortest_path_expr"],
            }
            for r in top_multi_opt
        ],
    }
    return summary, per_node


def write_markdown(out_path: Path, summaries: list[dict]) -> None:
    lines: list[str] = []
    lines.append("# C4 Game Board Analytics")
    lines.append("")
    lines.append("Definitions used for reviewer feedback:")
    lines.append("- Circuit: any root-to-node path in the game-board DAG.")
    lines.append("- Efficient/optimal circuit: a shortest root-to-node path.")
    lines.append("- Multiple optimal circuits: nodes with `shortest_path_count > 1`.")
    lines.append("")

    for s in summaries:
        lines.append(f"## {s['board_name']}")
        lines.append("")
        lines.append(f"- Nodes: {s['num_nodes']:,}")
        lines.append(f"- Edges: {s['num_edges']:,}")
        lines.append(f"- Roots: {s['num_roots']}")
        lines.append(f"- Nodes with multiple optimal circuits: {s['nodes_with_multiple_optimal_circuits']:,} ({100*s['fraction_with_multiple_optimal_circuits']:.2f}%)")
        lines.append(f"- Max number of optimal circuits for a single node: {s['max_optimal_circuit_count']:,}")
        lines.append(f"- Max number of total circuits for a single node: {s['max_total_circuit_count']:,}")
        lines.append(f"- Step vs shortest-depth mismatches: {s['step_vs_shortest_depth_mismatches']}")
        lines.append("")

        lines.append("### Depth Distribution (step)")
        for d, c in s["depth_distribution_by_step"].items():
            lines.append(f"- depth {d}: {c:,}")
        lines.append("")

        lines.append("### Optimal Circuit Count Buckets")
        for k in ["1", "2", "3-5", "6-10", ">10"]:
            if k in s["optimal_circuit_count_buckets"]:
                lines.append(f"- {k}: {s['optimal_circuit_count_buckets'][k]:,}")
        lines.append("")

        lines.append("### Top Nodes By Number of Optimal Circuits")
        for row in s["top_nodes_by_optimal_circuit_count"][:10]:
            sample_path = " -> ".join(row["example_shortest_path"])
            lines.append(
                f"- `{row['expr_str']}` (step={row['step']}, optimal={row['shortest_path_count']:,}, total={row['total_path_count']:,})"
            )
            lines.append(f"  example shortest path: {sample_path}")
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def write_per_node_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "board_name",
        "id",
        "expr_str",
        "step",
        "shortest_length",
        "shortest_path_count",
        "total_path_count",
        "multiple_optimal_circuits",
        "canonical_shortest_path_expr",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["canonical_shortest_path_expr"] = " | ".join(out["canonical_shortest_path_expr"])
            writer.writerow(out)


def main() -> None:
    base = Path(__file__).resolve().parent
    boards = [
        {
            "name": "C4-main-multivar",
            "nodes": base / "game_board_C4.nodes.jsonl",
            "edges": base / "game_board_C4.edges.jsonl",
            "analysis_jsonl": base / "game_board_C4.analysis.jsonl",
        },
        {
            "name": "C4-pretraining-singlevar",
            "nodes": base / "pre-training-data" / "game_board_C4.nodes.jsonl",
            "edges": base / "pre-training-data" / "game_board_C4.edges.jsonl",
            "analysis_jsonl": base / "pre-training-data" / "game_board_C4.recomputed.analysis.jsonl",
        },
    ]

    summaries: list[dict] = []
    merged_rows: list[dict] = []

    for board in boards:
        summary, rows = analyze_board(board["name"], board["nodes"], board["edges"])
        summaries.append(summary)

        board["analysis_jsonl"].parent.mkdir(parents=True, exist_ok=True)
        with board["analysis_jsonl"].open("w", encoding="utf-8") as handle:
            for r in rows:
                handle.write(json.dumps(r) + "\n")

        for r in rows:
            merged = dict(r)
            merged["board_name"] = board["name"]
            merged_rows.append(merged)

    out_json = base / "c4_boards_analytics_summary.json"
    out_md = base / "c4_boards_analytics_summary.md"
    out_csv = base / "c4_boards_per_node_metrics.csv"

    out_json.write_text(json.dumps({"boards": summaries}, indent=2), encoding="utf-8")
    write_markdown(out_md, summaries)
    write_per_node_csv(out_csv, merged_rows)

    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")
    print(f"Wrote {out_csv}")
    for board in boards:
        print(f"Wrote {board['analysis_jsonl']}")


if __name__ == "__main__":
    main()
