"""
Dataset generator for polynomial->circuit SymPy string pairs.

Uses Game-Board-Generation JSONL dumps (nodes/edges/analysis) to reconstruct
circuits, then emits training-ready examples.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterable, List, Sequence

try:
    from transformer.polynomial_to_circuit import build_dataset, CircuitExample
    from transformer.build_training_data import generate_board
except ModuleNotFoundError:
    from polynomial_to_circuit import build_dataset, CircuitExample
    from build_training_data import generate_board


def _analysis_matches_nodes(analysis_path: Path, nodes_path: Path, sample: int = 200) -> bool:
    node_ids = set()
    with nodes_path.open("r", encoding="utf-8") as handle:
        for i, line in enumerate(handle):
            if i >= sample:
                break
            node_ids.add(json.loads(line)["id"])

    match = 0
    with analysis_path.open("r", encoding="utf-8") as handle:
        for i, line in enumerate(handle):
            if i >= sample:
                break
            if json.loads(line).get("id") in node_ids:
                match += 1
    return match > 0


def _write_jsonl(path: Path, examples: Sequence[CircuitExample]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for ex in examples:
            handle.write(json.dumps({"poly": ex.polynomial, "target": ex.circuit}) + "\n")


def _split_examples(examples: List[CircuitExample], split: float) -> tuple[list[CircuitExample], list[CircuitExample]]:
    if split <= 0.0 or split >= 1.0:
        raise ValueError("split must be between 0 and 1")
    idx = int(len(examples) * split)
    return examples[:idx], examples[idx:]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build polynomial->circuit datasets")
    parser.add_argument("--nodes", type=Path, default=None, help="Nodes JSONL path")
    parser.add_argument("--edges", type=Path, default=None, help="Edges JSONL path")
    parser.add_argument("--analysis", type=Path, default=None, help="Analysis JSONL path")
    parser.add_argument("--board-dir", type=Path, default=None, help="Directory containing board JSONL files")
    parser.add_argument("--prefix", type=str, default=None, help="Board prefix like game_board_C4")
    parser.add_argument(
        "--auto-generate-board",
        action="store_true",
        help="Generate a game board if inputs are missing",
    )
    parser.add_argument("--steps", type=int, default=None, help="Complexity steps for auto-generation")
    parser.add_argument("--num-vars", type=int, default=1, help="Number of variables for auto-generation")
    parser.add_argument(
        "--board-out-dir",
        type=Path,
        default=Path("transformer/boards"),
        help="Output directory for auto-generated boards",
    )
    parser.add_argument("--max-complexity", type=int, default=None, help="Max shortest path length")
    parser.add_argument(
        "--include-non-multipath",
        action="store_true",
        help="Include nodes without multiple paths (default: multipath-only)",
    )
    parser.add_argument(
        "--allow-all",
        action="store_true",
        help="Allow building dataset without analysis JSONL",
    )
    parser.add_argument("--limit", type=int, default=None, help="Max number of examples")
    parser.add_argument("--seed", type=int, default=7, help="RNG seed")
    parser.add_argument("--output", type=Path, default=None, help="Write all examples to JSONL")
    parser.add_argument("--train-out", type=Path, default=None, help="Write train split JSONL")
    parser.add_argument("--val-out", type=Path, default=None, help="Write val split JSONL")
    parser.add_argument("--split", type=float, default=0.9, help="Train split fraction")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    nodes_path = args.nodes
    edges_path = args.edges
    analysis_path = args.analysis
    if args.board_dir and args.prefix:
        nodes_path = nodes_path or (args.board_dir / f"{args.prefix}.nodes.jsonl")
        edges_path = edges_path or (args.board_dir / f"{args.prefix}.edges.jsonl")
        analysis_path = analysis_path or (args.board_dir / f"{args.prefix}.analysis.jsonl")

    if args.auto_generate_board and (nodes_path is None or edges_path is None):
        if args.steps is None:
            raise SystemExit("Auto-generate requested but --steps is missing.")
        out_dir = args.board_dir or args.board_out_dir
        prefix = args.prefix or f"game_board_C{args.steps}_V{args.num_vars}"
        nodes_path, edges_path, analysis_path = generate_board(
            steps=args.steps,
            num_vars=args.num_vars,
            output_dir=out_dir,
            prefix=prefix,
            max_nodes=None,
            max_successors_per_node=None,
            max_samples=5,
            only_multipath=not args.include_non_multipath,
            analysis_max_step=args.max_complexity,
            skip_plot=True,
        )

    if nodes_path is None or edges_path is None:
        raise SystemExit("Must provide --nodes and --edges or --board-dir with --prefix")

    if analysis_path is None and not args.allow_all:
        raise SystemExit("Analysis JSONL missing. Provide --analysis or use --allow-all.")

    only_multipath = not args.include_non_multipath
    if analysis_path is None:
        only_multipath = False
    elif not _analysis_matches_nodes(analysis_path, nodes_path):
        raise SystemExit(
            "Analysis JSONL does not match node IDs. "
            "Double-check --analysis or regenerate the analysis for this board."
        )

    examples = build_dataset(
        nodes_path=nodes_path,
        edges_path=edges_path,
        analysis_path=analysis_path,
        max_complexity=args.max_complexity,
        only_multipath=only_multipath,
    )

    random.seed(args.seed)
    random.shuffle(examples)

    if args.limit is not None:
        examples = examples[: args.limit]

    if args.output:
        _write_jsonl(args.output, examples)

    if args.train_out or args.val_out:
        train_split, val_split = _split_examples(examples, args.split)
        if args.train_out:
            _write_jsonl(args.train_out, train_split)
        if args.val_out:
            _write_jsonl(args.val_out, val_split)

    print(f"Built {len(examples)} examples")


if __name__ == "__main__":
    main()
