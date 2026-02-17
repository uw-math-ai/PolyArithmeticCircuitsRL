"""
End-to-end pipeline: generate board -> train transformer -> eval (incl. unseen).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

try:
    from transformer.build_training_data import generate_board
except ModuleNotFoundError:
    from build_training_data import generate_board


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="End-to-end transformer pipeline")
    parser.add_argument("--steps", type=int, required=True, help="Board complexity C")
    parser.add_argument("--num-vars", type=int, choices=[1, 2], default=1, help="Number of variables")
    parser.add_argument(
        "--no-constant",
        action="store_true",
        help="Disable seeding the constant 1 node",
    )
    parser.add_argument(
        "--board-out-dir",
        type=Path,
        default=Path("transformer/boards"),
        help="Output directory for generated boards",
    )
    parser.add_argument("--prefix", type=str, default=None, help="Board prefix override")
    parser.add_argument("--max-complexity", type=int, default=None, help="Max shortest path length")
    parser.add_argument("--include-non-multipath", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Limit examples for quick runs")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Checkpoint output path")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--metrics-out", type=Path, default=None, help="Training metrics JSONL path")
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation split fraction")
    parser.add_argument("--plot-out", type=Path, default=None, help="Write metrics plot PNG")
    parser.add_argument("--unseen-samples", type=int, default=200)
    parser.add_argument("--unseen-steps", type=int, default=None)
    parser.add_argument("--unseen-max-coeff", type=int, default=5)
    parser.add_argument("--unseen-seed", type=int, default=7)
    return parser.parse_args()


def _run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def main() -> None:
    args = _parse_args()
    prefix = args.prefix or f"game_board_C{args.steps}_V{args.num_vars}"
    args.board_out_dir.mkdir(parents=True, exist_ok=True)
    generate_board(
        steps=args.steps,
        num_vars=args.num_vars,
        output_dir=args.board_out_dir,
        prefix=prefix,
        max_nodes=None,
        max_successors_per_node=None,
        max_samples=5,
        only_multipath=not args.include_non_multipath,
        analysis_max_step=args.max_complexity,
        skip_plot=True,
        include_constant=not args.no_constant,
    )

    train_cmd = [
        sys.executable,
        "-m",
        "transformer.train",
        "--board-dir",
        str(args.board_out_dir),
        "--prefix",
        prefix,
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--output",
        str(args.checkpoint),
    ]
    if args.max_complexity is not None:
        train_cmd += ["--max-complexity", str(args.max_complexity)]
    if args.include_non_multipath:
        train_cmd.append("--include-non-multipath")
    if args.limit is not None:
        train_cmd += ["--limit", str(args.limit)]
    if args.no_constant:
        train_cmd.append("--no-constant")
    if args.device:
        train_cmd += ["--device", args.device]
    if args.metrics_out:
        train_cmd += ["--metrics-out", str(args.metrics_out)]
    if args.val_split is not None:
        train_cmd += ["--val-split", str(args.val_split)]
    if args.plot_out:
        train_cmd += ["--plot-out", str(args.plot_out)]

    _run(train_cmd)

    eval_cmd = [
        sys.executable,
        "-m",
        "transformer.eval",
        "--board-dir",
        str(args.board_out_dir),
        "--prefix",
        prefix,
        "--checkpoint",
        str(args.checkpoint),
        "--num-vars",
        str(args.num_vars),
        "--unseen-samples",
        str(args.unseen_samples),
        "--unseen-max-coeff",
        str(args.unseen_max_coeff),
        "--unseen-seed",
        str(args.unseen_seed),
    ]
    unseen_steps = args.unseen_steps or args.max_complexity or args.steps
    eval_cmd += ["--unseen-steps", str(unseen_steps)]
    if args.device:
        eval_cmd += ["--device", args.device]
    if args.max_complexity is not None:
        eval_cmd += ["--max-complexity", str(args.max_complexity)]
    if args.include_non_multipath:
        eval_cmd.append("--include-non-multipath")
    if args.limit is not None:
        eval_cmd += ["--limit", str(args.limit)]
    if args.no_constant:
        eval_cmd.append("--no-constant")

    _run(eval_cmd)


if __name__ == "__main__":
    main()
