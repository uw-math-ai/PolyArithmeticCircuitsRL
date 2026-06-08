#!/usr/bin/env python3
"""Run matched-budget beam/Gumbel planner sweeps."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from lgs.data.benchmark_suite import make_structured_benchmark
from lgs.eval.planner_sweep import (
    PlannerSweepConfig,
    run_planner_sweep,
    summarize_planner_deltas,
    summarize_planner_sweep,
)
from lgs.eval.wandb_logging import add_wandb_args, finish_wandb, init_wandb
from lgs.training.train_ranker import load_ranker


def main() -> None:
    args = _parse_args()
    run = init_wandb(
        args,
        default_run_name=Path(args.output_dir).name,
        config=vars(args),
        tags=("planner-sweep", "guided" if args.checkpoint else "heuristic"),
    )
    try:
        benchmark = make_structured_benchmark(
            field_p=args.field_p,
            degree_cap=args.degree_cap,
            max_instances_per_family=args.max_instances_per_family,
        )
        ranker = None
        encoder = None
        include_guided = False
        if args.checkpoint:
            ranker, encoder = load_ranker(args.checkpoint)
            include_guided = True
        elif args.require_guided:
            raise SystemExit("--require-guided was set, but --checkpoint is missing")

        config = PlannerSweepConfig(
            beam_widths=_parse_int_tuple(args.beam_widths),
            candidate_ks=_parse_int_tuple(args.candidate_ks),
            tier2_ms=_parse_int_tuple(args.tier2_ms),
            expansion_budgets=_parse_int_tuple(args.expansion_budgets),
            gumbel_seeds=_parse_int_tuple(args.gumbel_seeds),
            gumbel_initial_width=args.gumbel_initial_width,
            gumbel_rounds=args.gumbel_rounds,
            rollout_depth=args.rollout_depth,
            gumbel_scale=args.gumbel_scale,
            lambda_model=args.lambda_model,
            include_guided=include_guided,
        )
        if not include_guided:
            print("checkpoint not provided; guided beam/Gumbel rows are skipped")

        rows = run_planner_sweep(
            benchmark.instances,
            ranker=ranker,
            encoder=encoder,
            config=config,
            benchmark_name=benchmark.name,
        )
        summary = summarize_planner_sweep(rows)
        deltas = summarize_planner_deltas(rows)

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_jsonl(output_dir / "planner_sweep_rows.jsonl", [asdict(row) for row in rows])
        _write_json(output_dir / "planner_sweep_summary.json", summary)
        _write_json(output_dir / "planner_sweep_deltas.json", deltas)
        _log_wandb_outputs(
            run,
            rows=rows,
            summary=summary,
            deltas=deltas,
            output_dir=output_dir,
            artifact_name=f"{output_dir.name}-planner-sweep",
        )

        _print_summary(summary)
        _print_deltas(deltas)
        print(
            "candidate recall is defined only for solved traces; "
            "unsolved histories have no recall report"
        )
        print(f"wrote {len(rows)} rows to {output_dir}")
    finally:
        finish_wandb(run)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-instances-per-family", type=int, default=2)
    parser.add_argument("--field-p", type=int, default=268435399)
    parser.add_argument("--degree-cap", type=int, default=8)
    parser.add_argument("--beam-widths", default="1,2,4")
    parser.add_argument("--candidate-ks", default="4,8,16")
    parser.add_argument("--tier2-ms", default="64")
    parser.add_argument("--expansion-budgets", default="64")
    parser.add_argument("--gumbel-seeds", default="0,1,2")
    parser.add_argument("--gumbel-initial-width", type=int, default=16)
    parser.add_argument("--gumbel-rounds", type=int, default=3)
    parser.add_argument("--rollout-depth", type=int, default=1)
    parser.add_argument("--gumbel-scale", type=float, default=1.0)
    parser.add_argument("--lambda-model", type=float, default=1.0)
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument(
        "--require-guided",
        action="store_true",
        help="Fail if no checkpoint is provided for guided methods.",
    )
    parser.add_argument("--output-dir", type=str, default="results/planner_sweep")
    add_wandb_args(parser)
    return parser.parse_args()


def _parse_int_tuple(value: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not values:
        raise ValueError("expected at least one comma-separated integer")
    return values


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def _print_summary(summary: list[dict[str, Any]]) -> None:
    print("method family complexity beam k tier2 budget count solve avg_ops avg_exp med_exp")
    for row in summary:
        avg_ops = row["avg_best_ops"]
        print(
            f"{row['method']:<16} {row['family']:<16} "
            f"{row['intended_complexity']!s:>10} {row['beam_width']:>4} "
            f"{row['candidate_k']:>3} {row['tier2_m']:>5} "
            f"{row['expansion_budget']:>6} {row['count']:>5} "
            f"{row['success_rate']:>5.2f} "
            f"{avg_ops if avg_ops is not None else 'None':>7} "
            f"{row['avg_expansions']:>7.1f} {row['median_expansions']:>7.1f}"
        )


def _print_deltas(deltas: list[dict[str, Any]]) -> None:
    if not deltas:
        print("planner deltas: none")
        return
    print("delta comparison family complexity beam k tier2 budget solve_delta exp_delta")
    for row in deltas[:40]:
        print(
            f"delta {row['comparison']:<34} {row['family']:<16} "
            f"{row['intended_complexity']!s:>10} {row['beam_width']:>4} "
            f"{row['candidate_k']:>3} {row['tier2_m']:>5} "
            f"{row['expansion_budget']:>6} {row['solve_rate_delta']:>7.3f} "
            f"{row['avg_expansion_delta']:>8.1f}"
        )


def _log_wandb_outputs(
    run: Any | None,
    *,
    rows: Sequence[Any],
    summary: Sequence[dict[str, Any]],
    deltas: Sequence[dict[str, Any]],
    output_dir: Path,
    artifact_name: str,
) -> None:
    if run is None:
        return
    import wandb

    wandb.log(_planner_scalar_metrics(rows))
    if summary:
        wandb.log({"planner_sweep/summary": _table_from_dicts(wandb, summary)})
    if deltas:
        wandb.log({"planner_sweep/deltas": _table_from_dicts(wandb, deltas)})

    artifact = wandb.Artifact(_safe_artifact_name(artifact_name), type="lgs-results")
    for file_name in (
        "planner_sweep_rows.jsonl",
        "planner_sweep_summary.json",
        "planner_sweep_deltas.json",
    ):
        path = output_dir / file_name
        if path.exists():
            artifact.add_file(str(path))
    wandb.log_artifact(artifact)


def _planner_scalar_metrics(rows: Sequence[Any]) -> dict[str, float | int]:
    metrics: dict[str, float | int] = {"planner_sweep/rows": len(rows)}
    by_method: dict[str, list[Any]] = {}
    for row in rows:
        by_method.setdefault(row.method, []).append(row)
    for method, method_rows in by_method.items():
        successes = sum(1 for row in method_rows if row.success)
        metrics[f"planner_sweep/{method}/rows"] = len(method_rows)
        metrics[f"planner_sweep/{method}/solve_rate"] = successes / len(method_rows)
        metrics[f"planner_sweep/{method}/avg_expansions"] = sum(
            row.expansions for row in method_rows
        ) / len(method_rows)
    return metrics


def _table_from_dicts(wandb: Any, rows: Sequence[dict[str, Any]]) -> Any:
    columns = sorted({key for row in rows for key in row})
    data = [[_table_value(row.get(column)) for column in columns] for row in rows]
    return wandb.Table(columns=columns, data=data)


def _table_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True)
    return value


def _safe_artifact_name(name: str) -> str:
    safe = "".join(
        char if char.isalnum() or char in "._-" else "-"
        for char in name.strip()
    ).strip(".-_")
    return safe or "planner-sweep"


if __name__ == "__main__":
    main()
