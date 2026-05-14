#!/usr/bin/env python3
"""Run a structured benchmark sweep, optionally with a saved ranker."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from lgs.data.benchmark_suite import make_structured_benchmark
from lgs.eval.sweep import (
    SweepConfig,
    run_search_sweep,
    summarize_failures,
    summarize_sweep,
)
from lgs.training.train_ranker import load_ranker


def main() -> None:
    args = _parse_args()
    benchmark = make_structured_benchmark(
        field_p=args.field_p,
        degree_cap=args.degree_cap,
        max_instances_per_family=args.max_instances_per_family,
    )
    ranker = None
    encoder = None
    if args.checkpoint:
        ranker, encoder = load_ranker(args.checkpoint)

    config = SweepConfig(
        beam_widths=_parse_int_tuple(args.beam_widths),
        candidate_ks=_parse_int_tuple(args.candidate_ks),
        tier2_ms=_parse_int_tuple(args.tier2_ms),
        lambda_model=args.lambda_model,
    )
    rows = run_search_sweep(
        benchmark.instances,
        ranker=ranker,
        encoder=encoder,
        config=config,
    )
    summary = summarize_sweep(rows)
    failures = summarize_failures(rows)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "sweep_rows.jsonl", [asdict(row) for row in rows])
    _write_json(output_dir / "sweep_summary.json", summary)
    _write_json(output_dir / "sweep_failures.json", failures)
    _print_summary(summary)
    _print_failures(failures)
    print(f"wrote {len(rows)} rows to {output_dir}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-instances-per-family", type=int, default=2)
    parser.add_argument("--field-p", type=int, default=268435399)
    parser.add_argument("--degree-cap", type=int, default=8)
    parser.add_argument("--beam-widths", default="1,2,4")
    parser.add_argument("--candidate-ks", default="4,8,16")
    parser.add_argument("--tier2-ms", default="128")
    parser.add_argument("--lambda-model", type=float, default=1.0)
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="results")
    return parser.parse_args()


def _parse_int_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in value.split(",") if part)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def _print_summary(summary: list[dict[str, object]]) -> None:
    print("method family beam k tier2 count solve avg_ops avg_exp med_exp avg_sec delta")
    for row in summary:
        avg_ops = row["avg_best_ops"]
        delta = row["guided_minus_heuristic_solve_rate"]
        print(
            f"{row['method']:<9} {row['family']:<16} "
            f"{row['beam_width']:>4} {row['candidate_k']:>3} {row['tier2_m']:>5} "
            f"{row['count']:>5} {row['solve_rate']:>5.2f} "
            f"{avg_ops if avg_ops is not None else 'None':>7} "
            f"{row['avg_expansions']:>7.1f} {row['median_expansions']:>7.1f} "
            f"{row['avg_runtime_sec']:>7.4f} "
            f"{delta if delta is not None else 'None'}"
        )


def _print_failures(failures: list[dict[str, object]]) -> None:
    if not failures:
        print("failures: none")
        return
    print("failures family complexity method beam k tier2 count avg_exp instances")
    for row in failures[:20]:
        print(
            f"failures {row['family']:<16} {row['intended_complexity']!s:>10} "
            f"{row['method']:<9} {row['beam_width']:>4} {row['candidate_k']:>3} "
            f"{row['tier2_m']:>5} {row['failure_count']:>5} "
            f"{row['avg_expansions']:>7.1f} {','.join(row['instance_ids'])}"
        )


if __name__ == "__main__":
    main()
