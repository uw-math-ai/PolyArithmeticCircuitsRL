#!/usr/bin/env python3
"""Train on structured benchmark instances and run an eval sweep."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from lgs.data.benchmark_suite import make_structured_benchmark
from lgs.data.curriculum import FixedCurriculum
from lgs.eval.sweep import (
    SweepConfig,
    run_search_sweep,
    summarize_failures,
    summarize_sweep,
)
from lgs.eval.wandb_logging import (
    add_wandb_args,
    finish_wandb,
    init_wandb,
    log_bootstrap_metrics,
    log_sweep_outputs,
)
from lgs.training.bootstrap_loop import BootstrapConfig, run_bootstrap_training
from lgs.training.train_ranker import save_ranker


def main() -> None:
    args = _parse_args()
    run = init_wandb(
        args,
        default_run_name=Path(args.output_dir).name,
        config=vars(args),
        tags=("train-eval", "bootstrap", "benchmark-sweep"),
    )
    try:
        benchmark = make_structured_benchmark(
            field_p=args.field_p,
            degree_cap=args.degree_cap,
            max_instances_per_family=args.max_instances_per_family,
        )
        train_instances, validation_instances, eval_instances = _split_instances(
            benchmark.instances
        )
        if run is not None:
            run.config.update(
                {
                    "num_train_instances": len(train_instances),
                    "num_validation_instances": len(validation_instances),
                    "num_eval_instances": len(eval_instances),
                }
            )
        curriculum = FixedCurriculum(
            train_instances=train_instances,
            validation_instances=validation_instances,
        )
        bootstrap_config = BootstrapConfig(
            num_rounds=args.rounds,
            beam_width=args.train_beam_width,
            candidate_k=args.train_candidate_k,
            tier2_m=args.train_tier2_m,
            epochs_per_round=args.epochs_per_round,
            batch_size=args.batch_size,
            seed=args.seed,
        )
        result = run_bootstrap_training(curriculum, bootstrap_config)

        sweep_config = SweepConfig(
            beam_widths=_parse_int_tuple(args.beam_widths),
            candidate_ks=_parse_int_tuple(args.candidate_ks),
            tier2_ms=_parse_int_tuple(args.tier2_ms),
            lambda_model=result.final_lambda_model,
        )
        rows = run_search_sweep(
            eval_instances,
            ranker=result.ranker,
            encoder=result.encoder,
            config=sweep_config,
        )
        summary = summarize_sweep(rows)
        failures = summarize_failures(rows)

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_jsonl(output_dir / "sweep_rows.jsonl", [asdict(row) for row in rows])
        _write_json(output_dir / "sweep_summary.json", summary)
        _write_json(output_dir / "sweep_failures.json", failures)
        _write_json(
            output_dir / "bootstrap_metrics.json",
            [asdict(metric) for metric in result.metrics],
        )
        save_ranker(output_dir / "ranker.pt", result.ranker, result.encoder)
        log_bootstrap_metrics(run, result.metrics)
        log_sweep_outputs(
            run,
            rows=rows,
            summary=summary,
            failures=failures,
            output_dir=output_dir,
            artifact_name=f"{Path(args.output_dir).name}-train-eval",
        )

        print(
            f"split train={len(train_instances)} validation={len(validation_instances)} "
            f"eval={len(eval_instances)} final_lambda={result.final_lambda_model:.2f}"
        )
        _print_bootstrap_metrics(result.metrics)
        _print_summary(summary)
        _print_failures(failures)
        print(f"wrote benchmark outputs to {output_dir}")
    finally:
        finish_wandb(run)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--max-instances-per-family", type=int, default=2)
    parser.add_argument("--field-p", type=int, default=268435399)
    parser.add_argument("--degree-cap", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs-per-round", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--train-beam-width", type=int, default=8)
    parser.add_argument("--train-candidate-k", type=int, default=32)
    parser.add_argument("--train-tier2-m", type=int, default=128)
    parser.add_argument("--beam-widths", default="1,2,4")
    parser.add_argument("--candidate-ks", default="4,8,16")
    parser.add_argument("--tier2-ms", default="128")
    parser.add_argument("--output-dir", default="results")
    add_wandb_args(parser)
    return parser.parse_args()


def _split_instances(instances):
    train = [instance for index, instance in enumerate(instances) if index % 3 == 0]
    validation = [instance for index, instance in enumerate(instances) if index % 3 == 1]
    eval_set = [instance for index, instance in enumerate(instances) if index % 3 == 2]
    if not validation:
        validation = list(train)
    if not eval_set:
        eval_set = list(validation)
    return train, validation, eval_set


def _parse_int_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in value.split(",") if part)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def _print_bootstrap_metrics(metrics) -> None:
    print("round lambda prefs total loss accuracy h_succ g_succ h_exp g_exp")
    for metric in metrics:
        loss = "None" if metric.train_loss_final is None else f"{metric.train_loss_final:.4f}"
        accuracy = (
            "None"
            if metric.train_accuracy_final is None
            else f"{metric.train_accuracy_final:.4f}"
        )
        print(
            f"{metric.round_idx:>5} {metric.lambda_model:>6.2f} "
            f"{metric.num_preferences_added:>5} {metric.total_preferences:>5} "
            f"{loss:>8} {accuracy:>8} "
            f"{metric.heuristic_val_success_rate:>6.2f} "
            f"{metric.guided_val_success_rate:>6.2f} "
            f"{metric.heuristic_val_avg_expansions:>6.1f} "
            f"{metric.guided_val_avg_expansions:>6.1f}"
        )


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
