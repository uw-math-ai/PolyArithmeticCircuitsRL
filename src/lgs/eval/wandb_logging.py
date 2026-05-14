"""Optional Weights & Biases helpers for experiment scripts."""

from __future__ import annotations

import json
import os
from argparse import ArgumentParser, BooleanOptionalAction, Namespace
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


DEFAULT_WANDB_PROJECT = "PolyArithmeticCircuitsRL"


def add_wandb_args(parser: ArgumentParser) -> None:
    parser.add_argument(
        "--wandb",
        action=BooleanOptionalAction,
        default=_env_flag("ENABLE_WANDB", default=True),
        help="Enable Weights & Biases logging. Use --no-wandb to disable.",
    )
    parser.add_argument(
        "--wandb-project",
        default=DEFAULT_WANDB_PROJECT,
        help=f"W&B project name (default: {DEFAULT_WANDB_PROJECT}).",
    )
    parser.add_argument(
        "--wandb-entity",
        default=None,
        help="W&B entity or team.",
    )
    parser.add_argument(
        "--wandb-run-name",
        default=None,
        help="W&B run name.",
    )


def init_wandb(
    args: Namespace,
    *,
    default_run_name: str,
    config: Mapping[str, Any],
    tags: Sequence[str],
) -> Any | None:
    if not getattr(args, "wandb", False):
        return None
    try:
        import wandb
    except ImportError as exc:
        raise SystemExit(
            "W&B logging requested, but wandb is not installed. "
            "Install it with `python -m pip install wandb`."
        ) from exc

    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity or None,
        name=args.wandb_run_name or default_run_name,
        config=dict(config),
        tags=list(tags),
    )


def finish_wandb(run: Any | None) -> None:
    if run is None:
        return
    import wandb

    wandb.finish()


def log_bootstrap_metrics(run: Any | None, metrics: Sequence[Any]) -> None:
    if run is None:
        return
    import wandb

    for metric in metrics:
        wandb.log(
            {
                "bootstrap/lambda_model": metric.lambda_model,
                "bootstrap/num_preferences_added": metric.num_preferences_added,
                "bootstrap/total_preferences": metric.total_preferences,
                "bootstrap/train_loss_final": metric.train_loss_final,
                "bootstrap/train_accuracy_final": metric.train_accuracy_final,
                "bootstrap/heuristic_val_success_rate": metric.heuristic_val_success_rate,
                "bootstrap/guided_val_success_rate": metric.guided_val_success_rate,
                "bootstrap/heuristic_val_avg_expansions": (
                    metric.heuristic_val_avg_expansions
                ),
                "bootstrap/guided_val_avg_expansions": (
                    metric.guided_val_avg_expansions
                ),
            },
            step=metric.round_idx,
        )


def log_sweep_outputs(
    run: Any | None,
    *,
    rows: Sequence[Any],
    summary: Sequence[Mapping[str, Any]],
    failures: Sequence[Mapping[str, Any]],
    output_dir: Path,
    artifact_name: str,
) -> None:
    if run is None:
        return
    import wandb

    wandb.log(_sweep_scalar_metrics(rows))
    if summary:
        wandb.log({"sweep/summary": _table_from_dicts(wandb, summary)})
    if failures:
        wandb.log({"sweep/failures": _table_from_dicts(wandb, failures)})

    artifact = wandb.Artifact(_safe_artifact_name(artifact_name), type="lgs-results")
    for file_name in (
        "sweep_rows.jsonl",
        "sweep_summary.json",
        "sweep_failures.json",
        "bootstrap_metrics.json",
        "ranker.pt",
    ):
        path = output_dir / file_name
        if path.exists():
            artifact.add_file(str(path))
    wandb.log_artifact(artifact)


def _sweep_scalar_metrics(rows: Sequence[Any]) -> dict[str, float | int]:
    metrics: dict[str, float | int] = {"sweep/rows": len(rows)}
    by_method: dict[str, list[Any]] = {}
    for row in rows:
        by_method.setdefault(row.method, []).append(row)

    for method, method_rows in by_method.items():
        successes = sum(1 for row in method_rows if row.success)
        metrics[f"sweep/{method}/rows"] = len(method_rows)
        metrics[f"sweep/{method}/solve_rate"] = successes / len(method_rows)
        metrics[f"sweep/{method}/avg_expansions"] = sum(
            row.expansions for row in method_rows
        ) / len(method_rows)
        metrics[f"sweep/{method}/avg_runtime_sec"] = sum(
            row.runtime_sec for row in method_rows
        ) / len(method_rows)
    return metrics


def _table_from_dicts(wandb: Any, rows: Sequence[Mapping[str, Any]]) -> Any:
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
    return safe or "lgs-results"


def _env_flag(name: str, *, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}
