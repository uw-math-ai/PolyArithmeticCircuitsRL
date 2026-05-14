#!/usr/bin/env python3
"""Run the tiny deterministic bootstrap loop."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from lgs.data.curriculum import FixedCurriculum
from lgs.data.target_generators import (
    make_tiny_train_instances,
    make_tiny_validation_instances,
)
from lgs.eval.wandb_logging import (
    add_wandb_args,
    finish_wandb,
    init_wandb,
    log_bootstrap_metrics,
)
from lgs.training.bootstrap_loop import BootstrapConfig, run_bootstrap_training


def main() -> None:
    args = _parse_args()
    run = init_wandb(
        args,
        default_run_name="tiny_bootstrap",
        config=vars(args),
        tags=("bootstrap", "tiny"),
    )
    try:
        curriculum = FixedCurriculum(
            train_instances=make_tiny_train_instances(field_p=17, degree_cap=2),
            validation_instances=make_tiny_validation_instances(
                field_p=17,
                degree_cap=2,
            ),
        )
        config = BootstrapConfig(
            num_rounds=2,
            beam_width=8,
            candidate_k=32,
            epochs_per_round=20,
            batch_size=16,
            seed=0,
        )
        result = run_bootstrap_training(curriculum, config)
        log_bootstrap_metrics(run, result.metrics)

        print("round lambda prefs total loss accuracy h_succ g_succ h_exp g_exp")
        for metric in result.metrics:
            loss = _fmt_optional(metric.train_loss_final)
            accuracy = _fmt_optional(metric.train_accuracy_final)
            print(
                f"{metric.round_idx:>5} "
                f"{metric.lambda_model:>6.2f} "
                f"{metric.num_preferences_added:>5} "
                f"{metric.total_preferences:>5} "
                f"{loss:>8} "
                f"{accuracy:>8} "
                f"{metric.heuristic_val_success_rate:>6.2f} "
                f"{metric.guided_val_success_rate:>6.2f} "
                f"{metric.heuristic_val_avg_expansions:>6.1f} "
                f"{metric.guided_val_avg_expansions:>6.1f}"
            )
        print(f"final_lambda_model={result.final_lambda_model:.2f}")
    finally:
        finish_wandb(run)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_wandb_args(parser)
    return parser.parse_args()


def _fmt_optional(value: float | None) -> str:
    if value is None:
        return "None"
    return f"{value:.4f}"


if __name__ == "__main__":
    main()
