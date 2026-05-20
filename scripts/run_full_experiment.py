#!/usr/bin/env python3
"""Run a larger end-to-end ExIt-style training loop."""

from __future__ import annotations

import argparse
import gc
import json
import multiprocessing
import os
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from random import Random
from typing import Callable

import torch

from decomp_rl.andor_search import AndOrSearch
from decomp_rl.baseline_cost import BaselineCostModel
from decomp_rl.config import SearchConfig
from decomp_rl.elite_buffer import EliteBuffer
from decomp_rl.evaluate import (
    evaluate_random_rollouts,
    summarize_search_results,
    summarize_supervised,
)
from decomp_rl.factor_fp import FiniteFieldFactorizer
from decomp_rl.family_generators import (
    elementary_symmetric_example,
    exact_small_example,
    horner_example,
    multivariate_horner_example,
    planted_factorable_example,
)
from decomp_rl.model import HeuristicPolicyValueModel, TorchPolicyValueNetwork, TorchPolicyValueWrapper
from decomp_rl.replay import PrioritizedReplayBuffer
from decomp_rl.train_search_distill import (
    distill_targets,
    make_distillation_training_examples,
    make_elite_training_examples,
)
from decomp_rl.train_supervised import (
    TorchTrainConfig,
    evaluate_training_examples,
    make_training_examples,
    train_torch_model,
)

try:
    import wandb
except ImportError:  # pragma: no cover - optional dependency
    wandb = None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--prime", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume-checkpoint", default="",
                        help="Path to a checkpoint .pt file, or 'latest' to auto-detect the "
                             "newest cycle checkpoint inside --output-dir/checkpoints/.")
    parser.add_argument("--auto-resume", action=argparse.BooleanOptionalAction, default=False,
                        help="Equivalent to --resume-checkpoint latest; useful for SLURM requeue.")
    parser.add_argument("--initial-supervised-count", type=int, default=96)
    parser.add_argument("--holdout-count", type=int, default=24)
    parser.add_argument("--cycles", type=int, default=8)
    parser.add_argument("--search-targets-per-cycle", type=int, default=24)
    parser.add_argument("--recent-distill-sample-size", type=int, default=24)
    parser.add_argument("--replay-sample-size", type=int, default=24)
    parser.add_argument("--elite-sample-size", type=int, default=16)
    parser.add_argument("--elite-max-training-examples", type=int, default=4096)
    parser.add_argument("--elite-max-nodes-per-trace", type=int, default=64)
    parser.add_argument("--synthetic-sample-size", type=int, default=24)
    parser.add_argument("--synthetic-workers", type=int, default=1)
    parser.add_argument("--synthetic-progress-interval", type=int, default=512)
    parser.add_argument("--synthetic-factor-cache-clear-interval", type=int, default=512)
    parser.add_argument("--replay-capacity", type=int, default=2048)
    parser.add_argument("--elite-capacity", type=int, default=256)
    parser.add_argument("--supervised-epochs", type=int, default=12)
    parser.add_argument("--cycle-epochs", type=int, default=6)
    parser.add_argument("--min-mixed-examples", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--cycle-learning-rate", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--value-loss-weight", type=float, default=0.25)
    parser.add_argument("--search-simulations", type=int, default=96)
    parser.add_argument("--random-rollouts-per-target", type=int, default=1)
    parser.add_argument("--random-rollout-max-steps", type=int, default=32)
    parser.add_argument("--random-rollout-candidates", type=int, default=16)
    parser.add_argument("--cycle-search-retries", type=int, default=1)
    parser.add_argument("--cycle-search-fresh-search-per-target", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cycle-search-progress-interval", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--auto-batch-size", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-auto-batch-size", type=int, default=0)
    parser.add_argument("--gpu-memory-target-fraction", type=float, default=0.9)
    parser.add_argument("--cache-dataset-on-device", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--model-hidden-dim", type=int, default=128)
    parser.add_argument("--model-shared-layers", type=int, default=3)
    parser.add_argument("--model-value-hidden-dim", type=int, default=128)
    parser.add_argument("--model-value-layers", type=int, default=2)
    parser.add_argument("--model-activation", choices=["relu", "gelu"], default="relu")
    parser.add_argument("--reserve-cpu-cores", type=int, default=4)
    parser.add_argument("--torch-cpu-threads", type=int, default=8)
    parser.add_argument("--torch-interop-threads", type=int, default=1)
    parser.add_argument("--nice-level", type=int, default=10)
    parser.add_argument("--replay-uniform-fraction", type=float, default=0.1)
    parser.add_argument("--curriculum-extra-primes", default="5,7")
    parser.add_argument("--curriculum-max-vars", type=int, default=5)
    parser.add_argument("--curriculum-max-support", type=int, default=6)
    parser.add_argument("--curriculum-max-degree", type=int, default=4)
    parser.add_argument("--curriculum-max-horner-degree", type=int, default=8)
    parser.add_argument("--curriculum-max-inner-support", type=int, default=4)
    parser.add_argument("--wandb-entity", default="p-agi")
    parser.add_argument("--wandb-project", default="PolyArithmeticCircuitsRL")
    parser.add_argument("--wandb-run-id", default="")
    parser.add_argument("--wandb-mode", choices=["auto", "online", "offline", "disabled"], default="auto")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else Path("artifacts") / (
        "full_experiment_" + datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    # Resolve auto-resume: --auto-resume or --resume-checkpoint latest both
    # search for the newest cycle checkpoint already saved in this output dir.
    if args.auto_resume and not args.resume_checkpoint:
        args.resume_checkpoint = "latest"
    if args.resume_checkpoint == "latest":
        found = find_latest_cycle_checkpoint(checkpoints_dir)
        args.resume_checkpoint = str(found) if found else ""
        if args.resume_checkpoint:
            print({"stage": "auto_resume_detected", "checkpoint": args.resume_checkpoint}, flush=True)
        else:
            print({"stage": "auto_resume_no_checkpoint_found", "starting_fresh": True}, flush=True)

    runtime_config = configure_runtime(args)
    config_payload = vars(args).copy()
    run_id = args.wandb_run_id or output_dir.name
    config_payload["run_id"] = run_id
    config_payload["wandb_path"] = f"{args.wandb_entity}/{args.wandb_project}/{run_id}"
    config_payload["resolved_output_dir"] = str(output_dir.resolve())
    config_payload["resume_checkpoint"] = str(Path(args.resume_checkpoint).resolve()) if args.resume_checkpoint else ""
    config_payload["timestamp_utc"] = datetime.utcnow().isoformat() + "Z"
    config_payload["runtime"] = runtime_config
    (output_dir / "config.json").write_text(json.dumps(config_payload, indent=2), encoding="utf-8")
    wandb_run, wandb_mode = init_wandb(args, config_payload, output_dir, run_id)
    print(
        {
            "stage": "startup",
            "output_dir": str(output_dir),
            "wandb_path": config_payload["wandb_path"],
            "wandb_mode": wandb_mode,
            "runtime": runtime_config,
        },
        flush=True,
    )

    rng = Random(args.seed)
    baseline_model = BaselineCostModel()
    heuristic_wrapper = HeuristicPolicyValueModel(baseline_model)
    replay = PrioritizedReplayBuffer(capacity=args.replay_capacity, seed=args.seed)
    elite = EliteBuffer(capacity=args.elite_capacity)
    initial_prime_pool = curriculum_prime_pool(args.prime, args.curriculum_extra_primes, progress=0.0)
    holdout_prime_pool = curriculum_prime_pool(args.prime, args.curriculum_extra_primes, progress=1.0)

    print({"stage": "generate_initial_examples", "count": args.initial_supervised_count}, flush=True)
    initial_examples = generate_curriculum_examples(
        rng,
        args.initial_supervised_count,
        progress=0.0,
        base_prime=args.prime,
        prime_pool=initial_prime_pool,
        max_var_count=args.curriculum_max_vars,
        max_support=args.curriculum_max_support,
        max_degree=args.curriculum_max_degree,
        max_horner_degree=args.curriculum_max_horner_degree,
        max_inner_support=args.curriculum_max_inner_support,
    )
    print({"stage": "generate_holdout_examples", "count": args.holdout_count}, flush=True)
    holdout_examples = generate_curriculum_examples(
        Random(args.seed + 999),
        args.holdout_count,
        progress=1.0,
        base_prime=args.prime,
        prime_pool=holdout_prime_pool,
        max_var_count=args.curriculum_max_vars,
        max_support=args.curriculum_max_support,
        max_degree=args.curriculum_max_degree,
        max_horner_degree=args.curriculum_max_horner_degree,
        max_inner_support=args.curriculum_max_inner_support,
    )
    holdout_targets = [example.target for example in holdout_examples]
    holdout_random_eval = evaluate_random_policy(
        holdout_targets,
        baseline_model,
        rollouts_per_target=args.random_rollouts_per_target,
        seed=args.seed + 4242,
        k_candidates=args.random_rollout_candidates,
        max_steps=args.random_rollout_max_steps,
    )

    supervised_training = make_training_examples(initial_examples)
    for example in supervised_training:
        baseline = float(baseline_model.direct_construction_cost(example.target))
        replay.add(
            example,
            priority=compute_priority(
                example,
                best_cost=baseline * (1.0 - example.value_target),
                baseline_cost=baseline,
                model=heuristic_wrapper,
            ),
        )

    network = build_network(args)
    wrapper = None
    best_holdout = None
    best_cycle = 0
    start_cycle = 1
    current_optimizer_state: dict | None = None
    if args.resume_checkpoint:
        checkpoint = load_checkpoint(Path(args.resume_checkpoint), network)
        network = checkpoint["network"]
        resume_device = resolve_model_device(args.device)
        network = network.to(resume_device)
        wrapper = TorchPolicyValueWrapper(network, device=resume_device)
        metadata = checkpoint["metadata"]
        best_holdout = resume_holdout_eval(metadata)
        best_cycle = int(
            metadata.get("source_cycle", metadata.get("cycle", 0)) or 0
        )
        start_cycle = max(1, best_cycle + 1)
        current_optimizer_state = checkpoint.get("optimizer_state_dict")
        print(
            {
                "stage": "resume",
                "checkpoint": str(Path(args.resume_checkpoint).resolve()),
                "checkpoint_stage": metadata.get("stage", "unknown"),
                "saved_at_utc": checkpoint.get("saved_at_utc"),
                "resume_cycle": start_cycle,
                "resume_is_approximate": best_cycle > 0,
                "best_cycle_so_far": best_cycle,
                "best_holdout": best_holdout,
                "optimizer_state_restored": current_optimizer_state is not None,
            },
            flush=True,
        )
    else:
        heuristic_supervised = evaluate_training_examples(supervised_training, heuristic_wrapper)
        print(
            {
                "stage": "stage_a_train_start",
                "examples": len(supervised_training),
                "epochs": args.supervised_epochs,
                "requested_batch_size": args.batch_size,
                "auto_batch_size": args.auto_batch_size,
                "gpu_memory_target_fraction": args.gpu_memory_target_fraction,
                "cache_dataset_on_device": args.cache_dataset_on_device,
                "model": model_config_dict(args),
            },
            flush=True,
        )
        trained = train_torch_model(
            supervised_training,
            TorchTrainConfig(
                epochs=args.supervised_epochs,
                seed=args.seed,
                device=args.device,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                value_loss_weight=args.value_loss_weight,
                auto_batch_size=args.auto_batch_size,
                max_auto_batch_size=args.max_auto_batch_size,
                target_gpu_memory_fraction=args.gpu_memory_target_fraction,
                cache_dataset_on_device=args.cache_dataset_on_device,
            ),
            network=network,
        )
        current_optimizer_state = trained.optimizer_state_dict

        holdout_before = evaluate_search_model(
            heuristic_wrapper,
            holdout_targets,
            baseline_model,
            args.search_simulations,
        )
        holdout_after = evaluate_search_model(
            trained.wrapper,
            holdout_targets,
            baseline_model,
            args.search_simulations,
        )
        save_checkpoint(
            checkpoints_dir / "stage_a.pt",
            trained.network,
            {
                "stage": "supervised",
                "supervised_before": summarize_supervised(heuristic_supervised),
                "supervised_after": summarize_supervised(trained.final_metrics),
                "holdout_before": holdout_before,
                "holdout_after": holdout_after,
                "holdout_random": holdout_random_eval,
                "batch_size": trained.batch_size_stats.__dict__,
            },
            optimizer_state_dict=trained.optimizer_state_dict,
        )
        best_holdout = holdout_after
        best_cycle = 0
        save_checkpoint(
            checkpoints_dir / "best_holdout.pt",
            trained.network,
            {
                "stage": "best_holdout",
                "source_cycle": 0,
                "holdout_eval": holdout_after,
            },
        )
        append_metrics(
            metrics_path,
            {
                "cycle": 0,
                "stage": "supervised",
                "supervised_before": summarize_supervised(heuristic_supervised),
                "supervised_after": summarize_supervised(trained.final_metrics),
                "holdout_before": holdout_before,
                "holdout_after": holdout_after,
                "holdout_random": holdout_random_eval,
                "replay_size": len(replay),
                "elite_size": len(elite),
                "wandb_mode": wandb_mode,
                "batch_size": trained.batch_size_stats.__dict__,
                "curriculum_prime_pool": initial_prime_pool,
            },
        )
        log_to_wandb(
            wandb_run,
            {
                "cycle": 0,
                "stage": "supervised",
                "supervised_before": summarize_supervised(heuristic_supervised),
                "supervised_after": summarize_supervised(trained.final_metrics),
                "holdout_before": holdout_before,
                "holdout_after": holdout_after,
                "holdout_random": holdout_random_eval,
                "replay_size": len(replay),
                "elite_size": len(elite),
                "batch_size": trained.batch_size_stats.__dict__,
                "curriculum_prime_pool": initial_prime_pool,
            },
            step=0,
        )
        print(
            {
                "stage": "supervised",
                "holdout_before": holdout_before,
                "holdout_after": holdout_after,
                "holdout_random": holdout_random_eval,
                "batch_size": trained.batch_size_stats.__dict__,
                "wandb_path": config_payload["wandb_path"],
                "wandb_mode": wandb_mode,
            },
            flush=True,
        )

        wrapper = trained.wrapper
        network = trained.network

    if best_holdout is None:
        best_holdout = {}

    for cycle in range(start_cycle, args.cycles + 1):
        cycle_rng = Random(args.seed + cycle * 17)
        progress = cycle / max(1, args.cycles)
        cycle_prime_pool = curriculum_prime_pool(args.prime, args.curriculum_extra_primes, progress=progress)
        curriculum = curriculum_profile(
            progress=progress,
            base_prime=args.prime,
            prime_pool=cycle_prime_pool,
            max_var_count=args.curriculum_max_vars,
            max_support=args.curriculum_max_support,
            max_degree=args.curriculum_max_degree,
            max_horner_degree=args.curriculum_max_horner_degree,
        )
        print({"stage": "cycle_start", "cycle": cycle}, flush=True)
        cycle_targets = generate_cycle_targets(
            cycle_rng,
            count=args.search_targets_per_cycle,
            cycle=cycle,
            base_prime=args.prime,
            prime_pool=cycle_prime_pool,
            max_var_count=args.curriculum_max_vars,
            max_support=args.curriculum_max_support,
            max_degree=args.curriculum_max_degree,
            max_horner_degree=args.curriculum_max_horner_degree,
            max_inner_support=args.curriculum_max_inner_support,
        )
        search = AndOrSearch(
            baseline_model=baseline_model,
            model=wrapper,
            search_config=SearchConfig(simulations=args.search_simulations),
        )
        print({"stage": "cycle_search", "cycle": cycle, "targets": len(cycle_targets), "simulations": args.search_simulations}, flush=True)
        cycle_search_started_at = time.perf_counter()
        last_progress_time = cycle_search_started_at

        def cycle_search_progress(payload: dict[str, object]) -> None:
            nonlocal last_progress_time
            if payload.get("status") != "done":
                if payload.get("status") == "error":
                    event = {
                        "stage": "cycle_search_error",
                        "cycle": cycle,
                        **payload,
                    }
                    print(event, flush=True)
                    log_progress_to_wandb(wandb_run, event, cycle * 100_000)
                return

            target_index = int(payload["target_index"])
            completed = target_index + 1
            should_log = (
                completed == 1
                or completed == len(cycle_targets)
                or completed % max(1, args.cycle_search_progress_interval) == 0
            )
            if should_log:
                now = time.perf_counter()
                event = {
                    "stage": "cycle_search_progress",
                    "cycle": cycle,
                    "completed": completed,
                    "total_targets": len(cycle_targets),
                    "elapsed_sec": round(now - cycle_search_started_at, 2),
                    "since_last_log_sec": round(now - last_progress_time, 2),
                    **payload,
                }
                print(event, flush=True)
                log_progress_to_wandb(wandb_run, event, cycle * 100_000 + completed)
                last_progress_time = now

        distilled = distill_targets(
            cycle_targets,
            search,
            elite_buffer=elite,
            fresh_search_per_target=args.cycle_search_fresh_search_per_target,
            retry_failures=args.cycle_search_retries,
            progress_callback=cycle_search_progress,
        )
        search.close()
        print(
            {
                "stage": "cycle_post_search",
                "cycle": cycle,
                "phase": "distillation_complete",
                "distilled": len(distilled),
                "elite_size": len(elite),
                "elapsed_sec": round(time.perf_counter() - cycle_search_started_at, 2),
            },
            flush=True,
        )
        gc_started_at = time.perf_counter()
        gc.collect()
        print(
            {
                "stage": "cycle_post_search",
                "cycle": cycle,
                "phase": "gc_complete",
                "elapsed_sec": round(time.perf_counter() - gc_started_at, 2),
            },
            flush=True,
        )
        recent_distill_examples = make_distillation_training_examples(distilled)
        print(
            {
                "stage": "cycle_post_search",
                "cycle": cycle,
                "phase": "recent_distill_examples",
                "examples": len(recent_distill_examples),
            },
            flush=True,
        )
        for distilled_example, training_example in zip(distilled, recent_distill_examples):
            baseline = float(baseline_model.direct_construction_cost(distilled_example.target))
            replay.add(
                training_example,
                priority=compute_priority(
                    training_example,
                    best_cost=distilled_example.best_cost,
                    baseline_cost=baseline,
                    model=wrapper,
                ),
            )

        replay_examples = replay.sample(
            args.replay_sample_size,
            uniform_fraction=args.replay_uniform_fraction,
        )
        print(
            {
                "stage": "cycle_post_search",
                "cycle": cycle,
                "phase": "replay_sample",
                "examples": len(replay_examples),
                "replay_size": len(replay),
            },
            flush=True,
        )
        elite_examples = make_elite_training_examples(
            elite.sample(args.elite_sample_size),
            baseline_model=baseline_model,
            k_candidates=16,
            max_examples=args.elite_max_training_examples,
            max_nodes_per_trace=args.elite_max_nodes_per_trace,
        )
        print(
            {
                "stage": "cycle_post_search",
                "cycle": cycle,
                "phase": "elite_examples",
                "examples": len(elite_examples),
                "elite_size": len(elite),
            },
            flush=True,
        )

        def synthetic_progress(payload: dict[str, object]) -> None:
            completed = int(payload.get("completed", 0) or 0)
            event = {
                "stage": "cycle_post_search",
                "cycle": cycle,
                "phase": "synthetic_examples",
                **payload,
            }
            print(event, flush=True)
            log_progress_to_wandb(wandb_run, event, cycle * 100_000 + 10_000 + completed)

        synthetic_examples = make_training_examples(
            generate_curriculum_examples_parallel(
                cycle_rng,
                args.synthetic_sample_size,
                progress=progress,
                base_prime=args.prime,
                prime_pool=cycle_prime_pool,
                max_var_count=args.curriculum_max_vars,
                max_support=args.curriculum_max_support,
                max_degree=args.curriculum_max_degree,
                max_horner_degree=args.curriculum_max_horner_degree,
                max_inner_support=args.curriculum_max_inner_support,
                progress_callback=synthetic_progress,
                progress_interval=args.synthetic_progress_interval,
                factor_cache_clear_interval=args.synthetic_factor_cache_clear_interval,
                workers=args.synthetic_workers,
            )
        )
        print(
            {
                "stage": "cycle_post_search",
                "cycle": cycle,
                "phase": "synthetic_examples_complete",
                "examples": len(synthetic_examples),
            },
            flush=True,
        )

        mixed_examples = (
            recent_distill_examples[: args.recent_distill_sample_size]
            + replay_examples
            + elite_examples[: args.elite_sample_size]
            + synthetic_examples
        )
        raw_mixed_size = len(mixed_examples)
        mixed_examples = repeat_training_examples(mixed_examples, args.min_mixed_examples)
        train_start_event = {
            "stage": "cycle_train",
            "cycle": cycle,
            "examples": len(mixed_examples),
            "unique_examples_before_repeat": raw_mixed_size,
            "epochs": args.cycle_epochs,
            "requested_batch_size": args.batch_size,
            "auto_batch_size": args.auto_batch_size,
        }
        print(train_start_event, flush=True)
        log_progress_to_wandb(wandb_run, train_start_event, cycle * 100_000 + 20_000)
        trained = train_torch_model(
            mixed_examples,
            TorchTrainConfig(
                epochs=args.cycle_epochs,
                seed=args.seed + cycle,
                device=args.device,
                batch_size=args.batch_size,
                learning_rate=args.cycle_learning_rate or args.learning_rate,
                weight_decay=args.weight_decay,
                value_loss_weight=args.value_loss_weight,
                auto_batch_size=args.auto_batch_size,
                max_auto_batch_size=args.max_auto_batch_size,
                target_gpu_memory_fraction=args.gpu_memory_target_fraction,
                cache_dataset_on_device=args.cache_dataset_on_device,
            ),
            network=network,
            optimizer_state_dict=current_optimizer_state,
        )
        current_optimizer_state = trained.optimizer_state_dict
        wrapper = trained.wrapper
        network = trained.network

        cycle_eval = evaluate_search_model(
            wrapper,
            holdout_targets,
            baseline_model,
            args.search_simulations,
        )
        cycle_distill_eval = evaluate_search_model(
            wrapper,
            cycle_targets,
            baseline_model,
            args.search_simulations,
        )
        cycle_random_eval = evaluate_random_policy(
            cycle_targets,
            baseline_model,
            rollouts_per_target=args.random_rollouts_per_target,
            seed=args.seed + cycle * 1009,
            k_candidates=args.random_rollout_candidates,
            max_steps=args.random_rollout_max_steps,
        )
        metric_row = {
            "cycle": cycle,
            "stage": "search_distill",
            "train_metrics": summarize_supervised(trained.final_metrics),
            "holdout_eval": cycle_eval,
            "cycle_eval": cycle_distill_eval,
            "holdout_random": holdout_random_eval,
            "cycle_random": cycle_random_eval,
            "replay_size": len(replay),
            "elite_size": len(elite),
            "recent_distill_count": len(recent_distill_examples),
            "raw_mixed_batch_size": raw_mixed_size,
            "mixed_batch_size": len(mixed_examples),
            "batch_size": trained.batch_size_stats.__dict__,
            "curriculum": curriculum,
        }
        append_metrics(metrics_path, metric_row)
        log_to_wandb(wandb_run, metric_row, step=cycle)
        save_checkpoint(
            checkpoints_dir / f"cycle_{cycle:03d}.pt",
            network,
            metric_row,
            optimizer_state_dict=current_optimizer_state,
        )
        if is_better_eval(cycle_eval, best_holdout):
            best_holdout = cycle_eval
            best_cycle = cycle
            save_checkpoint(
                checkpoints_dir / "best_holdout.pt",
                network,
                {
                    "stage": "best_holdout",
                    "source_cycle": cycle,
                    "holdout_eval": cycle_eval,
                    "cycle_eval": cycle_distill_eval,
                },
            )
        print(metric_row, flush=True)

    save_checkpoint(
        checkpoints_dir / "final.pt",
        network,
        {"stage": "final", "best_holdout_cycle": best_cycle, "best_holdout_eval": best_holdout},
    )
    if wandb_run is not None:
        log_checkpoint_artifact(wandb_run, output_dir, run_id)
        wandb_run.finish()


def curriculum_prime_pool(base_prime: int, extra_primes: str, progress: float) -> list[int]:
    primes = [base_prime]
    extras = [int(piece.strip()) for piece in extra_primes.split(",") if piece.strip()]
    if progress >= 0.34 and extras:
        primes.append(extras[0])
    if progress >= 0.67:
        primes.extend(extras[1:])
    return sorted({prime for prime in primes if prime > 1})


def make_variable_tuple(count: int) -> tuple[str, ...]:
    if count <= 1:
        return ("x",)
    if count == 2:
        return ("x", "y")
    return tuple(f"x{i + 1}" for i in range(count))


def configure_runtime(args) -> dict[str, object]:
    runtime: dict[str, object] = {}
    if args.nice_level > 0 and hasattr(os, "nice"):
        runtime["nice_level"] = os.nice(args.nice_level)

    available_cpus = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else list(range(os.cpu_count() or 1))
    runtime["available_cpu_ids"] = available_cpus
    reserved_count = max(0, min(args.reserve_cpu_cores, max(0, len(available_cpus) - 1)))
    runtime["reserved_cpu_ids"] = available_cpus[:reserved_count]
    if hasattr(os, "sched_setaffinity") and reserved_count:
        training_cpu_ids = available_cpus[reserved_count:]
        os.sched_setaffinity(0, training_cpu_ids)
        runtime["training_cpu_ids"] = sorted(os.sched_getaffinity(0))
    else:
        runtime["training_cpu_ids"] = available_cpus

    max_cpu_threads = max(1, min(args.torch_cpu_threads, len(runtime["training_cpu_ids"])))
    torch.set_num_threads(max_cpu_threads)
    runtime["torch_cpu_threads"] = torch.get_num_threads()
    if args.torch_interop_threads > 0:
        try:
            torch.set_num_interop_threads(args.torch_interop_threads)
            runtime["torch_interop_threads"] = args.torch_interop_threads
        except RuntimeError:
            runtime["torch_interop_threads"] = "unchanged"

    runtime["cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        runtime["tf32"] = True
    return runtime


def model_config_dict(args) -> dict[str, object]:
    return {
        "hidden_dim": args.model_hidden_dim,
        "shared_layers": args.model_shared_layers,
        "value_hidden_dim": args.model_value_hidden_dim,
        "value_layers": args.model_value_layers,
        "activation": args.model_activation,
    }


def build_network(args) -> TorchPolicyValueNetwork:
    return TorchPolicyValueNetwork(
        hidden_dim=args.model_hidden_dim,
        shared_layers=args.model_shared_layers,
        value_hidden_dim=args.model_value_hidden_dim,
        value_layers=args.model_value_layers,
        activation=args.model_activation,
    )


def resolve_model_device(device: str) -> str:
    if device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_checkpoint(path: Path, network) -> dict[str, object]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    network.load_state_dict(payload["model_state_dict"])
    return {
        "network": network,
        "metadata": payload.get("metadata", {}),
        "saved_at_utc": payload.get("saved_at_utc"),
        "optimizer_state_dict": payload.get("optimizer_state_dict"),
    }


def find_latest_cycle_checkpoint(checkpoints_dir: Path) -> Path | None:
    """Return the highest-numbered cycle_NNN.pt in checkpoints_dir, or None."""
    import re
    pattern = re.compile(r"^cycle_(\d+)\.pt$")
    best_num, best_path = -1, None
    if not checkpoints_dir.is_dir():
        return None
    for p in checkpoints_dir.iterdir():
        m = pattern.match(p.name)
        if m:
            num = int(m.group(1))
            if num > best_num:
                best_num, best_path = num, p
    return best_path


def resume_holdout_eval(metadata: dict[str, object]) -> dict[str, float]:
    holdout_after = metadata.get("holdout_after")
    if isinstance(holdout_after, dict):
        return holdout_after
    holdout_eval = metadata.get("holdout_eval")
    if isinstance(holdout_eval, dict):
        return holdout_eval
    return {}


def curriculum_profile(
    progress: float,
    base_prime: int,
    prime_pool: list[int],
    max_var_count: int,
    max_support: int,
    max_degree: int,
    max_horner_degree: int,
) -> dict[str, object]:
    capped_progress = max(0.0, min(1.0, progress))
    planted_support = min(3 + int(round((max_support - 3) * capped_progress)), max_support)
    planted_degree = min(2 + int(round((max_degree - 2) * capped_progress)), max_degree)
    horner_degree_min = 3 + int(round(2 * capped_progress))
    horner_degree_max = min(5 + int(round((max_horner_degree - 5) * capped_progress)), max_horner_degree)
    variable_count = min(max_var_count, 2 + int(round((max_var_count - 2) * capped_progress)))
    family_weights = (
        ("planted", 0.50 - 0.10 * capped_progress),
        ("horner", 0.25 - 0.05 * capped_progress),
        ("elementary", 0.20 + 0.05 * capped_progress),
        ("exact_small", 0.05 + 0.10 * capped_progress),
    )
    return {
        "base_prime": base_prime,
        "prime_pool": prime_pool,
        "variable_count": variable_count,
        "planted_support": planted_support,
        "planted_degree": planted_degree,
        "horner_degree_min": horner_degree_min,
        "horner_degree_max": horner_degree_max,
        "family_weights": family_weights,
    }


def generate_curriculum_examples(
    rng: Random,
    count: int,
    progress: float,
    base_prime: int,
    prime_pool: list[int],
    max_var_count: int,
    max_support: int,
    max_degree: int,
    max_horner_degree: int,
    max_inner_support: int,
    progress_callback: Callable[[dict[str, object]], None] | None = None,
    progress_interval: int = 0,
    factor_cache_clear_interval: int = 512,
):
    if count <= 0:
        return []
    profile = curriculum_profile(progress, base_prime, prime_pool, max_var_count, max_support, max_degree, max_horner_degree)
    family_weights = profile["family_weights"]
    families = [name for name, _ in family_weights]
    weights = [weight for _, weight in family_weights]
    examples = []
    family_counts: Counter[str] = Counter()
    baseline_model = BaselineCostModel()
    factorizer = FiniteFieldFactorizer()
    started_at = time.perf_counter()
    try:
        for index in range(count):
            if (
                progress_callback is not None
                and (
                    index == 0
                    or index == count - 1
                    or (
                        progress_interval > 0
                        and index > 0
                        and index % progress_interval == 0
                    )
                )
            ):
                progress_callback(
                    {
                        "status": "generating",
                        "completed": index,
                        "total": count,
                        "elapsed_sec": round(time.perf_counter() - started_at, 2),
                        "family_counts": dict(family_counts),
                    }
                )
            if (
                factor_cache_clear_interval > 0
                and index > 0
                and index % factor_cache_clear_interval == 0
            ):
                factorizer.clear()
            for _attempt in range(32):
                family = rng.choices(families, weights=weights, k=1)[0]
                prime = rng.choice(prime_pool)
                variable_count = int(profile["variable_count"])
                variables = make_variable_tuple(variable_count)
                try:
                    if family == "planted":
                        example = planted_factorable_example(
                            rng,
                            prime,
                            variables,
                            support_size=int(profile["planted_support"]),
                            max_degree=int(profile["planted_degree"]),
                            baseline_model=baseline_model,
                            factorizer=factorizer,
                        )
                    elif family == "horner":
                        if len(variables) > 1:
                            example = multivariate_horner_example(
                                rng,
                                prime,
                                variables,
                                outer_degree=min(max_horner_degree, int(profile["horner_degree_min"]) + 1),
                                inner_support_size=max(1, min(max_inner_support, int(profile["planted_support"]) - 1)),
                                inner_max_degree=max(1, int(profile["planted_degree"])),
                            )
                        else:
                            degree = rng.randint(int(profile["horner_degree_min"]), int(profile["horner_degree_max"]))
                            coefficients = [rng.randint(0, prime - 1) for _ in range(degree + 1)]
                            if all(coeff == 0 for coeff in coefficients):
                                coefficients[0] = 1
                            if coefficients[0] == 0:
                                coefficients[0] = 1
                            example = horner_example(coefficients, prime)
                    elif family == "elementary":
                        example = elementary_symmetric_example(
                            variable_count=max(4, variable_count),
                            degree=2,
                            prime=prime,
                        )
                    else:
                        example = exact_small_example(
                            rng,
                            prime,
                            variables=("x", "y"),
                            baseline_model=baseline_model,
                            factorizer=factorizer,
                        )
                except RuntimeError:
                    continue
                examples.append(example)
                family_counts[example.family] += 1
                break
            else:
                raise RuntimeError(f"Failed to generate a curriculum example after repeated attempts for profile {profile}")
    finally:
        factorizer.close()
    if progress_callback is not None:
        progress_callback(
            {
                "status": "complete",
                "completed": len(examples),
                "total": count,
                "elapsed_sec": round(time.perf_counter() - started_at, 2),
                "family_counts": dict(family_counts),
            }
        )
    return examples


def generate_curriculum_examples_parallel(
    rng: Random,
    count: int,
    progress: float,
    base_prime: int,
    prime_pool: list[int],
    max_var_count: int,
    max_support: int,
    max_degree: int,
    max_horner_degree: int,
    max_inner_support: int,
    progress_callback: Callable[[dict[str, object]], None] | None = None,
    progress_interval: int = 0,
    factor_cache_clear_interval: int = 512,
    workers: int = 1,
):
    if count <= 0:
        return []
    workers = max(1, min(int(workers), count))
    if workers <= 1:
        return generate_curriculum_examples(
            rng,
            count,
            progress=progress,
            base_prime=base_prime,
            prime_pool=prime_pool,
            max_var_count=max_var_count,
            max_support=max_support,
            max_degree=max_degree,
            max_horner_degree=max_horner_degree,
            max_inner_support=max_inner_support,
            progress_callback=progress_callback,
            progress_interval=progress_interval,
            factor_cache_clear_interval=factor_cache_clear_interval,
        )

    chunk_count = workers
    base_count, remainder = divmod(count, chunk_count)
    chunk_sizes = [base_count + (1 if index < remainder else 0) for index in range(chunk_count)]
    seeds = [rng.randrange(1, 2**63 - 1) for _ in range(chunk_count)]
    started_at = time.perf_counter()
    family_counts: Counter[str] = Counter()
    completed = 0
    results: dict[int, list] = {}

    if progress_callback is not None:
        progress_callback(
            {
                "status": "parallel_start",
                "completed": 0,
                "total": count,
                "workers": workers,
                "chunks": chunk_count,
                "elapsed_sec": 0.0,
                "family_counts": {},
            }
        )

    payloads = [
        {
            "seed": seeds[index],
            "count": chunk_sizes[index],
            "progress": progress,
            "base_prime": base_prime,
            "prime_pool": prime_pool,
            "max_var_count": max_var_count,
            "max_support": max_support,
            "max_degree": max_degree,
            "max_horner_degree": max_horner_degree,
            "max_inner_support": max_inner_support,
            "factor_cache_clear_interval": factor_cache_clear_interval,
        }
        for index in range(chunk_count)
        if chunk_sizes[index] > 0
    ]
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
        future_to_index = {
            executor.submit(_generate_curriculum_examples_chunk, payload): index
            for index, payload in enumerate(payloads)
        }
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            chunk_examples, chunk_counts = future.result()
            results[index] = chunk_examples
            completed += len(chunk_examples)
            family_counts.update(chunk_counts)
            if progress_callback is not None:
                progress_callback(
                    {
                        "status": "parallel_generating",
                        "completed": completed,
                        "total": count,
                        "workers": workers,
                        "chunks_done": len(results),
                        "chunks": len(payloads),
                        "elapsed_sec": round(time.perf_counter() - started_at, 2),
                        "family_counts": dict(family_counts),
                    }
                )

    examples = []
    for index in range(len(payloads)):
        examples.extend(results[index])
    if progress_callback is not None:
        progress_callback(
            {
                "status": "complete",
                "completed": len(examples),
                "total": count,
                "workers": workers,
                "chunks": len(payloads),
                "elapsed_sec": round(time.perf_counter() - started_at, 2),
                "family_counts": dict(family_counts),
            }
        )
    return examples


def _generate_curriculum_examples_chunk(payload: dict[str, object]):
    examples = generate_curriculum_examples(
        Random(int(payload["seed"])),
        int(payload["count"]),
        progress=float(payload["progress"]),
        base_prime=int(payload["base_prime"]),
        prime_pool=list(payload["prime_pool"]),
        max_var_count=int(payload["max_var_count"]),
        max_support=int(payload["max_support"]),
        max_degree=int(payload["max_degree"]),
        max_horner_degree=int(payload["max_horner_degree"]),
        max_inner_support=int(payload["max_inner_support"]),
        progress_callback=None,
        progress_interval=0,
        factor_cache_clear_interval=int(payload["factor_cache_clear_interval"]),
    )
    return examples, dict(Counter(example.family for example in examples))


def generate_cycle_targets(
    rng: Random,
    count: int,
    cycle: int,
    base_prime: int,
    prime_pool: list[int],
    max_var_count: int,
    max_support: int,
    max_degree: int,
    max_horner_degree: int,
    max_inner_support: int,
):
    targets = []
    profile = curriculum_profile(
        min(1.0, cycle / max(1, cycle + 2)),
        base_prime=base_prime,
        prime_pool=prime_pool,
        max_var_count=max_var_count,
        max_support=max_support,
        max_degree=max_degree,
        max_horner_degree=max_horner_degree,
    )
    baseline_model = BaselineCostModel()
    factorizer = FiniteFieldFactorizer()
    for index in range(count):
        try:
            for _attempt in range(32):
                prime = rng.choice(prime_pool)
                selector = index % 4
                try:
                    if selector == 0:
                        target = planted_factorable_example(
                            rng,
                            prime,
                            make_variable_tuple(int(profile["variable_count"])),
                            support_size=int(profile["planted_support"]),
                            max_degree=int(profile["planted_degree"]),
                            baseline_model=baseline_model,
                            factorizer=factorizer,
                        ).target
                    elif selector == 1:
                        variables = make_variable_tuple(int(profile["variable_count"]))
                        if len(variables) > 1:
                            target = multivariate_horner_example(
                                rng,
                                prime,
                                variables,
                                outer_degree=min(max_horner_degree, int(profile["horner_degree_min"]) + 1),
                                inner_support_size=max(1, min(max_inner_support, int(profile["planted_support"]) - 1)),
                                inner_max_degree=max(1, int(profile["planted_degree"])),
                            ).target
                        else:
                            degree = rng.randint(int(profile["horner_degree_min"]), int(profile["horner_degree_max"]))
                            coefficients = [rng.randint(0, prime - 1) for _ in range(degree + 1)]
                            if all(coeff == 0 for coeff in coefficients):
                                coefficients[0] = 1
                            if coefficients[0] == 0:
                                coefficients[0] = 1
                            target = horner_example(coefficients, prime).target
                    elif selector == 2:
                        variable_count = max(4, int(profile["variable_count"]))
                        target = elementary_symmetric_example(variable_count=variable_count, degree=2, prime=prime).target
                    else:
                        target = exact_small_example(
                            rng,
                            prime,
                            variables=("x", "y"),
                            baseline_model=baseline_model,
                            factorizer=factorizer,
                        ).target
                except RuntimeError:
                    continue
                targets.append(target)
                break
            else:
                raise RuntimeError(f"Failed to generate a cycle target after repeated attempts for profile {profile}")
        finally:
            if index > 0 and index % 64 == 0:
                factorizer.clear()
    factorizer.close()
    return targets


def evaluate_search_model(
    wrapper,
    targets,
    baseline_model: BaselineCostModel,
    search_simulations: int,
) -> dict[str, float]:
    baselines = [float(baseline_model.direct_construction_cost(target)) for target in targets]
    search = AndOrSearch(
        baseline_model=baseline_model,
        model=wrapper,
        search_config=SearchConfig(simulations=search_simulations),
    )
    try:
        results = [search.search(target) for target in targets]
        summary = summarize_search_results(results, baselines)
        return summary.__dict__
    finally:
        search.close()


def evaluate_random_policy(
    targets,
    baseline_model: BaselineCostModel,
    rollouts_per_target: int,
    seed: int,
    k_candidates: int,
    max_steps: int,
) -> dict[str, float]:
    metrics = evaluate_random_rollouts(
        list(targets),
        baseline_model=baseline_model,
        rollouts_per_target=rollouts_per_target,
        seed=seed,
        k_candidates=k_candidates,
        max_steps=max_steps,
    )
    return metrics.__dict__


def repeat_training_examples(examples, minimum_count: int):
    if minimum_count <= 0 or not examples or len(examples) >= minimum_count:
        return examples
    repeats, remainder = divmod(minimum_count, len(examples))
    return examples * repeats + examples[:remainder]


def compute_priority(training_example, best_cost: float, baseline_cost: float, model) -> float:
    priors, value_prediction = model.score_candidates(training_example.target, list(training_example.candidates))
    search_gain = max(0.0, baseline_cost - best_cost)
    policy_gap = 0.5 * sum(
        abs(target - prior)
        for target, prior in zip(training_example.policy_target, priors or training_example.policy_target)
    )
    value_signal = abs(training_example.value_target - value_prediction)
    novelty = (
        0.02 * training_example.target.support_size
        + 0.02 * training_example.target.total_degree
        + 0.01 * len(training_example.target.variables)
        + 0.01 * training_example.target.p
    )
    return max(1e-6, search_gain + 0.5 * value_signal + 0.35 * policy_gap + novelty)


def is_better_eval(candidate: dict[str, float], incumbent: dict[str, float]) -> bool:
    candidate_gain = candidate.get("average_search_gain", float("-inf"))
    incumbent_gain = incumbent.get("average_search_gain", float("-inf"))
    if candidate_gain != incumbent_gain:
        return candidate_gain > incumbent_gain
    return candidate.get("average_best_cost", float("inf")) < incumbent.get("average_best_cost", float("inf"))


def save_checkpoint(path: Path, network, metadata: dict, optimizer_state_dict=None) -> None:
    payload = {
        "model_state_dict": network.state_dict(),
        "metadata": metadata,
        "saved_at_utc": datetime.utcnow().isoformat() + "Z",
    }
    if optimizer_state_dict is not None:
        payload["optimizer_state_dict"] = optimizer_state_dict
    torch.save(payload, path)


def append_metrics(path: Path, payload: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def init_wandb(args, config_payload: dict, output_dir: Path, run_id: str):
    if args.wandb_mode == "disabled" or wandb is None:
        return None, "disabled" if wandb is None else args.wandb_mode

    init_kwargs = {
        "entity": args.wandb_entity,
        "project": args.wandb_project,
        "id": run_id,
        "name": run_id,
        "resume": "allow",
        "config": config_payload,
        "dir": str(output_dir),
    }
    modes = [args.wandb_mode] if args.wandb_mode != "auto" else ["online", "offline"]
    last_error = None
    for mode in modes:
        try:
            run = wandb.init(mode=mode, **init_kwargs)
            try:
                run.define_metric("progress/*", step_metric="progress/step")
            except Exception:
                pass
            return run, mode
        except Exception as exc:  # pragma: no cover - depends on environment auth
            last_error = exc
    warning_path = output_dir / "wandb_warning.txt"
    warning_path.write_text(
        f"Failed to initialize wandb in modes {modes}: {last_error}\n",
        encoding="utf-8",
    )
    return None, "disabled"


def log_to_wandb(run, payload: dict, step: int) -> None:
    if run is None:
        return
    run.log(flatten_dict(payload), step=step)


def log_progress_to_wandb(run, payload: dict, progress_step: int) -> None:
    if run is None:
        return
    run.log(
        flatten_dict(
            {
                "progress": {
                    "step": progress_step,
                    **payload,
                }
            }
        )
    )


def flatten_dict(payload: dict, prefix: str = "") -> dict[str, float | int | str]:
    flat: dict[str, float | int | str] = {}
    for key, value in payload.items():
        flat_key = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(flatten_dict(value, flat_key))
        else:
            flat[flat_key] = value
    return flat


def log_checkpoint_artifact(run, output_dir: Path, run_id: str) -> None:
    artifact = wandb.Artifact(f"{run_id}-checkpoints", type="experiment")
    for relative_path in (
        Path("config.json"),
        Path("metrics.jsonl"),
        Path("checkpoints/best_holdout.pt"),
        Path("checkpoints/final.pt"),
    ):
        path = output_dir / relative_path
        if path.exists():
            artifact.add_file(str(path), name=str(relative_path))
    run.log_artifact(artifact)


if __name__ == "__main__":
    main()
