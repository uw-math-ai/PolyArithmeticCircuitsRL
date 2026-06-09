"""Four-method planner sweeps for beam and Gumbel search."""

from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import median
from typing import Any

from lgs.env.problem_instance import ProblemInstance
from lgs.models.candidate_ranker import CandidateRanker
from lgs.models.feature_encoder import CandidateFeatureEncoder
from lgs.search.beam_search import beam_search
from lgs.search.gumbel_search import gumbel_search
from lgs.search.search_history import SearchHistory, history_expansion_count


@dataclass
class PlannerSweepRow:
    method: str
    planner: str
    guided: bool
    benchmark_name: str
    instance_id: str
    family: str
    intended_complexity: int | None
    generative_ops: int | None
    beam_width: int | None
    candidate_k: int
    tier2_m: int
    expansion_budget: int
    lambda_model: float
    seed: int | None
    success: bool
    best_ops: int | None
    expansions: int
    runtime_sec: float
    candidate_generation_calls: int | None = None
    total_candidates_generated: int | None = None
    total_candidates_scored: int | None = None
    model_forward_calls: int | None = None
    root_expansions: int | None = None
    rollout_expansions: int | None = None


@dataclass(frozen=True)
class PlannerSweepConfig:
    beam_widths: tuple[int, ...] = (1, 2, 4, 8)
    candidate_ks: tuple[int, ...] = (4, 8, 16, 32)
    tier2_ms: tuple[int, ...] = (64, 128)
    expansion_budgets: tuple[int, ...] = (64, 128, 256)
    gumbel_seeds: tuple[int, ...] = tuple(range(10))
    gumbel_initial_width: int = 16
    gumbel_rounds: int = 3
    rollout_depth: int = 1
    gumbel_scale: float = 1.0
    lambda_model: float = 1.0
    include_beam: bool = True
    include_gumbel: bool = True
    include_heuristic: bool = True
    include_guided: bool = True

    def __post_init__(self) -> None:
        _validate_positive_int_tuple("beam_widths", self.beam_widths)
        _validate_non_negative_int_tuple("candidate_ks", self.candidate_ks)
        _validate_non_negative_int_tuple("tier2_ms", self.tier2_ms)
        _validate_non_negative_int_tuple("expansion_budgets", self.expansion_budgets)
        _validate_int_tuple("gumbel_seeds", self.gumbel_seeds)
        _require_positive_int("gumbel_initial_width", self.gumbel_initial_width)
        _require_positive_int("gumbel_rounds", self.gumbel_rounds)
        _require_non_negative_int("rollout_depth", self.rollout_depth)
        _require_non_negative_float("gumbel_scale", self.gumbel_scale)
        _require_non_negative_float("lambda_model", self.lambda_model)


def run_planner_sweep(
    instances: Sequence[ProblemInstance],
    *,
    ranker: CandidateRanker | None = None,
    encoder: CandidateFeatureEncoder | None = None,
    config: PlannerSweepConfig,
    benchmark_name: str = "structured",
) -> list[PlannerSweepRow]:
    if not isinstance(config, PlannerSweepConfig):
        raise TypeError("config must be a PlannerSweepConfig")
    if config.include_guided and (ranker is None or encoder is None):
        raise ValueError("ranker and encoder are required when include_guided=True")

    rows: list[PlannerSweepRow] = []
    for beam_width in config.beam_widths:
        for candidate_k in config.candidate_ks:
            for tier2_m in config.tier2_ms:
                for expansion_budget in config.expansion_budgets:
                    for instance in instances:
                        if config.include_beam and config.include_heuristic:
                            rows.append(
                                _run_beam(
                                    instance,
                                    method="beam_heuristic",
                                    guided=False,
                                    benchmark_name=benchmark_name,
                                    beam_width=beam_width,
                                    candidate_k=candidate_k,
                                    tier2_m=tier2_m,
                                    expansion_budget=expansion_budget,
                                    ranker=None,
                                    encoder=None,
                                    lambda_model=0.0,
                                )
                            )
                        if config.include_beam and config.include_guided:
                            rows.append(
                                _run_beam(
                                    instance,
                                    method="beam_guided",
                                    guided=True,
                                    benchmark_name=benchmark_name,
                                    beam_width=beam_width,
                                    candidate_k=candidate_k,
                                    tier2_m=tier2_m,
                                    expansion_budget=expansion_budget,
                                    ranker=ranker,
                                    encoder=encoder,
                                    lambda_model=config.lambda_model,
                                )
                            )
                        if config.include_gumbel and config.include_heuristic:
                            for seed in config.gumbel_seeds:
                                rows.append(
                                    _run_gumbel(
                                        instance,
                                        method="gumbel_heuristic",
                                        guided=False,
                                        benchmark_name=benchmark_name,
                                        beam_width=beam_width,
                                        candidate_k=candidate_k,
                                        tier2_m=tier2_m,
                                        expansion_budget=expansion_budget,
                                        ranker=None,
                                        encoder=None,
                                        lambda_model=0.0,
                                        seed=seed,
                                        config=config,
                                    )
                                )
                        if config.include_gumbel and config.include_guided:
                            for seed in config.gumbel_seeds:
                                rows.append(
                                    _run_gumbel(
                                        instance,
                                        method="gumbel_guided",
                                        guided=True,
                                        benchmark_name=benchmark_name,
                                        beam_width=beam_width,
                                        candidate_k=candidate_k,
                                        tier2_m=tier2_m,
                                        expansion_budget=expansion_budget,
                                        ranker=ranker,
                                        encoder=encoder,
                                        lambda_model=config.lambda_model,
                                        seed=seed,
                                        config=config,
                                    )
                                )
    return rows


def summarize_planner_sweep(rows: Sequence[PlannerSweepRow]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[PlannerSweepRow]] = defaultdict(list)
    for row in rows:
        grouped[_summary_key(row)].append(row)

    summaries: list[dict[str, Any]] = []
    for key, group in sorted(grouped.items(), key=lambda item: item[0]):
        (
            method,
            planner,
            guided,
            family,
            intended_complexity,
            beam_width,
            candidate_k,
            tier2_m,
            expansion_budget,
            lambda_model,
        ) = key
        successes = [row for row in group if row.success]
        best_ops = [row.best_ops for row in successes if row.best_ops is not None]
        expansion_values = [row.expansions for row in group]
        runtime_values = [row.runtime_sec for row in group]
        seeds = {row.seed for row in group if row.seed is not None}
        summary = {
            "method": method,
            "planner": planner,
            "guided": guided,
            "family": family,
            "intended_complexity": intended_complexity,
            "beam_width": beam_width,
            "candidate_k": candidate_k,
            "tier2_m": tier2_m,
            "expansion_budget": expansion_budget,
            "lambda_model": lambda_model,
            "count": len(group),
            "num_seeds": len(seeds) if planner == "gumbel" else None,
            "success_rate": len(successes) / len(group),
            "success_rate_over_seeds": (
                len(successes) / len(group) if planner == "gumbel" else None
            ),
            "avg_best_ops": sum(best_ops) / len(best_ops) if best_ops else None,
            "avg_expansions": sum(expansion_values) / len(expansion_values),
            "median_expansions": float(median(expansion_values)),
            "avg_runtime_sec": sum(runtime_values) / len(runtime_values),
            "avg_candidate_generation_calls": _avg_optional(
                row.candidate_generation_calls for row in group
            ),
            "avg_total_candidates_generated": _avg_optional(
                row.total_candidates_generated for row in group
            ),
            "avg_total_candidates_scored": _avg_optional(
                row.total_candidates_scored for row in group
            ),
            "avg_model_forward_calls": _avg_optional(
                row.model_forward_calls for row in group
            ),
            "avg_root_expansions": _avg_optional(row.root_expansions for row in group),
            "avg_rollout_expansions": _avg_optional(
                row.rollout_expansions for row in group
            ),
        }
        summaries.append(summary)
    return summaries


def summarize_planner_deltas(rows: Sequence[PlannerSweepRow]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], dict[str, list[PlannerSweepRow]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        grouped[_delta_group_key(row)][row.method].append(row)

    comparisons = (
        ("gumbel_heuristic", "beam_heuristic"),
        ("beam_guided", "beam_heuristic"),
        ("gumbel_guided", "gumbel_heuristic"),
        ("gumbel_guided", "beam_guided"),
    )
    deltas: list[dict[str, Any]] = []
    for key, by_method in sorted(grouped.items(), key=lambda item: item[0]):
        for better_method, base_method in comparisons:
            if better_method not in by_method or base_method not in by_method:
                continue
            candidate_rows = by_method[better_method]
            base_rows = by_method[base_method]
            candidate_instances = _instance_set(candidate_rows)
            base_instances = _instance_set(base_rows)
            if candidate_instances != base_instances:
                continue
            candidate_stats = _method_stats(candidate_rows)
            base_stats = _method_stats(base_rows)
            (
                family,
                intended_complexity,
                beam_width,
                candidate_k,
                tier2_m,
                expansion_budget,
            ) = key
            base_lambda = _single_lambda_model(base_rows)
            candidate_lambda = _single_lambda_model(candidate_rows)
            deltas.append(
                {
                    "comparison": f"{better_method} - {base_method}",
                    "family": family,
                    "intended_complexity": intended_complexity,
                    "beam_width": beam_width,
                    "candidate_k": candidate_k,
                    "tier2_m": tier2_m,
                    "expansion_budget": expansion_budget,
                    "base_lambda_model": base_lambda,
                    "candidate_lambda_model": candidate_lambda,
                    "base_method": base_method,
                    "candidate_method": better_method,
                    "base_count": len(base_rows),
                    "candidate_count": len(candidate_rows),
                    "instance_set_size": len(base_instances),
                    "base_success_rate": base_stats["success_rate"],
                    "candidate_success_rate": candidate_stats["success_rate"],
                    "solve_rate_delta": (
                        candidate_stats["success_rate"] - base_stats["success_rate"]
                    ),
                    "avg_expansion_delta": (
                        candidate_stats["avg_expansions"] - base_stats["avg_expansions"]
                    ),
                }
            )
    return deltas


def _run_beam(
    instance: ProblemInstance,
    *,
    method: str,
    guided: bool,
    benchmark_name: str,
    beam_width: int,
    candidate_k: int,
    tier2_m: int,
    expansion_budget: int,
    ranker: CandidateRanker | None,
    encoder: CandidateFeatureEncoder | None,
    lambda_model: float,
) -> PlannerSweepRow:
    start = time.perf_counter()
    history = beam_search(
        instance,
        ranker=ranker,
        encoder=encoder,
        lambda_model=lambda_model,
        beam_width=beam_width,
        candidate_k=candidate_k,
        tier2_m=tier2_m,
        expansion_budget=expansion_budget,
    )
    return _row_from_history(
        instance,
        history=history,
        method=method,
        planner="beam",
        guided=guided,
        benchmark_name=benchmark_name,
        beam_width=beam_width,
        candidate_k=candidate_k,
        tier2_m=tier2_m,
        expansion_budget=expansion_budget,
        lambda_model=lambda_model,
        seed=None,
        runtime_sec=time.perf_counter() - start,
    )


def _run_gumbel(
    instance: ProblemInstance,
    *,
    method: str,
    guided: bool,
    benchmark_name: str,
    beam_width: int,
    candidate_k: int,
    tier2_m: int,
    expansion_budget: int,
    ranker: CandidateRanker | None,
    encoder: CandidateFeatureEncoder | None,
    lambda_model: float,
    seed: int,
    config: PlannerSweepConfig,
) -> PlannerSweepRow:
    start = time.perf_counter()
    history = gumbel_search(
        instance,
        ranker=ranker,
        encoder=encoder,
        lambda_model=lambda_model,
        candidate_k=candidate_k,
        tier2_m=tier2_m,
        initial_width=config.gumbel_initial_width,
        num_rounds=config.gumbel_rounds,
        rollout_depth=config.rollout_depth,
        gumbel_scale=config.gumbel_scale,
        expansion_budget=expansion_budget,
        seed=seed,
    )
    return _row_from_history(
        instance,
        history=history,
        method=method,
        planner="gumbel",
        guided=guided,
        benchmark_name=benchmark_name,
        beam_width=beam_width,
        candidate_k=candidate_k,
        tier2_m=tier2_m,
        expansion_budget=expansion_budget,
        lambda_model=lambda_model,
        seed=seed,
        runtime_sec=time.perf_counter() - start,
    )


def _row_from_history(
    instance: ProblemInstance,
    *,
    history: SearchHistory,
    method: str,
    planner: str,
    guided: bool,
    benchmark_name: str,
    beam_width: int,
    candidate_k: int,
    tier2_m: int,
    expansion_budget: int,
    lambda_model: float,
    seed: int | None,
    runtime_sec: float,
) -> PlannerSweepRow:
    best = history.best_finished()
    return PlannerSweepRow(
        method=method,
        planner=planner,
        guided=guided,
        benchmark_name=str(instance.metadata.get("benchmark_name", benchmark_name)),
        instance_id=str(instance.metadata.get("id", instance.metadata.get("target_id", ""))),
        family=str(instance.metadata.get("family", instance.family_name)),
        intended_complexity=_metadata_int(instance, "intended_complexity"),
        generative_ops=_metadata_int(instance, "generative_ops"),
        beam_width=beam_width,
        candidate_k=candidate_k,
        tier2_m=tier2_m,
        expansion_budget=expansion_budget,
        lambda_model=lambda_model,
        seed=seed,
        success=best is not None,
        best_ops=best.num_ops() if best is not None else None,
        expansions=history_expansion_count(history),
        runtime_sec=runtime_sec,
        candidate_generation_calls=_metadata_int_from_history(
            history,
            "candidate_generation_calls",
        ),
        total_candidates_generated=_metadata_int_from_history(
            history,
            "total_candidates_generated",
        ),
        total_candidates_scored=_metadata_int_from_history(
            history,
            "total_candidates_scored",
        ),
        model_forward_calls=_metadata_int_from_history(history, "model_forward_calls"),
        root_expansions=_metadata_int_from_history(history, "root_expansions"),
        rollout_expansions=_metadata_int_from_history(history, "rollout_expansions"),
    )


def _summary_key(row: PlannerSweepRow) -> tuple[Any, ...]:
    return (
        row.method,
        row.planner,
        row.guided,
        row.family,
        row.intended_complexity,
        row.beam_width,
        row.candidate_k,
        row.tier2_m,
        row.expansion_budget,
        row.lambda_model,
    )


def _delta_group_key(row: PlannerSweepRow) -> tuple[Any, ...]:
    return (
        row.family,
        row.intended_complexity,
        row.beam_width,
        row.candidate_k,
        row.tier2_m,
        row.expansion_budget,
    )


def _method_stats(rows: Sequence[PlannerSweepRow]) -> dict[str, float]:
    return {
        "success_rate": sum(1 for row in rows if row.success) / len(rows),
        "avg_expansions": sum(row.expansions for row in rows) / len(rows),
    }


def _single_lambda_model(rows: Sequence[PlannerSweepRow]) -> float | None:
    values = {row.lambda_model for row in rows}
    if len(values) == 1:
        return next(iter(values))
    return None


def _instance_set(rows: Sequence[PlannerSweepRow]) -> frozenset[str]:
    return frozenset(row.instance_id for row in rows)


def _avg_optional(values: Sequence[int | None] | Any) -> float | None:
    present = [value for value in values if value is not None]
    if not present:
        return None
    return sum(present) / len(present)


def _metadata_int(instance: ProblemInstance, key: str) -> int | None:
    value = instance.metadata.get(key)
    if type(value) is int:
        return value
    return None


def _metadata_int_from_history(history: SearchHistory, key: str) -> int | None:
    value = history.metadata.get(key)
    if type(value) is int:
        return value
    return None


def _validate_positive_int_tuple(name: str, values: tuple[int, ...]) -> None:
    if not values:
        raise ValueError(f"{name} must be non-empty")
    for value in values:
        _require_positive_int(name, value)


def _validate_non_negative_int_tuple(name: str, values: tuple[int, ...]) -> None:
    if not values:
        raise ValueError(f"{name} must be non-empty")
    for value in values:
        _require_non_negative_int(name, value)


def _validate_int_tuple(name: str, values: tuple[int, ...]) -> None:
    if not values:
        raise ValueError(f"{name} must be non-empty")
    for value in values:
        if type(value) is not int:
            raise ValueError(f"{name} must contain ints")


def _require_positive_int(name: str, value: int) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive int")


def _require_non_negative_int(name: str, value: int) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative int")


def _require_non_negative_float(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    if float(value) < 0.0:
        raise ValueError(f"{name} must be non-negative")
