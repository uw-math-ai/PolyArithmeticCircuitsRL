"""Search sweep helpers for heuristic and ranker-guided beam search."""

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


@dataclass(frozen=True)
class SweepConfig:
    beam_widths: tuple[int, ...] = (1, 2, 4, 8, 16)
    candidate_ks: tuple[int, ...] = (4, 8, 16, 32, 64)
    tier2_ms: tuple[int, ...] = (128,)
    lambda_model: float = 1.0
    learned_only: bool = False
    noise_sigma: float = 0.0

    def __post_init__(self) -> None:
        _validate_positive_int_tuple("beam_widths", self.beam_widths)
        _validate_non_negative_int_tuple("candidate_ks", self.candidate_ks)
        _validate_non_negative_int_tuple("tier2_ms", self.tier2_ms)
        if isinstance(self.lambda_model, bool) or not isinstance(
            self.lambda_model,
            (int, float),
        ):
            raise ValueError("lambda_model must be numeric")
        if float(self.lambda_model) < 0.0:
            raise ValueError("lambda_model must be non-negative")
        if not isinstance(self.learned_only, bool):
            raise ValueError("learned_only must be a bool")
        if isinstance(self.noise_sigma, bool) or not isinstance(self.noise_sigma, (int, float)):
            raise ValueError("noise_sigma must be numeric")
        if float(self.noise_sigma) < 0.0:
            raise ValueError("noise_sigma must be non-negative")


@dataclass
class SweepRow:
    method: str
    benchmark_name: str
    instance_id: str
    family: str
    beam_width: int
    candidate_k: int
    tier2_m: int
    success: bool
    best_ops: int | None
    expansions: int
    runtime_sec: float
    intended_complexity: int | None = None
    generative_ops: int | None = None


def run_search_sweep(
    instances: Sequence[ProblemInstance],
    *,
    ranker: CandidateRanker | None,
    encoder: CandidateFeatureEncoder | None,
    config: SweepConfig,
) -> list[SweepRow]:
    if not isinstance(config, SweepConfig):
        raise TypeError("config must be a SweepConfig")
    if (ranker is None) != (encoder is None):
        raise ValueError("ranker and encoder must be provided together")

    rows: list[SweepRow] = []
    for beam_width in config.beam_widths:
        for candidate_k in config.candidate_ks:
            for tier2_m in config.tier2_ms:
                for instance in instances:
                    rows.append(
                        _run_one(
                            instance,
                            method="noisy_heuristic" if config.noise_sigma > 0.0 else "heuristic",
                            beam_width=beam_width,
                            candidate_k=candidate_k,
                            tier2_m=tier2_m,
                            ranker=None,
                            encoder=None,
                            lambda_model=0.0,
                            noise_sigma=config.noise_sigma,
                        )
                    )
                    if ranker is not None and encoder is not None:
                        rows.append(
                            _run_one(
                                instance,
                                method="guided",
                                beam_width=beam_width,
                                candidate_k=candidate_k,
                                tier2_m=tier2_m,
                                ranker=ranker,
                                encoder=encoder,
                                lambda_model=config.lambda_model,
                                learned_only=config.learned_only,
                            )
                        )
    return rows


def summarize_sweep(rows: Sequence[SweepRow]) -> list[dict[str, Any]]:
    grouped: dict[
        tuple[str, str, int | None, int, int, int],
        list[SweepRow],
    ] = defaultdict(list)
    for row in rows:
        grouped[
            (
                row.method,
                row.family,
                row.intended_complexity,
                row.beam_width,
                row.candidate_k,
                row.tier2_m,
            )
        ].append(row)

    summaries: list[dict[str, Any]] = []
    solve_rates: dict[
        tuple[str, int | None, int, int, int],
        dict[str, float],
    ] = defaultdict(dict)
    for key, group in sorted(
        grouped.items(),
        key=lambda item: _summary_group_sort_key(item[0]),
    ):
        method, family, intended_complexity, beam_width, candidate_k, tier2_m = key
        successes = [row for row in group if row.success]
        solve_rate = len(successes) / len(group)
        best_ops = [row.best_ops for row in successes if row.best_ops is not None]
        expansion_values = [row.expansions for row in group]
        runtime_values = [row.runtime_sec for row in group]
        summary = {
            "method": method,
            "family": family,
            "intended_complexity": intended_complexity,
            "beam_width": beam_width,
            "candidate_k": candidate_k,
            "tier2_m": tier2_m,
            "count": len(group),
            "solve_rate": solve_rate,
            "avg_best_ops": (sum(best_ops) / len(best_ops)) if best_ops else None,
            "avg_expansions": sum(expansion_values) / len(expansion_values),
            "median_expansions": float(median(expansion_values)),
            "avg_runtime_sec": sum(runtime_values) / len(runtime_values),
            "solved_by_complexity": _solved_by_complexity(group),
        }
        summaries.append(summary)
        solve_rates[
            (family, intended_complexity, beam_width, candidate_k, tier2_m)
        ][method] = solve_rate

    for summary in summaries:
        rates = solve_rates[
            (
                summary["family"],
                summary["intended_complexity"],
                summary["beam_width"],
                summary["candidate_k"],
                summary["tier2_m"],
            )
        ]
        if "guided" in rates and "heuristic" in rates:
            summary["guided_minus_heuristic_solve_rate"] = (
                rates["guided"] - rates["heuristic"]
            )
        else:
            summary["guided_minus_heuristic_solve_rate"] = None
    return summaries


def summarize_failures(rows: Sequence[SweepRow]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int | None, str, int, int, int], list[SweepRow]] = (
        defaultdict(list)
    )
    for row in rows:
        if row.success:
            continue
        grouped[
            (
                row.family,
                row.intended_complexity,
                row.method,
                row.beam_width,
                row.candidate_k,
                row.tier2_m,
            )
        ].append(row)

    summaries: list[dict[str, Any]] = []
    for key, group in sorted(grouped.items(), key=lambda item: _failure_group_sort_key(item[0])):
        family, complexity, method, beam_width, candidate_k, tier2_m = key
        summaries.append(
            {
                "family": family,
                "intended_complexity": complexity,
                "method": method,
                "beam_width": beam_width,
                "candidate_k": candidate_k,
                "tier2_m": tier2_m,
                "failure_count": len(group),
                "avg_expansions": sum(row.expansions for row in group) / len(group),
                "instance_ids": [row.instance_id for row in group],
            }
        )
    return summaries


def _run_one(
    instance: ProblemInstance,
    *,
    method: str,
    beam_width: int,
    candidate_k: int,
    tier2_m: int,
    ranker: CandidateRanker | None,
    encoder: CandidateFeatureEncoder | None,
    lambda_model: float,
    learned_only: bool = False,
    noise_sigma: float = 0.0,
) -> SweepRow:
    start = time.perf_counter()
    history = beam_search(
        instance,
        ranker=ranker,
        encoder=encoder,
        lambda_model=lambda_model,
        learned_only=learned_only,
        beam_width=beam_width,
        candidate_k=candidate_k,
        tier2_m=tier2_m,
        noise_sigma=noise_sigma,
    )
    runtime_sec = time.perf_counter() - start
    best = history.best_finished()
    return SweepRow(
        method=method,
        benchmark_name=str(instance.metadata.get("benchmark_name", "benchmark")),
        instance_id=str(instance.metadata.get("id", instance.metadata.get("target_id", ""))),
        family=str(instance.metadata.get("family", instance.family_name)),
        beam_width=beam_width,
        candidate_k=candidate_k,
        tier2_m=tier2_m,
        success=best is not None,
        best_ops=best.num_ops() if best is not None else None,
        expansions=len(history.records),
        runtime_sec=runtime_sec,
        intended_complexity=_metadata_int(instance, "intended_complexity"),
        generative_ops=_metadata_int(instance, "generative_ops"),
    )


def _solved_by_complexity(group: Sequence[SweepRow]) -> dict[str, float]:
    by_complexity: dict[int, list[SweepRow]] = defaultdict(list)
    for row in group:
        if row.intended_complexity is None:
            continue
        by_complexity[row.intended_complexity].append(row)
    return {
        str(complexity): len([row for row in rows if row.success]) / len(rows)
        for complexity, rows in sorted(by_complexity.items())
    }


def _metadata_int(instance: ProblemInstance, key: str) -> int | None:
    value = instance.metadata.get(key)
    if type(value) is int:
        return value
    return None


def _failure_group_sort_key(
    key: tuple[str, int | None, str, int, int, int],
) -> tuple[str, int, str, int, int, int]:
    family, complexity, method, beam_width, candidate_k, tier2_m = key
    normalized_complexity = -1 if complexity is None else complexity
    return (family, normalized_complexity, method, beam_width, candidate_k, tier2_m)


def _summary_group_sort_key(
    key: tuple[str, str, int | None, int, int, int],
) -> tuple[str, str, int, int, int, int]:
    method, family, complexity, beam_width, candidate_k, tier2_m = key
    normalized_complexity = -1 if complexity is None else complexity
    return (method, family, normalized_complexity, beam_width, candidate_k, tier2_m)


def _validate_positive_int_tuple(name: str, values: tuple[int, ...]) -> None:
    if not values:
        raise ValueError(f"{name} must be non-empty")
    for value in values:
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must contain positive ints")


def _validate_non_negative_int_tuple(name: str, values: tuple[int, ...]) -> None:
    if not values:
        raise ValueError(f"{name} must be non-empty")
    for value in values:
        if type(value) is not int or value < 0:
            raise ValueError(f"{name} must contain non-negative ints")
