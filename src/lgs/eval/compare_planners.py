"""Compare beam search with Gumbel / Sequential-Halving search."""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass

from lgs.env.problem_instance import ProblemInstance
from lgs.models.candidate_ranker import CandidateRanker
from lgs.models.feature_encoder import CandidateFeatureEncoder
from lgs.search.beam_search import beam_search
from lgs.search.gumbel_search import gumbel_search
from lgs.search.search_history import history_expansion_count


@dataclass(frozen=True)
class PlannerComparisonResult:
    planner: str
    seed: int | None
    instance_id: str
    family: str
    intended_complexity: int | None
    success: bool
    best_ops: int | None
    expansions: int
    runtime_sec: float


def compare_beam_vs_gumbel(
    instances: Sequence[ProblemInstance],
    *,
    ranker: CandidateRanker | None = None,
    encoder: CandidateFeatureEncoder | None = None,
    lambda_model: float = 0.0,
    beam_width: int = 4,
    candidate_k: int = 32,
    tier2_m: int = 128,
    gumbel_initial_width: int = 16,
    gumbel_rounds: int = 3,
    rollout_depth: int = 1,
    gumbel_scale: float = 1.0,
    expansion_budget: int | None = None,
    seed: int = 0,
) -> list[PlannerComparisonResult]:
    rows: list[PlannerComparisonResult] = []
    for instance in instances:
        start = time.perf_counter()
        beam_history = beam_search(
            instance,
            ranker=ranker,
            encoder=encoder,
            lambda_model=lambda_model,
            beam_width=beam_width,
            candidate_k=candidate_k,
            tier2_m=tier2_m,
            expansion_budget=expansion_budget,
        )
        rows.append(
            _result_row(
                planner="beam",
                seed=None,
                instance=instance,
                runtime_sec=time.perf_counter() - start,
                expansions=history_expansion_count(beam_history),
                success=beam_history.success(),
                best_ops=(
                    best.num_ops()
                    if (best := beam_history.best_finished()) is not None
                    else None
                ),
            )
        )

        start = time.perf_counter()
        gumbel_history = gumbel_search(
            instance,
            ranker=ranker,
            encoder=encoder,
            lambda_model=lambda_model,
            candidate_k=candidate_k,
            tier2_m=tier2_m,
            initial_width=gumbel_initial_width,
            num_rounds=gumbel_rounds,
            rollout_depth=rollout_depth,
            gumbel_scale=gumbel_scale,
            expansion_budget=expansion_budget,
            seed=seed,
        )
        rows.append(
            _result_row(
                planner="gumbel",
                seed=seed,
                instance=instance,
                runtime_sec=time.perf_counter() - start,
                expansions=history_expansion_count(gumbel_history),
                success=gumbel_history.success(),
                best_ops=(
                    best.num_ops()
                    if (best := gumbel_history.best_finished()) is not None
                    else None
                ),
            )
        )
    return rows


def _result_row(
    *,
    planner: str,
    seed: int | None,
    instance: ProblemInstance,
    runtime_sec: float,
    expansions: int,
    success: bool,
    best_ops: int | None,
) -> PlannerComparisonResult:
    return PlannerComparisonResult(
        planner=planner,
        seed=seed,
        instance_id=str(instance.metadata.get("id", instance.metadata.get("target_id", ""))),
        family=str(instance.metadata.get("family", instance.family_name)),
        intended_complexity=_metadata_int(instance, "intended_complexity"),
        success=success,
        best_ops=best_ops,
        expansions=expansions,
        runtime_sec=runtime_sec,
    )


def _metadata_int(instance: ProblemInstance, key: str) -> int | None:
    value = instance.metadata.get(key)
    if type(value) is int:
        return value
    return None
