"""Compare heuristic-only and ranker-guided beam search."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from lgs.env.problem_instance import ProblemInstance
from lgs.search.beam_search import beam_search


@dataclass(frozen=True)
class SearchComparisonResult:
    family_name: str
    target_id: str
    heuristic_success: bool
    guided_success: bool
    heuristic_best_ops: int | None
    guided_best_ops: int | None
    heuristic_expansions: int
    guided_expansions: int


def compare_heuristic_vs_ranker(
    instances: Sequence[ProblemInstance],
    ranker,
    encoder,
    *,
    lambda_model: float,
    beam_width: int,
    candidate_k: int,
    tier2_m: int,
) -> list[SearchComparisonResult]:
    results: list[SearchComparisonResult] = []
    for index, instance in enumerate(instances):
        heuristic = beam_search(
            instance,
            beam_width=beam_width,
            candidate_k=candidate_k,
            tier2_m=tier2_m,
        )
        guided = beam_search(
            instance,
            ranker=ranker,
            encoder=encoder,
            lambda_model=lambda_model,
            beam_width=beam_width,
            candidate_k=candidate_k,
            tier2_m=tier2_m,
        )
        heuristic_best = heuristic.best_finished()
        guided_best = guided.best_finished()
        results.append(
            SearchComparisonResult(
                family_name=instance.family_name,
                target_id=str(instance.metadata.get("target_id", index)),
                heuristic_success=heuristic.success(),
                guided_success=guided.success(),
                heuristic_best_ops=(
                    heuristic_best.num_ops() if heuristic_best is not None else None
                ),
                guided_best_ops=guided_best.num_ops() if guided_best is not None else None,
                heuristic_expansions=len(heuristic.records),
                guided_expansions=len(guided.records),
            )
        )
    return results
