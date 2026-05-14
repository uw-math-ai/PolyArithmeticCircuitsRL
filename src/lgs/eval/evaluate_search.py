"""Evaluation helpers for exact beam search."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from lgs.env.problem_instance import ProblemInstance
from lgs.models.candidate_ranker import CandidateRanker
from lgs.models.feature_encoder import CandidateFeatureEncoder
from lgs.search.beam_search import beam_search


@dataclass(frozen=True)
class SearchEvalMetrics:
    success_rate: float
    avg_best_ops: float | None
    avg_expansions: float
    num_instances: int


def evaluate_beam_search(
    instances: Sequence[ProblemInstance],
    *,
    ranker: CandidateRanker | None = None,
    encoder: CandidateFeatureEncoder | None = None,
    lambda_model: float = 0.0,
    beam_width: int = 16,
    candidate_k: int = 64,
    tier2_m: int = 128,
) -> SearchEvalMetrics:
    histories = [
        beam_search(
            instance,
            ranker=ranker,
            encoder=encoder,
            lambda_model=lambda_model,
            beam_width=beam_width,
            candidate_k=candidate_k,
            tier2_m=tier2_m,
        )
        for instance in instances
    ]
    num_instances = len(histories)
    if num_instances == 0:
        return SearchEvalMetrics(
            success_rate=0.0,
            avg_best_ops=None,
            avg_expansions=0.0,
            num_instances=0,
        )

    best_ops = [
        best.num_ops()
        for history in histories
        if (best := history.best_finished()) is not None
    ]
    avg_best_ops = (
        sum(best_ops) / len(best_ops)
        if best_ops
        else None
    )
    return SearchEvalMetrics(
        success_rate=len(best_ops) / num_instances,
        avg_best_ops=avg_best_ops,
        avg_expansions=sum(len(history.records) for history in histories) / num_instances,
        num_instances=num_instances,
    )
