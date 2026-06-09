"""Candidate-set recall diagnostics for known action traces."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from lgs.env.action import Action
from lgs.env.circuit_state import (
    BudgetExceededError,
    CircuitState,
    InvalidActionError,
)
from lgs.env.problem_instance import ProblemInstance
from lgs.poly.fast_poly import PolynomialDegreeError
from lgs.search.candidate_generator import generate_candidates
from lgs.search.search_history import SearchHistory


@dataclass(frozen=True)
class CandidateRecallStep:
    step: int
    action: Action
    k: int
    hit: bool
    rank: int | None
    candidate_count: int
    heuristic_score: float | None
    total_score: float | None
    source_tags: tuple[str, ...]


@dataclass
class CandidateRecallReport:
    instance_id: str
    family: str
    intended_complexity: int | None
    steps: list[CandidateRecallStep]

    def recall_at(self, k: int) -> float:
        matching = [step for step in self.steps if step.k == k]
        if not matching:
            return 0.0
        return sum(1 for step in matching if step.hit) / len(matching)

    def mean_rank_at(self, k: int) -> float | None:
        ranks = [
            step.rank
            for step in self.steps
            if step.k == k and step.rank is not None
        ]
        if not ranks:
            return None
        return sum(ranks) / len(ranks)

    def misses_at(self, k: int) -> list[CandidateRecallStep]:
        return [
            step
            for step in self.steps
            if step.k == k and not step.hit
        ]


def compute_candidate_recall_for_trace(
    instance: ProblemInstance,
    trace: Sequence[Action],
    *,
    candidate_ks: Sequence[int] = (4, 8, 16, 32, 64),
    tier2_m: int = 128,
) -> CandidateRecallReport:
    if not isinstance(instance, ProblemInstance):
        raise TypeError("instance must be a ProblemInstance")
    candidate_ks = _validate_candidate_ks(candidate_ks)
    if type(tier2_m) is not int or tier2_m < 0:
        raise ValueError("tier2_m must be a non-negative int")

    state = CircuitState.initial(instance)
    steps: list[CandidateRecallStep] = []
    for step_index, action in enumerate(trace):
        if not isinstance(action, Action):
            raise ValueError(f"trace step {step_index} is not an Action")

        for k in candidate_ks:
            candidates = generate_candidates(
                instance,
                state,
                K=k,
                tier2_m=tier2_m,
            )
            match_index = next(
                (
                    index
                    for index, candidate in enumerate(candidates)
                    if candidate.action == action
                ),
                None,
            )
            matched = candidates[match_index] if match_index is not None else None
            steps.append(
                CandidateRecallStep(
                    step=step_index,
                    action=action,
                    k=k,
                    hit=matched is not None,
                    rank=(match_index + 1) if match_index is not None else None,
                    candidate_count=len(candidates),
                    heuristic_score=(
                        matched.heuristic_score if matched is not None else None
                    ),
                    total_score=matched.total_score if matched is not None else None,
                    source_tags=(
                        tuple(sorted(matched.source_tags))
                        if matched is not None
                        else ()
                    ),
                )
            )

        try:
            state = state.apply(action)
        except (InvalidActionError, BudgetExceededError, PolynomialDegreeError) as exc:
            raise ValueError(
                f"trace action at step {step_index} is invalid: {action}"
            ) from exc

    return CandidateRecallReport(
        instance_id=str(instance.metadata.get("id", instance.metadata.get("target_id", ""))),
        family=str(instance.metadata.get("family", instance.family_name)),
        intended_complexity=_metadata_int(instance, "intended_complexity"),
        steps=steps,
    )


def compute_recall_for_best_finished(
    history: SearchHistory,
    *,
    candidate_ks: Sequence[int] = (4, 8, 16, 32, 64),
    tier2_m: int = 128,
) -> CandidateRecallReport | None:
    if not isinstance(history, SearchHistory):
        raise TypeError("history must be a SearchHistory")
    best = history.best_finished()
    if best is None:
        return None
    return compute_candidate_recall_for_trace(
        history.instance,
        tuple(best.actions),
        candidate_ks=candidate_ks,
        tier2_m=tier2_m,
    )


def _validate_candidate_ks(candidate_ks: Sequence[int]) -> tuple[int, ...]:
    values = tuple(candidate_ks)
    if not values:
        raise ValueError("candidate_ks must be non-empty")
    for value in values:
        if type(value) is not int or value < 0:
            raise ValueError("candidate_ks must contain non-negative ints")
    return values


def _metadata_int(instance: ProblemInstance, key: str) -> int | None:
    value = instance.metadata.get(key)
    if type(value) is int:
        return value
    return None
