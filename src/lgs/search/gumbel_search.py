"""Gumbel / Sequential-Halving planner over exact symbolic candidates."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Hashable

import torch

from lgs.env.candidate import Candidate
from lgs.env.circuit_state import CircuitState
from lgs.env.problem_instance import ProblemInstance
from lgs.models.candidate_ranker import CandidateRanker
from lgs.models.feature_encoder import CandidateFeatureEncoder
from lgs.poly.fast_poly import PolynomialDegreeError
from lgs.poly.poly_utils import require_same_domain
from lgs.search.candidate_generator import generate_candidates
from lgs.search.search_history import ExpandedStateRecord, SearchHistory

TERMINAL_BONUS = 1000.0
ROLLOUT_TERMINAL_BONUS = 500.0
DEPTH_PENALTY = 0.01
MUL_COVERAGE_WEIGHT = 20.0
ADD_COVERAGE_WEIGHT = 10.0
OUTSIDE_FRACTION_WEIGHT = 5.0


@dataclass
class _BudgetCounter:
    expansion_budget: int | None
    used: int = 0

    def can_expand(self) -> bool:
        return self.expansion_budget is None or self.used < self.expansion_budget

    def consume(self) -> None:
        if not self.can_expand():
            raise RuntimeError("expansion budget exhausted")
        self.used += 1


@dataclass(frozen=True)
class _EvaluatedBranch:
    candidate: Candidate
    next_state: CircuitState
    score: float


def gumbel_search(
    instance: ProblemInstance,
    *,
    ranker: CandidateRanker | None = None,
    encoder: CandidateFeatureEncoder | None = None,
    lambda_model: float = 0.0,
    candidate_k: int = 64,
    tier2_m: int = 128,
    max_depth: int | None = None,
    num_rounds: int = 3,
    initial_width: int = 16,
    keep_per_round: int | None = None,
    rollout_depth: int = 1,
    gumbel_scale: float = 1.0,
    seed: int = 0,
    expansion_budget: int | None = None,
) -> SearchHistory:
    _validate_inputs(
        instance=instance,
        ranker=ranker,
        encoder=encoder,
        lambda_model=lambda_model,
        candidate_k=candidate_k,
        tier2_m=tier2_m,
        max_depth=max_depth,
        num_rounds=num_rounds,
        initial_width=initial_width,
        keep_per_round=keep_per_round,
        rollout_depth=rollout_depth,
        gumbel_scale=gumbel_scale,
        seed=seed,
        expansion_budget=expansion_budget,
    )

    depth_limit = instance.op_budget if max_depth is None else max_depth
    history = SearchHistory(
        instance=instance,
        num_expansions=0,
        metadata=_initial_accounting_metadata(seed=seed),
    )
    initial_state = CircuitState.initial(instance)
    if initial_state.contains(instance.target):
        history.finished.append(initial_state)
        return history

    rng = random.Random(seed)
    budget = _BudgetCounter(expansion_budget=expansion_budget)
    frontier: list[tuple[CircuitState, float]] = [(initial_state, 0.0)]
    survivor_count = (
        max(1, initial_width // 2)
        if keep_per_round is None
        else keep_per_round
    )

    for depth in range(depth_limit):
        if not budget.can_expand():
            break
        next_frontier: list[tuple[CircuitState, float]] = []
        for state, _ in frontier:
            if not budget.can_expand():
                break
            if state.remaining_budget() <= 0:
                continue
            candidates = generate_candidates(
                instance,
                state,
                K=candidate_k,
                tier2_m=tier2_m,
            )
            _record_candidate_generation(history, candidates)
            _score_candidates(
                instance=instance,
                state=state,
                candidates=candidates,
                history=history,
                ranker=ranker,
                encoder=encoder,
                lambda_model=float(lambda_model),
            )
            candidates = sorted(candidates, key=_candidate_sort_key)
            survivors = _sequential_halving_select(
                instance=instance,
                state=state,
                candidates=candidates,
                history=history,
                depth=depth,
                ranker=ranker,
                encoder=encoder,
                lambda_model=float(lambda_model),
                initial_width=initial_width,
                keep_per_round=survivor_count,
                num_rounds=num_rounds,
                rollout_depth=rollout_depth,
                gumbel_scale=float(gumbel_scale),
                rng=rng,
                budget=budget,
            )
            next_frontier.extend(
                (branch.next_state, branch.score)
                for branch in survivors
                if branch.next_state.remaining_budget() > 0
            )

        if not next_frontier:
            break
        frontier = _select_frontier(next_frontier, initial_width)

    return history


def _score_candidates(
    *,
    instance: ProblemInstance,
    state: CircuitState,
    candidates: list[Candidate],
    history: SearchHistory,
    ranker: CandidateRanker | None,
    encoder: CandidateFeatureEncoder | None,
    lambda_model: float,
) -> None:
    if lambda_model == 0.0 or ranker is None:
        for candidate in candidates:
            candidate.model_score = 0.0
            candidate.total_score = candidate.heuristic_score
        return

    was_training = ranker.training
    ranker.eval()
    try:
        device = _ranker_device(ranker)
        with torch.no_grad():
            for candidate in candidates:
                features = encoder.encode(instance, state, candidate)
                feature_tensor = torch.tensor([features], dtype=torch.float32, device=device)
                candidate.model_score = float(ranker(feature_tensor).item())
                history.metadata["model_forward_calls"] += 1
                candidate.total_score = (
                    candidate.heuristic_score
                    + lambda_model * candidate.model_score
                )
    finally:
        if was_training:
            ranker.train()


def _sample_gumbel_scores(
    candidates: list[Candidate],
    *,
    rng: random.Random,
    gumbel_scale: float,
) -> list[tuple[Candidate, float]]:
    scored: list[tuple[Candidate, float]] = []
    for candidate in candidates:
        u = min(max(rng.random(), 1e-12), 1.0 - 1e-12)
        noise = -math.log(-math.log(u))
        scored.append((candidate, candidate.total_score + gumbel_scale * noise))
    return scored


def _sequential_halving_select(
    *,
    instance: ProblemInstance,
    state: CircuitState,
    candidates: list[Candidate],
    history: SearchHistory,
    depth: int,
    ranker: CandidateRanker | None,
    encoder: CandidateFeatureEncoder | None,
    lambda_model: float,
    initial_width: int,
    keep_per_round: int,
    num_rounds: int,
    rollout_depth: int,
    gumbel_scale: float,
    rng: random.Random,
    budget: _BudgetCounter,
) -> list[_EvaluatedBranch]:
    if not candidates:
        return []

    sampled = sorted(
        _sample_gumbel_scores(candidates, rng=rng, gumbel_scale=gumbel_scale),
        key=lambda item: (-item[1], _candidate_action_key(item[0]), item[0].result_poly.key()),
    )[:initial_width]

    active_candidates = [candidate for candidate, _ in sampled]
    evaluated: list[_EvaluatedBranch] = []
    for candidate in active_candidates:
        if not budget.can_expand():
            break
        branch = _evaluate_candidate_branch(
            instance=instance,
            state=state,
            candidate=candidate,
            candidate_list=candidates,
            history=history,
            depth=depth,
            ranker=ranker,
            encoder=encoder,
            lambda_model=lambda_model,
            rollout_depth=rollout_depth,
            budget=budget,
        )
        if branch is not None:
            evaluated.append(branch)

    active = sorted(evaluated, key=_branch_sort_key)
    for _ in range(max(0, num_rounds - 1)):
        active = active[:keep_per_round]
        refreshed: list[_EvaluatedBranch] = []
        for branch in active:
            rollout_score = _greedy_rollout_score(
                instance=instance,
                state=branch.next_state,
                history=history,
                depth=depth + 1,
                ranker=ranker,
                encoder=encoder,
                lambda_model=lambda_model,
                rollout_depth=rollout_depth,
                budget=budget,
            )
            refreshed.append(
                _EvaluatedBranch(
                    candidate=branch.candidate,
                    next_state=branch.next_state,
                    score=(
                        branch.candidate.total_score
                        + _candidate_potential_score(branch.candidate)
                        + _terminal_bonus(instance, branch.next_state)
                        + rollout_score
                        - DEPTH_PENALTY * branch.next_state.num_ops()
                    ),
                )
            )
        active = sorted(refreshed, key=_branch_sort_key)

    return active[:keep_per_round]


def _evaluate_candidate_branch(
    *,
    instance: ProblemInstance,
    state: CircuitState,
    candidate: Candidate,
    candidate_list: list[Candidate],
    history: SearchHistory,
    depth: int,
    ranker: CandidateRanker | None,
    encoder: CandidateFeatureEncoder | None,
    lambda_model: float,
    rollout_depth: int,
    budget: _BudgetCounter,
) -> _EvaluatedBranch | None:
    if not budget.can_expand():
        return None
    try:
        next_state = state.apply(candidate.action)
    except PolynomialDegreeError:
        return None
    _consume_expansion(budget, history, kind="root")

    base_score = (
        candidate.total_score
        + _candidate_potential_score(candidate)
        + _terminal_bonus(instance, next_state)
        - DEPTH_PENALTY * next_state.num_ops()
    )
    record = ExpandedStateRecord(
        instance=instance,
        state=state,
        candidates=candidate_list,
        candidate=candidate,
        next_state=next_state,
        depth=depth,
        state_score=base_score,
    )
    history.records.append(record)
    if next_state.contains(instance.target):
        history.finished.append(next_state)

    rollout_score = _greedy_rollout_score(
        instance=instance,
        state=next_state,
        history=history,
        depth=depth + 1,
        ranker=ranker,
        encoder=encoder,
        lambda_model=lambda_model,
        rollout_depth=rollout_depth,
        budget=budget,
    )
    branch_score = base_score + rollout_score
    record.state_score = branch_score
    return _EvaluatedBranch(candidate=candidate, next_state=next_state, score=branch_score)


def _record_rollout_expansion(
    *,
    instance: ProblemInstance,
    state: CircuitState,
    candidates: list[Candidate],
    candidate: Candidate,
    next_state: CircuitState,
    history: SearchHistory,
    depth: int,
    state_score: float,
) -> None:
    history.records.append(
        ExpandedStateRecord(
            instance=instance,
            state=state,
            candidates=candidates,
            candidate=candidate,
            next_state=next_state,
            depth=depth,
            state_score=state_score,
        )
    )
    if next_state.contains(instance.target):
        history.finished.append(next_state)


def _greedy_rollout_score(
    *,
    instance: ProblemInstance,
    state: CircuitState,
    history: SearchHistory,
    depth: int,
    ranker: CandidateRanker | None,
    encoder: CandidateFeatureEncoder | None,
    lambda_model: float,
    rollout_depth: int,
    budget: _BudgetCounter,
) -> float:
    if rollout_depth <= 0 or state.contains(instance.target):
        return 0.0

    score = 0.0
    current = state
    for step in range(rollout_depth):
        if current.remaining_budget() <= 0 or not budget.can_expand():
            break
        candidates = generate_candidates(instance, current, K=16, tier2_m=64)
        _record_candidate_generation(history, candidates)
        _score_candidates(
            instance=instance,
            state=current,
            candidates=candidates,
            history=history,
            ranker=ranker,
            encoder=encoder,
            lambda_model=lambda_model,
        )
        for candidate in sorted(candidates, key=_candidate_sort_key):
            if not budget.can_expand():
                return score
            try:
                next_state = current.apply(candidate.action)
            except PolynomialDegreeError:
                continue
            _consume_expansion(budget, history, kind="rollout")
            step_score = (
                0.25 * candidate.total_score
                + _candidate_potential_score(candidate)
                + _terminal_bonus(instance, next_state)
                - DEPTH_PENALTY * next_state.num_ops()
            )
            score += step_score
            _record_rollout_expansion(
                instance=instance,
                state=current,
                candidates=candidates,
                candidate=candidate,
                next_state=next_state,
                history=history,
                depth=depth + step,
                state_score=step_score,
            )
            if next_state.contains(instance.target):
                score += ROLLOUT_TERMINAL_BONUS - DEPTH_PENALTY * (step + 1)
                return score
            current = next_state
            break
    return score


def _consume_expansion(
    budget: _BudgetCounter,
    history: SearchHistory,
    *,
    kind: str,
) -> None:
    budget.consume()
    history.num_expansions = budget.used
    if kind == "root":
        history.metadata["root_expansions"] += 1
    elif kind == "rollout":
        history.metadata["rollout_expansions"] += 1
    else:
        raise ValueError(f"unknown expansion kind {kind!r}")


def _initial_accounting_metadata(*, seed: int) -> dict[str, int | str]:
    return {
        "planner": "gumbel",
        "seed": seed,
        "candidate_generation_calls": 0,
        "total_candidates_generated": 0,
        "total_candidates_scored": 0,
        "model_forward_calls": 0,
        "root_expansions": 0,
        "rollout_expansions": 0,
    }


def _record_candidate_generation(
    history: SearchHistory,
    candidates: list[Candidate],
) -> None:
    history.metadata["candidate_generation_calls"] += 1
    history.metadata["total_candidates_generated"] += len(candidates)
    history.metadata["total_candidates_scored"] += len(candidates)


def _candidate_potential_score(candidate: Candidate) -> float:
    features = candidate.features
    return (
        MUL_COVERAGE_WEIGHT * features.get("max_mul_support_coverage", 0.0)
        + ADD_COVERAGE_WEIGHT * features.get("max_add_support_coverage", 0.0)
        - OUTSIDE_FRACTION_WEIGHT * features.get("min_mul_outside_fraction", 1.0)
    )


def _terminal_bonus(instance: ProblemInstance, state: CircuitState) -> float:
    return TERMINAL_BONUS if state.contains(instance.target) else 0.0


def _select_frontier(
    scored_states: list[tuple[CircuitState, float]],
    width: int,
) -> list[tuple[CircuitState, float]]:
    best_by_signature: dict[tuple[Hashable, ...], tuple[CircuitState, float]] = {}
    for state, score in scored_states:
        signature = tuple(sorted(state.node_keys, key=repr))
        existing = best_by_signature.get(signature)
        if existing is None or _frontier_sort_key(state, score) < _frontier_sort_key(*existing):
            best_by_signature[signature] = (state, score)
    return sorted(best_by_signature.values(), key=lambda item: _frontier_sort_key(*item))[:width]


def _ranker_device(ranker: CandidateRanker) -> torch.device:
    try:
        return next(ranker.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _candidate_sort_key(candidate: Candidate) -> tuple[float, tuple[str, int, int], Hashable]:
    return (-candidate.total_score, _candidate_action_key(candidate), candidate.result_poly.key())


def _candidate_action_key(candidate: Candidate) -> tuple[str, int, int]:
    return (candidate.action.op, candidate.action.i, candidate.action.j)


def _branch_sort_key(branch: _EvaluatedBranch) -> tuple[float, tuple[str, int, int], Hashable]:
    return (
        -branch.score,
        _candidate_action_key(branch.candidate),
        branch.candidate.result_poly.key(),
    )


def _frontier_sort_key(
    state: CircuitState,
    score: float,
) -> tuple[float, int, tuple[tuple[str, int, int], ...]]:
    action_key = tuple((action.op, action.i, action.j) for action in state.actions)
    return (-score, state.num_ops(), action_key)


def _validate_inputs(
    *,
    instance: ProblemInstance,
    ranker: CandidateRanker | None,
    encoder: CandidateFeatureEncoder | None,
    lambda_model: float,
    candidate_k: int,
    tier2_m: int,
    max_depth: int | None,
    num_rounds: int,
    initial_width: int,
    keep_per_round: int | None,
    rollout_depth: int,
    gumbel_scale: float,
    seed: int,
    expansion_budget: int | None,
) -> None:
    if not isinstance(instance, ProblemInstance):
        raise TypeError("instance must be a ProblemInstance")
    if isinstance(lambda_model, bool) or not isinstance(lambda_model, (int, float)):
        raise ValueError("lambda_model must be numeric")
    if float(lambda_model) < 0.0:
        raise ValueError("lambda_model must be non-negative")
    if float(lambda_model) > 0.0 and ranker is None:
        raise ValueError("ranker is required when lambda_model > 0")
    if float(lambda_model) > 0.0 and encoder is None:
        raise ValueError("encoder is required when lambda_model > 0")
    if type(candidate_k) is not int or candidate_k < 0:
        raise ValueError("candidate_k must be a non-negative int")
    if type(tier2_m) is not int or tier2_m < 0:
        raise ValueError("tier2_m must be a non-negative int")
    if max_depth is not None and (type(max_depth) is not int or max_depth < 0):
        raise ValueError("max_depth must be None or a non-negative int")
    if type(num_rounds) is not int or num_rounds <= 0:
        raise ValueError("num_rounds must be a positive int")
    if type(initial_width) is not int or initial_width <= 0:
        raise ValueError("initial_width must be a positive int")
    if keep_per_round is not None and (
        type(keep_per_round) is not int or keep_per_round <= 0
    ):
        raise ValueError("keep_per_round must be None or a positive int")
    if type(rollout_depth) is not int or rollout_depth < 0:
        raise ValueError("rollout_depth must be a non-negative int")
    if isinstance(gumbel_scale, bool) or not isinstance(gumbel_scale, (int, float)):
        raise ValueError("gumbel_scale must be numeric")
    if float(gumbel_scale) < 0.0:
        raise ValueError("gumbel_scale must be non-negative")
    if type(seed) is not int:
        raise ValueError("seed must be an int")
    if expansion_budget is not None and (
        type(expansion_budget) is not int or expansion_budget < 0
    ):
        raise ValueError("expansion_budget must be None or a non-negative int")
    initial = CircuitState.initial(instance)
    require_same_domain(instance.target, initial.nodes[0])
