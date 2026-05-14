"""Heuristic-only beam search over exact circuit states."""

from __future__ import annotations

from typing import Any, Hashable

import torch

from lgs.env.action import Action
from lgs.env.candidate import Candidate
from lgs.env.circuit_state import CircuitState
from lgs.env.problem_instance import ProblemInstance
from lgs.poly.fast_poly import PolynomialDegreeError
from lgs.poly.poly_utils import require_same_domain
from lgs.search.candidate_generator import generate_candidates
from lgs.search.search_history import ExpandedStateRecord, SearchHistory

STATE_COST_ALPHA = 0.01


def beam_search(
    instance: ProblemInstance,
    *,
    ranker: Any | None = None,
    encoder: Any | None = None,
    lambda_model: float = 0.0,
    beam_width: int = 16,
    candidate_k: int = 64,
    tier2_m: int = 128,
    max_depth: int | None = None,
    exploration_eps: float = 0.0,
    stop_on_first_success: bool = False,
) -> SearchHistory:
    _validate_inputs(
        instance=instance,
        beam_width=beam_width,
        candidate_k=candidate_k,
        tier2_m=tier2_m,
        max_depth=max_depth,
        exploration_eps=exploration_eps,
        ranker=ranker,
        encoder=encoder,
        lambda_model=lambda_model,
    )
    depth_limit = instance.op_budget if max_depth is None else max_depth
    history = SearchHistory(instance=instance)
    initial_state = CircuitState.initial(instance)
    if initial_state.contains(instance.target):
        history.finished.append(initial_state)
        if stop_on_first_success:
            return history

    beam: list[tuple[CircuitState, float]] = [(initial_state, 0.0)]
    for depth in range(depth_limit):
        scored_next_states: list[tuple[CircuitState, float]] = []
        for state, _ in beam:
            if state.remaining_budget() <= 0:
                continue
            candidates = generate_candidates(
                instance,
                state,
                K=candidate_k,
                tier2_m=tier2_m,
            )
            _score_candidates_with_model(
                instance=instance,
                state=state,
                candidates=candidates,
                ranker=ranker,
                encoder=encoder,
                lambda_model=float(lambda_model),
            )
            candidates = sorted(candidates, key=_candidate_sort_key)
            for candidate in candidates:
                try:
                    next_state = state.apply(candidate.action)
                except PolynomialDegreeError:
                    continue
                state_score = _score_next_state(candidate.total_score, next_state)
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
                    if stop_on_first_success:
                        return history
                scored_next_states.append((next_state, state_score))

        if not scored_next_states:
            break
        beam = _select_beam(scored_next_states, beam_width)

    return history


def recover_trace(state: CircuitState) -> list[Action]:
    return list(state.actions)


def _score_next_state(candidate_score: float, next_state: CircuitState) -> float:
    return candidate_score - STATE_COST_ALPHA * next_state.num_ops()


def _score_candidates_with_model(
    *,
    instance: ProblemInstance,
    state: CircuitState,
    candidates: list[Candidate],
    ranker: Any | None,
    encoder: Any | None,
    lambda_model: float,
) -> None:
    if lambda_model == 0.0 or ranker is None:
        for candidate in candidates:
            candidate.model_score = 0.0
            candidate.total_score = candidate.heuristic_score
        return

    was_training = bool(getattr(ranker, "training", False))
    ranker.eval()
    try:
        device = next(ranker.parameters()).device
        with torch.no_grad():
            for candidate in candidates:
                features = encoder.encode(instance, state, candidate)
                feature_tensor = torch.tensor([features], dtype=torch.float32, device=device)
                candidate.model_score = float(ranker(feature_tensor).item())
                candidate.total_score = (
                    candidate.heuristic_score
                    + lambda_model * candidate.model_score
                )
    finally:
        if was_training:
            ranker.train()


def _candidate_sort_key(candidate: Candidate) -> tuple[float, tuple[str, int, int], Hashable]:
    action_key = (candidate.action.op, candidate.action.i, candidate.action.j)
    return (-candidate.total_score, action_key, candidate.result_poly.key())


def _select_beam(
    scored_states: list[tuple[CircuitState, float]],
    beam_width: int,
) -> list[tuple[CircuitState, float]]:
    best_by_signature: dict[tuple[Hashable, ...], tuple[CircuitState, float]] = {}
    for state, score in scored_states:
        signature = _state_signature(state)
        existing = best_by_signature.get(signature)
        if existing is None or _beam_sort_key(state, score) < _beam_sort_key(*existing):
            best_by_signature[signature] = (state, score)

    selected = sorted(
        best_by_signature.values(),
        key=lambda item: _beam_sort_key(item[0], item[1]),
    )
    return selected[:beam_width]


def _state_signature(state: CircuitState) -> tuple[Hashable, ...]:
    return tuple(sorted(state.node_keys, key=repr))


def _beam_sort_key(state: CircuitState, score: float) -> tuple[float, int, tuple[tuple[str, int, int], ...]]:
    action_key = tuple((action.op, action.i, action.j) for action in state.actions)
    return (-score, state.num_ops(), action_key)


def _validate_inputs(
    *,
    instance: ProblemInstance,
    beam_width: int,
    candidate_k: int,
    tier2_m: int,
    max_depth: int | None,
    exploration_eps: float,
    ranker: Any | None,
    encoder: Any | None,
    lambda_model: float,
) -> None:
    if not isinstance(instance, ProblemInstance):
        raise TypeError("instance must be a ProblemInstance")
    if type(beam_width) is not int or beam_width <= 0:
        raise ValueError("beam_width must be a positive int")
    if type(candidate_k) is not int or candidate_k < 0:
        raise ValueError("candidate_k must be a non-negative int")
    if type(tier2_m) is not int or tier2_m < 0:
        raise ValueError("tier2_m must be a non-negative int")
    if max_depth is not None and (type(max_depth) is not int or max_depth < 0):
        raise ValueError("max_depth must be None or a non-negative int")
    if isinstance(exploration_eps, bool) or not isinstance(exploration_eps, (int, float)):
        raise ValueError("exploration_eps must be numeric")
    if float(exploration_eps) != 0.0:
        raise NotImplementedError("exploration_eps is reserved for later milestones")
    if isinstance(lambda_model, bool) or not isinstance(lambda_model, (int, float)):
        raise ValueError("lambda_model must be numeric")
    if float(lambda_model) < 0.0:
        raise ValueError("lambda_model must be non-negative")
    if float(lambda_model) > 0.0 and ranker is None:
        raise ValueError("ranker is required when lambda_model > 0")
    if float(lambda_model) > 0.0 and encoder is None:
        raise ValueError("encoder is required when lambda_model > 0")
    initial = CircuitState.initial(instance)
    require_same_domain(instance.target, initial.nodes[0])
