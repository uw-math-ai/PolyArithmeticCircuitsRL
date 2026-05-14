"""Preference extraction from exact beam-search histories."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from lgs.env.action import Action
from lgs.env.candidate import Candidate
from lgs.env.circuit_state import CircuitState
from lgs.env.problem_instance import ProblemInstance
from lgs.search.search_history import ExpandedStateRecord, SearchHistory


@dataclass
class PreferenceExample:
    instance: ProblemInstance
    state: CircuitState
    better: Candidate
    worse: Candidate
    weight: float
    return_better: float
    return_worse: float
    metadata: dict[str, Any] = field(default_factory=dict)


def extract_preferences(
    history: SearchHistory,
    delta: float = 1.0,
) -> list[PreferenceExample]:
    if not isinstance(history, SearchHistory):
        raise TypeError("history must be a SearchHistory")
    if isinstance(delta, bool) or not isinstance(delta, (int, float)):
        raise ValueError("delta must be numeric")

    best = history.best_finished()
    if best is None:
        return []

    instance = history.instance
    best_trace = tuple(best.actions)
    return_good = 100.0 - best.num_ops()
    preferences: list[PreferenceExample] = []

    for step, good_action in enumerate(best_trace):
        prefix = best_trace[:step]
        record = _find_record_for_trace_step(history, prefix, good_action)
        better = _find_candidate(record.candidates, good_action)

        for other in record.candidates:
            if other.action == good_action:
                continue
            return_alt = _alternative_return(
                history=history,
                source_state=record.state,
                prefix=prefix,
                candidate=other,
            )
            if return_good <= return_alt + float(delta):
                continue

            preferences.append(
                PreferenceExample(
                    instance=instance,
                    state=record.state,
                    better=better,
                    worse=other,
                    weight=_preference_weight(return_good, return_alt),
                    return_better=return_good,
                    return_worse=return_alt,
                    metadata={
                        "step": step,
                        "prefix_length": step,
                        "best_num_ops": best.num_ops(),
                        "delta": float(delta),
                        "return_gap": return_good - return_alt,
                        "source_num_ops": record.state.num_ops(),
                        "better_action": _action_string(better.action),
                        "worse_action": _action_string(other.action),
                        "better_tags": tuple(sorted(better.source_tags)),
                        "worse_tags": tuple(sorted(other.source_tags)),
                    },
                )
            )

    return preferences


def serialize_preference(pref: PreferenceExample) -> tuple[
    tuple[Action, ...],
    Action,
    Action,
    float,
    float,
    float,
]:
    return (
        tuple(pref.state.actions),
        pref.better.action,
        pref.worse.action,
        pref.return_better,
        pref.return_worse,
        pref.weight,
    )


def _find_record_for_trace_step(
    history: SearchHistory,
    prefix: tuple[Action, ...],
    action: Action,
) -> ExpandedStateRecord:
    matches = [
        record
        for record in history.records
        if tuple(record.state.actions) == prefix
        and record.candidate.action == action
    ]
    if not matches:
        raise ValueError(
            "search history is inconsistent: no record for trace step "
            f"prefix={tuple(_action_string(a) for a in prefix)} "
            f"action={_action_string(action)}"
        )
    return sorted(matches, key=_record_sort_key)[0]


def _find_candidate(candidates: list[Candidate], action: Action) -> Candidate:
    matches = [
        candidate
        for candidate in candidates
        if candidate.action == action
    ]
    if not matches:
        raise ValueError(
            "search history is inconsistent: good action is absent from "
            f"candidate list: {_action_string(action)}"
        )
    return sorted(matches, key=_candidate_sort_key)[0]


def _alternative_return(
    *,
    history: SearchHistory,
    source_state: CircuitState,
    prefix: tuple[Action, ...],
    candidate: Candidate,
) -> float:
    alt_next_state = _candidate_next_state(history, source_state, candidate)
    if alt_next_state.contains(history.instance.target):
        return 100.0 - alt_next_state.num_ops()

    finished = _best_finished_with_prefix(history, (*prefix, candidate.action))
    if finished is not None:
        return 100.0 - finished.num_ops()
    return -20.0


def _candidate_next_state(
    history: SearchHistory,
    state: CircuitState,
    candidate: Candidate,
) -> CircuitState:
    matches = [
        record
        for record in history.records
        if tuple(record.state.actions) == tuple(state.actions)
        and record.candidate.action == candidate.action
    ]
    if matches:
        return sorted(matches, key=_record_sort_key)[0].next_state
    return state.apply(candidate.action)


def _best_finished_with_prefix(
    history: SearchHistory,
    prefix_actions: tuple[Action, ...],
) -> CircuitState | None:
    matches = [
        state
        for state in history.finished
        if tuple(state.actions[: len(prefix_actions)]) == prefix_actions
        and state.contains(history.instance.target)
    ]
    if not matches:
        return None
    return sorted(matches, key=_state_sort_key)[0]


def _preference_weight(return_good: float, return_alt: float) -> float:
    if return_alt < 0:
        return 1.0
    if return_good - return_alt >= 5.0:
        return 1.0
    return 0.25


def _action_string(action: Action) -> str:
    return f"{action.op}({action.i},{action.j})"


def _action_tuple(actions: tuple[Action, ...]) -> tuple[tuple[str, int, int], ...]:
    return tuple((action.op, action.i, action.j) for action in actions)


def _state_sort_key(state: CircuitState) -> tuple[int, tuple[tuple[str, int, int], ...]]:
    return (state.num_ops(), _action_tuple(tuple(state.actions)))


def _candidate_sort_key(candidate: Candidate) -> tuple[tuple[str, int, int], object]:
    return ((candidate.action.op, candidate.action.i, candidate.action.j), candidate.result_poly.key())


def _record_sort_key(record: ExpandedStateRecord) -> tuple[
    int,
    tuple[tuple[str, int, int], ...],
    tuple[str, int, int],
]:
    action = record.candidate.action
    return (
        record.depth,
        _action_tuple(tuple(record.state.actions)),
        (action.op, action.i, action.j),
    )
