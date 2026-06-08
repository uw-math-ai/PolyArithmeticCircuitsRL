"""Search history records for heuristic-only planning."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from lgs.env.candidate import Candidate
from lgs.env.circuit_state import CircuitState
from lgs.env.problem_instance import ProblemInstance


@dataclass
class ExpandedStateRecord:
    instance: ProblemInstance
    state: CircuitState
    candidates: list[Candidate]
    candidate: Candidate
    next_state: CircuitState
    depth: int
    state_score: float


@dataclass
class SearchHistory:
    instance: ProblemInstance
    records: list[ExpandedStateRecord] = field(default_factory=list)
    finished: list[CircuitState] = field(default_factory=list)
    num_expansions: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def best_finished(self) -> CircuitState | None:
        verified = [
            state
            for state in self.finished
            if state.contains(self.instance.target)
        ]
        if not verified:
            return None
        return min(verified, key=_finished_sort_key)

    def success(self) -> bool:
        return self.best_finished() is not None


def history_expansion_count(history: SearchHistory) -> int:
    if not isinstance(history, SearchHistory):
        raise TypeError("history must be a SearchHistory")
    if history.num_expansions is not None:
        if type(history.num_expansions) is not int or history.num_expansions < 0:
            raise ValueError("history.num_expansions must be None or a non-negative int")
        return history.num_expansions
    return len(history.records)


def _finished_sort_key(state: CircuitState) -> tuple[int, tuple[tuple[str, int, int], ...]]:
    action_key = tuple((action.op, action.i, action.j) for action in state.actions)
    return (state.num_ops(), action_key)
