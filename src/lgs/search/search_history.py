"""Search history records for heuristic-only planning."""

from __future__ import annotations

from dataclasses import dataclass, field

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


def _finished_sort_key(state: CircuitState) -> tuple[int, tuple[tuple[str, int, int], ...]]:
    action_key = tuple((action.op, action.i, action.j) for action in state.actions)
    return (state.num_ops(), action_key)
