"""Environment data structures and exact verification helpers."""

from lgs.env.action import Action
from lgs.env.candidate import Candidate
from lgs.env.circuit_state import (
    BudgetExceededError,
    CircuitState,
    InvalidActionError,
    InvalidCircuitStateError,
)
from lgs.env.problem_instance import ProblemInstance
from lgs.env.verification import (
    VerificationError,
    execute_trace,
    replay_trace,
    require_valid_candidate_transition,
    require_verified_state,
    require_verified_trace,
    verify_candidate_transition,
    verify_state,
    verify_trace,
)

__all__ = [
    "Action",
    "BudgetExceededError",
    "Candidate",
    "CircuitState",
    "InvalidActionError",
    "InvalidCircuitStateError",
    "ProblemInstance",
    "VerificationError",
    "execute_trace",
    "replay_trace",
    "require_valid_candidate_transition",
    "require_verified_state",
    "require_verified_trace",
    "verify_candidate_transition",
    "verify_state",
    "verify_trace",
]
