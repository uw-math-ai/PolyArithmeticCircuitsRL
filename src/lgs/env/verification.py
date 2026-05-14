"""Exact verification helpers for states, traces, and candidate transitions."""

from __future__ import annotations

from collections.abc import Iterable

from lgs.env.action import Action
from lgs.env.candidate import Candidate
from lgs.env.circuit_state import CircuitState
from lgs.env.problem_instance import ProblemInstance
from lgs.poly.fast_poly import Polynomial, PolynomialDomainError
from lgs.poly.poly_utils import require_same_domain


class VerificationError(AssertionError):
    """Raised when an exact verification requirement fails."""


def verify_state(state: CircuitState, target: Polynomial) -> bool:
    if not isinstance(state, CircuitState):
        raise VerificationError("verify_state requires a CircuitState")
    if not isinstance(target, Polynomial):
        raise PolynomialDomainError("target must be a Polynomial")
    require_same_domain(state.nodes[0], target)
    return state.contains(target)


def require_verified_state(state: CircuitState, target: Polynomial) -> None:
    if not verify_state(state, target):
        raise VerificationError("state does not contain the exact target polynomial")


def execute_trace(instance: ProblemInstance, trace: Iterable[Action]) -> CircuitState:
    if not isinstance(instance, ProblemInstance):
        raise VerificationError("execute_trace requires a ProblemInstance")

    state = CircuitState.initial(instance)
    for step, action in enumerate(trace):
        if not isinstance(action, Action):
            raise VerificationError(f"trace step {step} is not an Action")
        state = state.apply(action)
    return state


def replay_trace(trace: Iterable[Action], instance: ProblemInstance) -> CircuitState:
    return execute_trace(instance, trace)


def verify_trace(instance: ProblemInstance, trace: Iterable[Action]) -> bool:
    state = execute_trace(instance, trace)
    return verify_state(state, instance.target)


def require_verified_trace(instance: ProblemInstance, trace: Iterable[Action]) -> CircuitState:
    state = execute_trace(instance, trace)
    if not verify_state(state, instance.target):
        raise VerificationError("trace does not construct the exact target polynomial")
    return state


def verify_candidate_transition(state: CircuitState, candidate: Candidate) -> bool:
    if not isinstance(state, CircuitState):
        raise VerificationError("verify_candidate_transition requires a CircuitState")
    if not isinstance(candidate, Candidate):
        raise VerificationError("candidate must be a Candidate")

    next_state = state.apply(candidate.action)
    produced = next_state.nodes[-1]
    require_same_domain(produced, candidate.result_poly)
    return produced == candidate.result_poly


def require_valid_candidate_transition(
    state: CircuitState,
    candidate: Candidate,
) -> CircuitState:
    if not verify_candidate_transition(state, candidate):
        raise VerificationError("candidate result does not match exact action replay")
    return state.apply(candidate.action)
