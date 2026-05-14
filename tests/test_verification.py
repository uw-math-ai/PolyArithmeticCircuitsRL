import pytest

from lgs.env.action import Action
from lgs.env.candidate import Candidate
from lgs.env.circuit_state import CircuitState, InvalidActionError
from lgs.env.problem_instance import ProblemInstance
from lgs.env.verification import (
    VerificationError,
    execute_trace,
    require_valid_candidate_transition,
    require_verified_trace,
    replay_trace,
    verify_candidate_transition,
    verify_state,
    verify_trace,
)
from lgs.poly.fast_poly import FastPoly, Polynomial, PolynomialDomainError


def make_square_instance():
    x = FastPoly.variable(0, 2, 2, 11)
    y = FastPoly.variable(1, 2, 2, 11)
    return ProblemInstance(
        target=(x + y) * (x + y),
        variables=("x", "y"),
        field_p=11,
        degree_cap=2,
        op_budget=2,
        family_name="test",
    )


def make_ab_ac_instance():
    a = FastPoly.variable(0, 3, 2, 11)
    b = FastPoly.variable(1, 3, 2, 11)
    c = FastPoly.variable(2, 3, 2, 11)
    return ProblemInstance(
        target=a * b + a * c,
        variables=("a", "b", "c"),
        field_p=11,
        degree_cap=2,
        op_budget=2,
        family_name="test",
    )


def test_verify_state_and_trace_for_square_success():
    instance = make_square_instance()
    trace = (Action("add", 0, 1), Action("mul", 3, 3))

    state = execute_trace(instance, trace)

    assert verify_state(state, instance.target)
    assert verify_trace(instance, trace)
    assert require_verified_trace(instance, trace) == state
    assert replay_trace(trace, instance) == state


def test_verify_trace_for_ab_plus_ac_success():
    instance = make_ab_ac_instance()
    trace = (Action("add", 1, 2), Action("mul", 0, 4))

    assert verify_trace(instance, trace)


def test_verify_trace_failure_raises_only_in_require_helper():
    instance = make_square_instance()
    failing_trace = (Action("mul", 0, 1),)

    assert not verify_trace(instance, failing_trace)
    with pytest.raises(VerificationError):
        require_verified_trace(instance, failing_trace)


def test_verification_rejects_domain_mismatch():
    instance = make_square_instance()
    state = CircuitState.initial(instance)
    other_field_target = Polynomial.variable(0, field_p=13, num_vars=2)

    with pytest.raises(PolynomialDomainError):
        verify_state(state, other_field_target)


def test_candidate_transition_is_replayed_exactly():
    instance = make_square_instance()
    state = CircuitState.initial(instance)
    x_plus_y = state.get_node(0) + state.get_node(1)

    candidate = Candidate(
        action=Action("add", 0, 1),
        result_poly=x_plus_y,
        source_tags={"basic_pair"},
        features={"support_size": 2},
    )

    assert verify_candidate_transition(state, candidate)
    next_state = require_valid_candidate_transition(state, candidate)
    assert next_state.contains(x_plus_y)


def test_candidate_transition_detects_incorrect_result():
    instance = make_square_instance()
    state = CircuitState.initial(instance)
    x = Polynomial.variable(0, field_p=11, num_vars=2, degree_cap=2)

    candidate = Candidate(
        action=Action("add", 0, 1),
        result_poly=x,
        source_tags={"bad"},
        features={},
    )

    assert not verify_candidate_transition(state, candidate)
    with pytest.raises(VerificationError):
        require_valid_candidate_transition(state, candidate)


def test_trace_rejects_non_action_entries():
    instance = make_square_instance()

    with pytest.raises(VerificationError):
        execute_trace(instance, [(0, 1, "add")])


def test_invalid_trace_raises():
    instance = make_square_instance()

    with pytest.raises(InvalidActionError):
        verify_trace(instance, [Action("add", 0, 99)])
