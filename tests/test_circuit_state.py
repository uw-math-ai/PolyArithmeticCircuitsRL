import pytest

from lgs.env.action import Action
from lgs.env.circuit_state import (
    BudgetExceededError,
    CircuitState,
    InvalidActionError,
    InvalidCircuitStateError,
)
from lgs.env.problem_instance import ProblemInstance
from lgs.poly.fast_poly import FastPoly, Polynomial, PolynomialDomainError
from lgs.poly.poly_utils import PolynomialDegreeError


def make_instance(target, op_budget=3, degree_cap=3):
    return ProblemInstance(
        target=target,
        variables=("x", "y"),
        field_p=target.field_p,
        degree_cap=degree_cap,
        op_budget=op_budget,
        family_name="test",
    )


def test_action_canonicalizes_commutative_indices():
    assert Action("add", 4, 2) == Action("add", 2, 4)
    assert Action("mul", 3, 3).i == 3

    with pytest.raises(ValueError):
        Action("sub", 0, 1)
    with pytest.raises(ValueError):
        Action("add", -1, 1)


def test_initial_state_and_apply_records_exact_metadata():
    x = FastPoly.variable(0, 2, 2, 7)
    y = FastPoly.variable(1, 2, 2, 7)
    target = (x + y) * (x + y)
    instance = make_instance(target, op_budget=2, degree_cap=2)

    state = CircuitState.initial(instance)
    assert state.num_nodes() == 3
    assert state.get_node(0) == x
    assert state.get_node(1) == y
    assert state.get_node(2) == FastPoly.one(2, 2, 7)
    assert state.num_ops() == 0
    assert state.remaining_budget() == 2

    next_state = state.apply(Action.make("add", 0, 1))
    final_state = next_state.apply(Action.make("mul", 3, 3))

    assert next_state.num_nodes() == 4
    assert next_state.num_ops() == 1
    assert next_state.get_node(3) == x + y
    assert next_state.parents[-1] == (0, 1, "add")
    assert next_state.actions[-1] == Action("add", 0, 1)
    assert final_state.get_node(4) == instance.target
    assert final_state.parents[-1] == (3, 3, "mul")
    assert final_state.contains(instance.target)
    assert state.num_nodes() == 3
    assert state.num_ops() == 0


def test_apply_rejects_invalid_action_and_budget_exhaustion():
    x = Polynomial.variable(0, field_p=7, num_vars=2)
    instance = make_instance(x, op_budget=1)
    state = CircuitState.initial(instance)

    with pytest.raises(InvalidActionError):
        state.apply(Action("add", 0, 99))

    used = state.apply(Action("add", 0, 2))
    with pytest.raises(BudgetExceededError):
        used.apply(Action("add", 0, 2))


def test_degree_cap_is_enforced_without_truncation():
    x = Polynomial.variable(0, field_p=7, num_vars=2)
    instance = make_instance(x, op_budget=2, degree_cap=1)
    state = CircuitState.initial(instance)

    with pytest.raises(PolynomialDegreeError):
        state.apply(Action("mul", 0, 0))


def test_problem_instance_rejects_domain_and_degree_mismatch():
    target = Polynomial.variable(0, field_p=7, num_vars=1)

    with pytest.raises(PolynomialDomainError):
        ProblemInstance(
            target=target,
            variables=("x",),
            field_p=5,
            degree_cap=1,
            op_budget=1,
        )

    with pytest.raises(ValueError):
        ProblemInstance(
            target=target,
            variables=("x", "x"),
            field_p=7,
            degree_cap=1,
            op_budget=1,
        )

    with pytest.raises(PolynomialDegreeError):
        ProblemInstance(
            target=target**2,
            variables=("x",),
            field_p=7,
            degree_cap=1,
            op_budget=1,
        )


def test_circuit_state_rejects_inconsistent_node_keys():
    x = Polynomial.variable(0, field_p=7, num_vars=1)
    one = Polynomial.one(field_p=7, num_vars=1)

    with pytest.raises(InvalidCircuitStateError):
        CircuitState(nodes=(one, x), node_keys=frozenset({one.key()}), op_budget=1)


def test_circuit_state_rejects_state_degree_cap_mismatch():
    x = Polynomial.variable(0, field_p=7, num_vars=1, degree_cap=1)
    one = Polynomial.one(field_p=7, num_vars=1, degree_cap=1)

    with pytest.raises(InvalidCircuitStateError):
        CircuitState(nodes=(one, x), op_budget=1, degree_cap=2)
