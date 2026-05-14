import pytest

from lgs.env.action import Action
from lgs.env.circuit_state import CircuitState
from lgs.env.problem_instance import ProblemInstance
from lgs.poly.fast_poly import FastPoly
from lgs.search.candidate_generator import generate_candidates


def make_square_instance(degree_cap: int = 2) -> ProblemInstance:
    x = FastPoly.variable(0, 2, degree_cap, 17)
    y = FastPoly.variable(1, 2, degree_cap, 17)
    return ProblemInstance(
        target=(x + y) * (x + y),
        variables=("x", "y"),
        field_p=17,
        degree_cap=degree_cap,
        op_budget=3,
        family_name="test_square",
    )


def make_common_factor_instance(degree_cap: int = 2) -> ProblemInstance:
    a = FastPoly.variable(0, 3, degree_cap, 17)
    b = FastPoly.variable(1, 3, degree_cap, 17)
    c = FastPoly.variable(2, 3, degree_cap, 17)
    return ProblemInstance(
        target=a * b + a * c,
        variables=("a", "b", "c"),
        field_p=17,
        degree_cap=degree_cap,
        op_budget=3,
        family_name="test_common_factor",
    )


def find_by_action(candidates, op, i, j):
    action = Action.make(op, i, j)
    return next((candidate for candidate in candidates if candidate.action == action), None)


def assert_unique_result_keys(candidates) -> None:
    keys = [candidate.result_poly.key() for candidate in candidates]
    assert len(keys) == len(set(keys))


def test_square_target_initial_state_keeps_essential_intermediates():
    instance = make_square_instance()
    state = CircuitState.initial(instance)
    candidates = generate_candidates(instance, state, K=64)

    assert find_by_action(candidates, "add", 0, 1) is not None
    assert find_by_action(candidates, "mul", 0, 0) is not None
    assert find_by_action(candidates, "mul", 0, 1) is not None
    assert find_by_action(candidates, "mul", 1, 1) is not None
    assert_unique_result_keys(candidates)


def test_square_target_after_sum_ranks_exact_completion_first():
    instance = make_square_instance()
    state = CircuitState.initial(instance).apply(Action.make("add", 0, 1))
    candidates = generate_candidates(instance, state, K=64)

    candidate = find_by_action(candidates, "mul", 3, 3)

    assert candidate is not None
    assert candidate.result_poly == instance.target
    assert candidate.features["equals_target"] == 1.0
    assert "exact_target" in candidate.source_tags
    assert candidates[0] == candidate


def test_common_factor_initial_state_keeps_sum_and_detects_divisibility():
    instance = make_common_factor_instance()
    state = CircuitState.initial(instance)
    candidates = generate_candidates(instance, state, K=64, tier2_m=128)

    sum_candidate = find_by_action(candidates, "add", 1, 2)
    ab_candidate = find_by_action(candidates, "mul", 0, 1)
    ac_candidate = find_by_action(candidates, "mul", 0, 2)

    assert sum_candidate is not None
    assert ab_candidate is not None
    assert ac_candidate is not None
    assert sum_candidate.features["divides_target"] == 1.0
    assert sum_candidate.features["quotient_exists"] == 1.0
    assert sum_candidate.features["quotient_degree"] == 1.0
    assert sum_candidate.features["quotient_support_size"] == 1.0
    assert "divides_target" in sum_candidate.source_tags
    assert "quotient_exists" in sum_candidate.source_tags
    assert_unique_result_keys(candidates)


def test_common_factor_after_sum_ranks_exact_completion_first():
    instance = make_common_factor_instance()
    state = CircuitState.initial(instance).apply(Action.make("add", 1, 2))
    candidates = generate_candidates(instance, state, K=64)

    candidate = find_by_action(candidates, "mul", 0, 4)

    assert candidate is not None
    assert candidate.result_poly == instance.target
    assert candidate.features["equals_target"] == 1.0
    assert "exact_target" in candidate.source_tags
    assert candidates[0] == candidate


def test_duplicate_result_polynomials_are_filtered_against_state_and_output():
    instance = make_square_instance()
    state = CircuitState.initial(instance).apply(Action.make("add", 0, 1))
    existing_sum = state.nodes[-1]
    candidates = generate_candidates(instance, state, K=64)

    assert all(candidate.result_poly != existing_sum for candidate in candidates)
    assert_unique_result_keys(candidates)


def test_degree_cap_overflow_candidates_are_skipped_without_crashing():
    instance = make_square_instance(degree_cap=2)
    state = CircuitState.initial(instance).apply(Action.make("mul", 0, 0))

    candidates = generate_candidates(instance, state, K=64)

    assert candidates
    assert all((3, 0) not in candidate.result_poly.support() for candidate in candidates)
    assert find_by_action(candidates, "add", 0, 1) is not None


def test_k_limit_is_respected():
    instance = make_square_instance()
    state = CircuitState.initial(instance)

    assert len(generate_candidates(instance, state, K=3)) <= 3


def test_no_candidates_when_budget_exhausted():
    instance = make_square_instance()
    state = CircuitState.initial(instance)
    state = state.apply(Action.make("add", 0, 1))
    state = state.apply(Action.make("mul", 0, 1))
    state = state.apply(Action.make("mul", 1, 1))

    assert generate_candidates(instance, state) == []
