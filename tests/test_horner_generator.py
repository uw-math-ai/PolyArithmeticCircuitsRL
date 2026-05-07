from random import Random

from decomp_rl.family_generators import (
    exact_small_example,
    horner_example,
    multivariate_horner_example,
)


def test_horner_generator_builds_valid_supervised_example():
    example = horner_example([1, 2, 0, 1], 3)
    assert example.target.support_size >= 1
    assert example.preferred_action.g + example.preferred_action.h == example.target
    assert any(
        candidate.key() == example.preferred_action.key()
        for candidate in example.candidates
    )


def test_exact_small_generator_builds_valid_example():
    example = exact_small_example(Random(0), 3, variables=("x", "y"))
    assert example.preferred_action.g + example.preferred_action.h == example.target
    assert example.value_target > 0


def test_multivariate_horner_generator_builds_valid_example():
    example = multivariate_horner_example(Random(0), 3, variables=("x1", "x2", "x3"))
    assert example.preferred_action.g + example.preferred_action.h == example.target
    assert example.value_target > 0
    assert example.target.variables == ("x1", "x2", "x3")
