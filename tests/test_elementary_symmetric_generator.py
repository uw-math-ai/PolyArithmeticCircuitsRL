from decomp_rl.family_generators import elementary_symmetric_example


def test_elementary_symmetric_recurrence_example_is_valid():
    example = elementary_symmetric_example(variable_count=4, degree=2, prime=3)
    assert example.preferred_action.g + example.preferred_action.h == example.target
    assert example.target.support_size == 6
    assert any(
        candidate.key() == example.preferred_action.key()
        for candidate in example.candidates
    )

