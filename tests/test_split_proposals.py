from decomp_rl.baseline_cost import BaselineCostModel
from decomp_rl.family_generators import elementary_symmetric_polynomial
from decomp_rl.polynomial import SparsePolynomial
from decomp_rl.split_proposals import propose_splits


def test_proposals_include_valid_horner_split():
    variables = ("x", "y")
    p = 3
    x = SparsePolynomial.variable("x", p, variables)
    y = SparsePolynomial.variable("y", p, variables)
    target = x * y + y

    candidates = propose_splits(target, 8, baseline_model=BaselineCostModel())

    assert candidates
    assert any(candidate.source == "horner" for candidate in candidates)
    assert all(candidate.g + candidate.h == target for candidate in candidates)


def test_proposals_include_common_factor_templates():
    variables = ("x", "y")
    p = 3
    target = SparsePolynomial.from_terms(
        (
            (1, (2, 1)),
            (1, (1, 1)),
            (1, (0, 0)),
        ),
        p,
        variables,
    )

    candidates = propose_splits(target, 16, baseline_model=BaselineCostModel())

    assert any(candidate.source == "common_factor" for candidate in candidates)


def test_proposals_include_family_template_for_elementary_symmetric():
    variables = ("x1", "x2", "x3", "x4")
    target = elementary_symmetric_polynomial(variables, degree=2, prime=3)

    candidates = propose_splits(target, 16, baseline_model=BaselineCostModel())

    assert any(candidate.source == "family_template" for candidate in candidates)
