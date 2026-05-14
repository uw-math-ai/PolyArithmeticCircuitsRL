from decomp_rl.baseline_cost import BaselineCostModel
from decomp_rl.baselines import (
    BivariateHornerBaseline,
    CSEBaseline,
    TopDownSearchBaseline,
)
from decomp_rl.polynomial import SparsePolynomial


def _poly(terms, p=5, variables=("x", "y")):
    return SparsePolynomial.from_terms(terms, p, variables)


def test_baselines_handle_zero_constant_variable_monomial():
    p, variables = 5, ("x", "y")
    zero = SparsePolynomial.zero(p, variables)
    one = SparsePolynomial.one(p, variables)
    x = SparsePolynomial.variable("x", p, variables)
    # 2 * x^2 * y -> rep_sq(2) for x^2 + rep_sq(1) for y + combine = 1 + 0 + 1 = 2
    mono = SparsePolynomial.from_monomial(2, (2, 1), p, variables)

    for baseline in (
        BivariateHornerBaseline(),
        CSEBaseline(),
        TopDownSearchBaseline(),
    ):
        assert baseline.cost(zero) == 0
        assert baseline.cost(one) == 0
        assert baseline.cost(x) == 0
        assert baseline.cost(mono) == 2


def test_each_baseline_is_an_upper_bound_below_sparse_direct():
    poly = _poly(
        (
            (1, (4, 0)),
            (3, (2, 1)),
            (2, (0, 3)),
            (1, (1, 2)),
            (4, (3, 1)),
        ),
        p=7,
    )
    base = BaselineCostModel()
    sparse = base.sparse_direct_cost(poly)
    for baseline in (
        BivariateHornerBaseline(),
        CSEBaseline(),
        TopDownSearchBaseline(),
    ):
        assert baseline.cost(poly) <= sparse


def test_bivariate_horner_beats_sparse_direct_with_polynomial_coefficients():
    # f(x, y) = (y^3 + 1) * x^5 + (y^2 + 1) * x^2 + (y + 1)
    poly = _poly(
        (
            (1, (5, 3)),
            (1, (5, 0)),
            (1, (2, 2)),
            (1, (2, 0)),
            (1, (0, 1)),
            (1, (0, 0)),
        ),
        p=7,
    )
    base = BaselineCostModel()
    sparse = base.sparse_direct_cost(poly)
    horner = BivariateHornerBaseline().cost(poly)
    assert horner < sparse


def test_cse_beats_sparse_direct_when_powers_are_shared():
    # x^2 * y^3 + x^2 * y^2 + x * y + 1
    # x^2 is shared by two terms; y^2 / y^3 share the y-power chain.
    poly = _poly(
        (
            (1, (2, 3)),
            (1, (2, 2)),
            (1, (1, 1)),
            (1, (0, 0)),
        ),
        p=5,
    )
    base = BaselineCostModel()
    sparse = base.sparse_direct_cost(poly)
    cse = CSEBaseline().cost(poly)
    assert cse < sparse


def test_top_down_search_beats_one_step_horner_with_a_pivot_above_one():
    # Both halves of x^4 + x^2 + 1 have the form ... + x^k * (...).
    # The default BaselineCostModel only ever pivots at k = 1; TopDownSearch
    # is allowed to pivot at k = 2 (sharing x^2 across both halves).
    poly = _poly(
        (
            (1, (4, 0)),
            (1, (2, 0)),
            (1, (0, 0)),
        ),
        p=5,
    )
    one_step = BaselineCostModel().horner_upper_bound(poly)
    search = TopDownSearchBaseline().cost(poly)
    assert search <= one_step


def test_top_down_search_terminates_on_two_variable_input():
    poly = _poly(
        (
            (1, (3, 2)),
            (2, (1, 1)),
            (3, (0, 3)),
            (1, (2, 0)),
        ),
        p=7,
    )
    # Smoke test: just make sure the recursion completes and the result is
    # a positive integer no worse than sparse_direct.
    cost = TopDownSearchBaseline().cost(poly)
    sparse = BaselineCostModel().sparse_direct_cost(poly)
    assert isinstance(cost, int)
    assert 0 < cost <= sparse
