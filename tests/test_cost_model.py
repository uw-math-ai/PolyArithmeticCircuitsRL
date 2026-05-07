from decomp_rl.cost_model import monomial_build_cost, repeated_squaring_cost
from decomp_rl.polynomial import SparsePolynomial


def test_repeated_squaring_cost_behaves_reasonably():
    assert repeated_squaring_cost(0) == 0
    assert repeated_squaring_cost(1) == 0
    assert repeated_squaring_cost(2) == 1
    assert repeated_squaring_cost(3) == 2


def test_monomial_build_cost_counts_exponentiation_and_multiplies():
    poly = SparsePolynomial.from_monomial(1, (2, 1), 3, ("x", "y"))
    assert monomial_build_cost(poly) == 2

