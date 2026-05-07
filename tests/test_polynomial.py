from decomp_rl.polynomial import SparsePolynomial


def test_sparse_polynomial_add_mul_and_split():
    variables = ("x", "y")
    p = 3
    x = SparsePolynomial.variable("x", p, variables)
    y = SparsePolynomial.variable("y", p, variables)

    poly = x * y + y
    assert poly.support_size == 2
    assert poly.total_degree == 2

    remainder, quotient = poly.split_by_variable(0)
    assert remainder == y
    assert quotient == y

    reconstructed = remainder + x * quotient
    assert reconstructed == poly


def test_monic_normalization_round_trip():
    variables = ("x",)
    poly = SparsePolynomial.from_terms(((2, (2,)), (1, (0,))), 3, variables)
    monic, extracted = poly.make_monic()
    assert extracted == 2
    assert monic.scale(extracted) == poly

