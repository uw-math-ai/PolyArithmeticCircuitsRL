import pytest

from lgs.poly.fast_poly import (
    FastPoly,
    Polynomial,
    PolynomialDegreeError,
    PolynomialDomainError,
    PolynomialOverflowError,
    assert_i64_accumulation_safe,
)
from lgs.poly.poly_utils import (
    exact_divides,
    make_variables,
    per_variable_degrees,
    poly_from_terms,
    support,
    total_degree,
)


def test_construct_variables_constant_and_basic_arithmetic():
    x, y, z = make_variables(n_vars=3, degree_cap=3, p=7)
    one = FastPoly.one(3, 3, 7)

    assert x.terms == {(1, 0, 0): 1}
    assert y.terms == {(0, 1, 0): 1}
    assert z.terms == {(0, 0, 1): 1}
    assert one.terms == {(0, 0, 0): 1}
    assert (x + y).terms == {(1, 0, 0): 1, (0, 1, 0): 1}
    assert (x * y).terms == {(1, 1, 0): 1}


def test_square_and_canonical_key():
    x = FastPoly.variable(0, 2, 3, 7)
    y = FastPoly.variable(1, 2, 3, 7)

    poly = (x + y) * (x + y)

    assert poly.terms == {
        (2, 0): 1,
        (1, 1): 2,
        (0, 2): 1,
    }
    assert poly.degree() == 2
    assert poly.support() == {(2, 0), (1, 1), (0, 2)}
    assert poly.copy() == poly
    assert poly == Polynomial(poly.terms, field_p=7, num_vars=2, degree_cap=3)
    assert poly.key() == Polynomial(
        poly.terms,
        field_p=7,
        num_vars=2,
        degree_cap=3,
    ).key()


def test_coefficients_are_reduced_and_zero_coefficients_removed():
    x = FastPoly.variable(0, 1, 4, 5)
    poly = FastPoly.constant(-1, 1, 4, 5) * (x**4)
    zero = x + FastPoly.monomial((1,), -1, 1, 4, 5)
    duplicate_cancel = poly_from_terms(
        [(3, (1,)), (2, (1,)), (10, (0,))],
        n_vars=1,
        degree_cap=4,
        p=5,
    )

    assert poly.terms == {(4,): 4}
    assert zero.is_zero()
    assert duplicate_cancel.is_zero()
    assert total_degree(poly) == 4
    assert per_variable_degrees(poly) == (4,)
    assert support(poly) == frozenset({(4,)})


def test_domain_mismatch_raises_loudly():
    x_mod_5 = FastPoly.variable(0, 1, 2, 5)
    x_mod_7 = FastPoly.variable(0, 1, 2, 7)
    x_two_vars = FastPoly.variable(0, 2, 2, 5)
    x_cap_1 = FastPoly.variable(0, 1, 1, 5)
    x_cap_2 = FastPoly.variable(0, 1, 2, 5)

    with pytest.raises(PolynomialDomainError):
        _ = x_mod_5 + x_mod_7
    with pytest.raises(PolynomialDomainError):
        _ = x_mod_5 * x_two_vars
    with pytest.raises(PolynomialDomainError):
        _ = x_cap_1 - x_cap_2
    with pytest.raises(PolynomialDomainError):
        _ = x_cap_1 * x_cap_2
    with pytest.raises(PolynomialDomainError):
        _ = x_mod_5 == x_mod_7


def test_degree_cap_is_in_key_and_multiplication_raises_on_overflow():
    x = FastPoly.variable(0, 1, 1, 7)
    x_uncapped = Polynomial.variable(0, field_p=7, num_vars=1)

    assert x.key() == FastPoly.variable(0, 1, 1, 7).key()
    assert x.key() != x_uncapped.key()
    assert x.key()[2] == 1

    with pytest.raises(PolynomialDegreeError):
        _ = x * x
    with pytest.raises(PolynomialDegreeError):
        FastPoly.monomial((2,), 0, 1, 1, 7)


def test_invalid_polynomial_inputs_raise():
    with pytest.raises(PolynomialDomainError):
        Polynomial.variable(0, field_p=8, num_vars=1)
    with pytest.raises(PolynomialDomainError):
        Polynomial({(1, 0): 1}, field_p=5, num_vars=1)
    with pytest.raises(PolynomialDomainError):
        Polynomial({(-1,): 1}, field_p=5, num_vars=1)
    with pytest.raises(PolynomialDomainError):
        Polynomial({(1,): 1.5}, field_p=5, num_vars=1)


def test_exact_division_returns_quotient_or_none():
    x = Polynomial.variable(0, field_p=11, num_vars=2)
    y = Polynomial.variable(1, field_p=11, num_vars=2)
    factor = x + y
    quotient = x + Polynomial.constant(3, field_p=11, num_vars=2)
    numerator = factor * quotient

    assert exact_divides(numerator, factor) == quotient
    assert exact_divides(numerator + Polynomial.one(field_p=11, num_vars=2), factor) is None

    with pytest.raises(ZeroDivisionError):
        exact_divides(numerator, Polynomial.zero(field_p=11, num_vars=2))


def test_exact_division_requested_cases():
    a = Polynomial.variable(0, field_p=13, num_vars=3)
    b = Polynomial.variable(1, field_p=13, num_vars=3)
    c = Polynomial.variable(2, field_p=13, num_vars=3)

    numerator = a * b + a * c
    assert exact_divides(numerator, a) == b + c
    assert exact_divides((a + b) ** 2, a + b) == a + b
    assert exact_divides(numerator, b) is None


def test_fixed_width_overflow_helper_raises_at_project_bound():
    assert_i64_accumulation_safe(101, 10)

    with pytest.raises(PolynomialOverflowError):
        assert_i64_accumulation_safe(2**31 - 1, 3)
