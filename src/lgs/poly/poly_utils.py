"""Exact utility functions for sparse finite-field polynomials."""

from __future__ import annotations

from typing import Hashable

from lgs.poly.fast_poly import (
    Exponent,
    Polynomial,
    PolynomialDegreeError,
    PolynomialDomainError,
    PolynomialError,
)


def canonical_key(poly: Polynomial) -> Hashable:
    _require_polynomial(poly)
    return poly.key()


def coefficients(poly: Polynomial) -> dict[Exponent, int]:
    _require_polynomial(poly)
    return poly.terms


def make_variables(n_vars: int, degree_cap: int, p: int) -> list[Polynomial]:
    return [
        Polynomial.variable(index, n_vars, degree_cap, p)
        for index in range(n_vars)
    ]


def poly_from_terms(
    terms: list[tuple[int, Exponent]] | tuple[tuple[int, Exponent], ...],
    n_vars: int,
    degree_cap: int,
    p: int,
) -> Polynomial:
    return Polynomial(
        ((exponent, coefficient) for coefficient, exponent in terms),
        field_p=p,
        num_vars=n_vars,
        degree_cap=degree_cap,
    )


def support(poly: Polynomial) -> frozenset[Exponent]:
    _require_polynomial(poly)
    return frozenset(exponent for exponent, _ in poly.items())


def support_size(poly: Polynomial) -> int:
    _require_polynomial(poly)
    return len(poly.items())


def total_degree(poly: Polynomial) -> int:
    """Return total degree, using ``-1`` for the zero polynomial."""

    _require_polynomial(poly)
    if poly.is_zero():
        return -1
    return max(sum(exponent) for exponent, _ in poly.items())


def per_variable_degrees(poly: Polynomial) -> tuple[int, ...]:
    """Return maximum exponent per variable, using zeros for the zero polynomial."""

    _require_polynomial(poly)
    degrees = [0] * poly.num_vars
    for exponent, _ in poly.items():
        for index, power in enumerate(exponent):
            degrees[index] = max(degrees[index], power)
    return tuple(degrees)


def max_per_variable_degree(poly: Polynomial) -> int:
    degrees = per_variable_degrees(poly)
    return max(degrees, default=0)


def assert_degree_cap(poly: Polynomial, degree_cap: int) -> None:
    _require_polynomial(poly)
    if type(degree_cap) is not int:
        raise PolynomialDegreeError("degree_cap must be an int")
    if degree_cap < 0:
        raise PolynomialDegreeError("degree_cap must be non-negative")

    degrees = per_variable_degrees(poly)
    if any(power > degree_cap for power in degrees):
        raise PolynomialDegreeError(
            f"polynomial per-variable degrees {degrees} exceed cap {degree_cap}"
        )


def leading_term(poly: Polynomial) -> tuple[Exponent, int]:
    """Return the leading term under graded lexicographic order."""

    _require_polynomial(poly)
    if poly.is_zero():
        raise ZeroDivisionError("zero polynomial has no leading term")
    return max(poly.items(), key=lambda item: (sum(item[0]), item[0]))


def exact_divides(numerator: Polynomial, denominator: Polynomial) -> Polynomial | None:
    """Return the exact quotient if ``denominator`` divides ``numerator``.

    Returns ``None`` when division is not exact. Division by zero raises.
    """

    require_same_domain(numerator, denominator)
    if denominator.is_zero():
        raise ZeroDivisionError("division by the zero polynomial")
    if numerator.is_zero():
        return Polynomial.zero(
            field_p=numerator.field_p,
            num_vars=numerator.num_vars,
            degree_cap=numerator.degree_cap,
        )

    quotient = Polynomial.zero(
        field_p=numerator.field_p,
        num_vars=numerator.num_vars,
        degree_cap=numerator.degree_cap,
    )
    remainder = numerator
    den_exp, den_coeff = leading_term(denominator)
    den_coeff_inv = pow(den_coeff, -1, numerator.field_p)

    while not remainder.is_zero():
        rem_exp, rem_coeff = leading_term(remainder)
        if not monomial_divides(den_exp, rem_exp):
            return None

        quotient_exp = tuple(
            rem_power - den_power
            for rem_power, den_power in zip(rem_exp, den_exp)
        )
        quotient_coeff = (rem_coeff * den_coeff_inv) % numerator.field_p
        quotient_term = Polynomial.monomial(
            quotient_coeff,
            quotient_exp,
            field_p=numerator.field_p,
            num_vars=numerator.num_vars,
            degree_cap=numerator.degree_cap,
        )
        quotient = quotient + quotient_term
        remainder = remainder - quotient_term * denominator

    if denominator * quotient != numerator:
        raise PolynomialError("internal exact division check failed")
    return quotient


def additive_residual(target: Polynomial, partial: Polynomial) -> Polynomial:
    require_same_domain(target, partial)
    return target - partial


def multiplicative_quotient(target: Polynomial, factor: Polynomial) -> Polynomial | None:
    return exact_divides(target, factor)


def monomial_divides(divisor: Exponent, dividend: Exponent) -> bool:
    if len(divisor) != len(dividend):
        raise PolynomialDomainError(
            f"monomial layout mismatch: {len(divisor)} vs {len(dividend)}"
        )
    return all(left <= right for left, right in zip(divisor, dividend))


def require_same_domain(*polys: Polynomial) -> None:
    if not polys:
        return
    first = polys[0]
    _require_polynomial(first)
    for poly in polys[1:]:
        _require_polynomial(poly)
        if poly.field_p != first.field_p:
            raise PolynomialDomainError(
                f"field mismatch: F_{first.field_p} vs F_{poly.field_p}"
            )
        if poly.num_vars != first.num_vars:
            raise PolynomialDomainError(
                f"variable-count mismatch: {first.num_vars} vs {poly.num_vars}"
            )
        if poly.degree_cap != first.degree_cap:
            raise PolynomialDomainError(
                f"degree-cap mismatch: {first.degree_cap} vs {poly.degree_cap}"
            )


def _require_polynomial(poly: Polynomial) -> None:
    if not isinstance(poly, Polynomial):
        raise PolynomialDomainError(f"expected Polynomial, got {type(poly).__name__}")
