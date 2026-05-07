"""Cost model helpers for direct construction and factor rebuilds."""

from __future__ import annotations

from .factor_fp import FactorizationResult
from .polynomial import SparsePolynomial


def repeated_squaring_cost(exponent: int) -> int:
    if exponent <= 1:
        return 0
    return (exponent.bit_length() - 1) + (bin(exponent).count("1") - 1)


def monomial_build_cost(poly: SparsePolynomial) -> int:
    if not poly.is_monomial:
        raise ValueError("monomial_build_cost expects a monomial")
    if poly.is_zero or poly.is_constant or poly.is_variable():
        return 0
    _, exponent = poly.terms[0]
    non_zero_powers = [value for value in exponent if value > 0]
    power_cost = sum(repeated_squaring_cost(value) for value in non_zero_powers)
    combine_cost = max(0, len(non_zero_powers) - 1)
    return power_cost + combine_cost


def rebuild_cost(factorization: FactorizationResult) -> int:
    if factorization.unit == 0:
        return 0
    active_terms = 0
    total = 0
    for factor, exponent in factorization.factors:
        if factor.is_constant:
            continue
        active_terms += 1
        total += repeated_squaring_cost(exponent)
    total += max(0, active_terms - 1)
    return total


def unresolved_children(
    factorization: FactorizationResult,
) -> tuple[SparsePolynomial, ...]:
    children = [
        factor
        for factor, _ in factorization.factors
        if not factor.is_constant
    ]
    dedup: dict[str, SparsePolynomial] = {}
    for child in children:
        dedup.setdefault(child.to_key(), child)
    return tuple(dedup.values())

