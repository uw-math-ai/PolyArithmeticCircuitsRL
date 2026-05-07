"""Canonicalization helpers for sparse finite-field polynomials."""

from __future__ import annotations

from typing import Iterable

Term = tuple[int, tuple[int, ...]]


def normalize_coeff(coeff: int, prime: int) -> int:
    """Return a canonical coefficient representative in F_p."""
    return coeff % prime


def term_sort_key(term: Term) -> tuple[int, tuple[int, ...], int]:
    coeff, exponent = term
    return (-sum(exponent), exponent, coeff)


def canonicalize_terms(
    terms: Iterable[tuple[int, tuple[int, ...]]],
    prime: int,
    variable_count: int,
) -> tuple[Term, ...]:
    merged: dict[tuple[int, ...], int] = {}
    for coeff, exponent in terms:
        if len(exponent) != variable_count:
            raise ValueError(
                f"Expected exponent vectors of length {variable_count}, got {len(exponent)}"
            )
        coeff = normalize_coeff(coeff, prime)
        if coeff == 0:
            continue
        merged[tuple(int(v) for v in exponent)] = (
            merged.get(tuple(int(v) for v in exponent), 0) + coeff
        ) % prime

    canonical = [
        (coeff, exponent)
        for exponent, coeff in merged.items()
        if coeff % prime != 0
    ]
    canonical.sort(key=term_sort_key)
    return tuple(canonical)


def split_order_key(poly_key: str) -> tuple[int, str]:
    return (len(poly_key), poly_key)


def order_pair_by_key(left_key: str, right_key: str) -> bool:
    return split_order_key(left_key) <= split_order_key(right_key)


def modular_inverse(value: int, prime: int) -> int:
    value %= prime
    if value == 0:
        raise ZeroDivisionError("0 has no multiplicative inverse in a finite field")
    return pow(value, prime - 2, prime)

