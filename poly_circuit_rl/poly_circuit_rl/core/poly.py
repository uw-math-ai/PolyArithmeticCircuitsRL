"""Exact sparse polynomial arithmetic over the rationals.

A polynomial is represented as a dictionary mapping monomial exponent tuples
to rational coefficients (Fraction).  This exact representation avoids any
floating-point rounding and makes polynomial identity testing fully reliable.

  Poly  =  Dict[Exponent, Fraction]
  Exponent = Tuple[int, ...]   (one int per variable, giving that variable's degree)

Example (2 variables x0, x1):
  x0^2 * x1 + 3  →  {(2, 1): Fraction(1), (0, 0): Fraction(3)}

The zero polynomial is represented as the empty dict {}.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Dict, Iterable, Tuple

# Exponent tuple: element i is the degree of variable x_i in the monomial.
Exponent = Tuple[int, ...]

# Polynomial type: maps each monomial to its rational coefficient.
# Zero-coefficient terms are not stored (see canonicalize).
Poly = Dict[Exponent, Fraction]


@dataclass(frozen=True)
class PolyStats:
    """Summary statistics of a polynomial, used for analysis and logging."""

    deg_total: int          # Total degree (max sum of exponents across all monomials)
    deg_per_var: Tuple[int, ...]  # Max degree of each variable
    num_terms: int          # Number of non-zero monomials
    coeff_l1: Fraction      # L1 norm of coefficients (sum of |coeff|)


def make_zero() -> Poly:
    """Return the zero polynomial (empty dict)."""
    return {}


def make_const(n_vars: int, value: int | Fraction) -> Poly:
    """Return the constant polynomial c (a degree-0 term in n_vars variables)."""
    coeff = Fraction(value)
    if coeff == 0:
        return {}
    return {(0,) * n_vars: coeff}


def make_var(n_vars: int, idx: int) -> Poly:
    """Return the polynomial representing the single variable x_idx."""
    if idx < 0 or idx >= n_vars:
        raise ValueError(f"Invalid variable index {idx} for n_vars={n_vars}")
    exp = [0] * n_vars
    exp[idx] = 1
    return {tuple(exp): Fraction(1)}


def canonicalize(poly: Poly) -> Poly:
    """Drop any monomials with coefficient zero.

    All arithmetic functions call this internally, so results are always in
    canonical form (no zero terms).
    """
    return {mono: coeff for mono, coeff in poly.items() if coeff != 0}


def add(a: Poly, b: Poly) -> Poly:
    """Return the polynomial a + b."""
    if not a:
        return dict(b)
    if not b:
        return dict(a)
    out: Poly = dict(a)
    for mono, coeff in b.items():
        out[mono] = out.get(mono, Fraction(0)) + coeff
    return canonicalize(out)


def sub(a: Poly, b: Poly) -> Poly:
    """Return the polynomial a - b."""
    if not b:
        return dict(a)
    out: Poly = dict(a)
    for mono, coeff in b.items():
        out[mono] = out.get(mono, Fraction(0)) - coeff
    return canonicalize(out)


def mul(a: Poly, b: Poly) -> Poly:
    """Return the polynomial a * b (distributive multiplication of monomials)."""
    if not a or not b:
        return {}
    out: Poly = {}
    for mono_a, coeff_a in a.items():
        for mono_b, coeff_b in b.items():
            # Multiply monomials by adding their exponents component-wise
            mono = tuple(x + y for x, y in zip(mono_a, mono_b))
            out[mono] = out.get(mono, Fraction(0)) + coeff_a * coeff_b
    return canonicalize(out)


def equal(a: Poly, b: Poly) -> bool:
    """Return True iff a and b represent the same polynomial (exact comparison)."""
    return canonicalize(a) == canonicalize(b)


def stats(poly: Poly, n_vars: int) -> PolyStats:
    """Compute summary statistics for a polynomial."""
    if not poly:
        return PolyStats(
            deg_total=0,
            deg_per_var=(0,) * n_vars,
            num_terms=0,
            coeff_l1=Fraction(0),
        )
    deg_per_var = [0] * n_vars
    deg_total = 0
    coeff_l1 = Fraction(0)
    for mono, coeff in poly.items():
        coeff_l1 += abs(coeff)
        deg_total = max(deg_total, sum(mono))
        for i, exp in enumerate(mono):
            if exp > deg_per_var[i]:
                deg_per_var[i] = exp
    return PolyStats(
        deg_total=deg_total,
        deg_per_var=tuple(deg_per_var),
        num_terms=len(poly),
        coeff_l1=coeff_l1,
    )


def eval_poly(poly: Poly, values: Iterable[int | Fraction]) -> Fraction:
    """Evaluate the polynomial at a point by substituting each variable.

    Args:
        poly:   The polynomial to evaluate.
        values: Sequence of values, one per variable (in order x0, x1, ...).

    Returns:
        The exact rational value of poly at the given point.
    """
    vals = list(values)
    if not poly:
        return Fraction(0)
    total = Fraction(0)
    for mono, coeff in poly.items():
        term = coeff
        for exp, v in zip(mono, vals):
            if exp == 0:
                continue
            term *= Fraction(v) ** exp
        total += term
    return total
