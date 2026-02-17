"""Polynomial utility functions with modular arithmetic.

Contains both SymPy-based functions (for display/debugging) and
conversion utilities between SymPy expressions and FastPoly objects.
"""

from typing import List, Optional

import numpy as np
import sympy
from sympy import Symbol, Expr, expand, Poly, ZZ, symbols as sympy_symbols

from .fast_polynomial import FastPoly


def create_variables(n: int) -> List[Symbol]:
    """Create n SymPy symbols: x0, x1, ..., x_{n-1}."""
    return list(sympy_symbols(f"x0:{n}"))


def mod_reduce(expr: Expr, syms: List[Symbol], mod: int) -> Expr:
    """Expand expression and reduce all coefficients mod p.

    Returns a SymPy expression with all integer coefficients in [0, mod-1].
    """
    expr = expand(expr)
    if expr.is_number:
        return sympy.Integer(int(expr) % mod)

    poly = Poly(expr, *syms, domain=ZZ)
    reduced_terms = []
    for monom, coeff in zip(poly.monoms(), poly.coeffs()):
        c = int(coeff) % mod
        if c == 0:
            continue
        term = sympy.Integer(c)
        for sym, power in zip(syms, monom):
            if power > 0:
                term *= sym ** power
        reduced_terms.append(term)

    if not reduced_terms:
        return sympy.Integer(0)
    return sum(reduced_terms)


def canonical_key(expr: Expr, syms: List[Symbol], mod: int) -> str:
    """Return a canonical string key for deduplication.

    Uses srepr of the mod-reduced expanded expression.
    """
    reduced = mod_reduce(expr, syms, mod)
    return sympy.srepr(reduced)


def poly_equal(expr1: Expr, expr2: Expr, syms: List[Symbol], mod: int) -> bool:
    """Check if two expressions are equal mod p."""
    diff = mod_reduce(expand(expr1 - expr2), syms, mod)
    return diff == sympy.Integer(0)


def get_monomial_list(n_vars: int, max_degree: int) -> List[tuple]:
    """Get sorted list of monomial exponent tuples for n variables up to max_degree.

    Returns tuples (e0, e1, ..., e_{n-1}) where sum(ei) <= max_degree,
    sorted in graded lexicographic order.
    """
    if n_vars == 0:
        return [()]

    monomials = []

    def _recurse(remaining_vars: int, remaining_degree: int, current: tuple):
        if remaining_vars == 0:
            monomials.append(current)
            return
        for d in range(remaining_degree + 1):
            _recurse(remaining_vars - 1, remaining_degree - d, current + (d,))

    _recurse(n_vars, max_degree, ())
    # Sort: first by total degree, then lexicographically
    monomials.sort(key=lambda m: (sum(m), m))
    return monomials


def poly_to_coefficient_vector(expr: Expr, syms: List[Symbol], mod: int,
                                max_degree: int) -> List[int]:
    """Convert a polynomial to a coefficient vector over all monomials up to max_degree.

    Returns a list of integer coefficients mod p, one per monomial.
    """
    reduced = mod_reduce(expr, syms, mod)
    monomials = get_monomial_list(len(syms), max_degree)

    if reduced == sympy.Integer(0):
        return [0] * len(monomials)

    poly = Poly(reduced, *syms, domain=ZZ) if not reduced.is_number else None

    coeff_dict = {}
    if poly is not None:
        for monom, coeff in zip(poly.monoms(), poly.coeffs()):
            coeff_dict[monom] = int(coeff) % mod
    elif reduced.is_number:
        zero_monom = (0,) * len(syms)
        coeff_dict[zero_monom] = int(reduced) % mod

    result = []
    for monom in monomials:
        result.append(coeff_dict.get(monom, 0))
    return result


def term_similarity(current: Expr, target: Expr, syms: List[Symbol],
                     mod: int, max_degree: int) -> float:
    """Compute fraction of matching terms between current and target polynomials.

    Used for potential-based reward shaping.
    """
    curr_vec = poly_to_coefficient_vector(current, syms, mod, max_degree)
    targ_vec = poly_to_coefficient_vector(target, syms, mod, max_degree)

    if all(c == 0 for c in targ_vec):
        return 1.0 if all(c == 0 for c in curr_vec) else 0.0

    # Count matching non-zero coefficients
    total_target_terms = sum(1 for c in targ_vec if c != 0)
    if total_target_terms == 0:
        return 1.0

    matching = sum(1 for c, t in zip(curr_vec, targ_vec) if c == t and t != 0)
    return matching / total_target_terms


# ---- SymPy <-> FastPoly conversions ----

def sympy_to_fast(expr: Expr, syms: List[Symbol], mod: int, max_degree: int) -> FastPoly:
    """Convert a SymPy expression to a FastPoly.

    Useful for creating test targets from readable SymPy expressions.
    """
    reduced = mod_reduce(expr, syms, mod)
    n_vars = len(syms)
    shape = (max_degree + 1,) * n_vars
    coeffs = np.zeros(shape, dtype=np.int64)

    if reduced == sympy.Integer(0):
        return FastPoly(coeffs, mod)

    if reduced.is_number:
        idx = (0,) * n_vars
        coeffs[idx] = int(reduced) % mod
        return FastPoly(coeffs, mod)

    poly = Poly(reduced, *syms, domain=ZZ)
    for monom, coeff in zip(poly.monoms(), poly.coeffs()):
        c = int(coeff) % mod
        if c != 0 and all(e <= max_degree for e in monom):
            coeffs[monom] = c

    return FastPoly(coeffs, mod)


def fast_to_sympy(poly: FastPoly, syms: List[Symbol]) -> Expr:
    """Convert a FastPoly back to a SymPy expression.

    Useful for display and debugging.
    """
    terms = []
    for idx in zip(*np.nonzero(poly.coeffs)):
        c = int(poly.coeffs[idx])
        term = sympy.Integer(c)
        for var_i, power in enumerate(idx):
            if power > 0:
                term *= syms[var_i] ** power
        terms.append(term)

    if not terms:
        return sympy.Integer(0)
    return sum(terms)
