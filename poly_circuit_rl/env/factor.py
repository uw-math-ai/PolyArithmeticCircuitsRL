from __future__ import annotations

from fractions import Fraction
from typing import Dict, Tuple

import sympy as sp

Exponent = Tuple[int, ...]
PolyDict = Dict[Exponent, Fraction | int]


def poly_dict_to_expr(poly: PolyDict, n_vars: int) -> sp.Expr:
    """Convert internal sparse poly dict to a SymPy expression."""
    var_syms = [sp.Symbol("x")] if n_vars == 1 else list(sp.symbols(f"x0:{n_vars}"))
    expr = sp.Integer(0)
    for mono, coeff in poly.items():
        term = sp.Rational(coeff)
        for v, e in zip(var_syms, mono):
            if e:
                term *= v ** e
        expr += term
    return expr
