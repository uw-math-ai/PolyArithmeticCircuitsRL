"""Utilities shared by symbolic baselines."""

from __future__ import annotations

from typing import Iterable, Sequence

import sympy as sp

from ..core.poly import Poly
from ..env.factor import poly_dict_to_expr


def to_sympy_expr(poly: Poly, n_vars: int) -> sp.Expr:
    """Convert internal Poly dict representation to a SymPy expression."""
    return poly_dict_to_expr(poly, n_vars)


def default_symbols(n_vars: int) -> Sequence[sp.Symbol]:
    """Return default variable symbols in x0..x{n-1} order."""
    if n_vars == 1:
        return (sp.Symbol("x"),)
    return tuple(sp.symbols(f"x0:{n_vars}"))


def count_add_mul_ops(expr: sp.Expr) -> int:
    """Estimate operation count as number of ADD/MUL gates in expression tree.

    Rules:
    - Add(a,b,c) contributes 2 additions.
    - Mul(a,b,c) contributes 2 multiplications.
    - Pow(base, n) with integer n>=0 contributes (n-1) multiplications.
    - Atoms (symbols/constants) contribute 0.
    """
    if expr.is_Atom:
        return 0

    if expr.func is sp.Add:
        return max(len(expr.args) - 1, 0) + sum(
            count_add_mul_ops(a) for a in expr.args
        )

    if expr.func is sp.Mul:
        return max(len(expr.args) - 1, 0) + sum(
            count_add_mul_ops(a) for a in expr.args
        )

    if expr.func is sp.Pow and len(expr.args) == 2:
        base, exp = expr.args
        if exp.is_Integer and int(exp) >= 0:
            n = int(exp)
            return count_add_mul_ops(base) + max(n - 1, 0)
        return count_add_mul_ops(base) + count_add_mul_ops(exp)

    return sum(count_add_mul_ops(a) for a in expr.args)


def aggregate_stats(results: Iterable[dict]) -> dict:
    """Aggregate `solve()` outputs across a batch."""
    rows = list(results)
    solved_rows = [r for r in rows if r.get("solved", False)]
    return {
        "success_rate": len(solved_rows) / max(len(rows), 1),
        "avg_ops": (
            sum(r["ops_used"] for r in solved_rows) / max(len(solved_rows), 1)
        ),
        "total_targets": len(rows),
        "total_solved": len(solved_rows),
    }
