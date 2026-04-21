"""Memoized common-subexpression symbolic baseline via CSE."""

from __future__ import annotations

from typing import List, Optional

import sympy as sp

from ..config import Config
from ..core.poly import Poly
from .symbolic_utils import aggregate_stats, count_add_mul_ops, to_sympy_expr


class MemoizedCSEBaseline:
    """Baseline using symbolic CSE (dynamic-programming style reuse)."""

    def __init__(self, config: Config):
        self.config = config

    def solve(self, target: Poly, max_ops: Optional[int] = None) -> dict:
        expr = to_sympy_expr(target, self.config.n_vars)
        replacements, reduced = sp.cse(expr, optimizations="basic")

        ops_used = 0
        for _, rhs in replacements:
            ops_used += count_add_mul_ops(rhs)
        for out_expr in reduced:
            ops_used += count_add_mul_ops(out_expr)

        solved = max_ops is None or ops_used <= max_ops
        return {
            "solved": solved,
            "ops_used": ops_used,
            "method": "memoized_cse",
            "num_temporaries": len(replacements),
            "expr": str(reduced[0]) if reduced else "0",
        }

    def evaluate_batch(
        self,
        targets: List[Poly],
        max_ops: Optional[int] = None,
    ) -> dict:
        return aggregate_stats(self.solve(t, max_ops=max_ops) for t in targets)
