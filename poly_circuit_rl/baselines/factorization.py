"""Symbolic factorization baseline."""

from __future__ import annotations

from typing import List, Optional

import sympy as sp

from ..config import Config
from ..core.poly import Poly
from .symbolic_utils import aggregate_stats, count_add_mul_ops, to_sympy_expr


class FactorizationBaseline:
    """Baseline using direct symbolic factorization."""

    def __init__(self, config: Config):
        self.config = config

    def solve(self, target: Poly, max_ops: Optional[int] = None) -> dict:
        expr = to_sympy_expr(target, self.config.n_vars)
        factored = sp.factor(expr)
        ops_used = count_add_mul_ops(factored)
        solved = max_ops is None or ops_used <= max_ops
        return {
            "solved": solved,
            "ops_used": ops_used,
            "method": "factorization",
            "expr": str(factored),
        }

    def evaluate_batch(
        self,
        targets: List[Poly],
        max_ops: Optional[int] = None,
    ) -> dict:
        return aggregate_stats(self.solve(t, max_ops=max_ops) for t in targets)
