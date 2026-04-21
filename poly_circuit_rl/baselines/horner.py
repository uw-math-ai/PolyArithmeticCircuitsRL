"""Horner-style symbolic baseline."""

from __future__ import annotations

from typing import List, Optional, Sequence

import sympy as sp

from ..config import Config
from ..core.poly import Poly
from .symbolic_utils import (
    aggregate_stats,
    count_add_mul_ops,
    default_symbols,
    to_sympy_expr,
)


class HornerBaseline:
    """Baseline using nested Horner decomposition."""

    def __init__(self, config: Config):
        self.config = config

    def solve(
        self,
        target: Poly,
        max_ops: Optional[int] = None,
        var_order: Optional[Sequence[int]] = None,
    ) -> dict:
        expr = to_sympy_expr(target, self.config.n_vars)
        symbols = list(default_symbols(self.config.n_vars))

        if var_order is None:
            order = list(range(self.config.n_vars))
        else:
            order = list(var_order)
            if sorted(order) != list(range(self.config.n_vars)):
                raise ValueError("var_order must be a permutation of variable indices")

        horner_expr = expr
        for idx in order:
            horner_expr = sp.horner(horner_expr, symbols[idx])

        ops_used = count_add_mul_ops(horner_expr)
        solved = max_ops is None or ops_used <= max_ops
        return {
            "solved": solved,
            "ops_used": ops_used,
            "method": "horner",
            "expr": str(horner_expr),
        }

    def evaluate_batch(
        self,
        targets: List[Poly],
        max_ops: Optional[int] = None,
        var_order: Optional[Sequence[int]] = None,
    ) -> dict:
        return aggregate_stats(
            self.solve(t, max_ops=max_ops, var_order=var_order)
            for t in targets
        )
