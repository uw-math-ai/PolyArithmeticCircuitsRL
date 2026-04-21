"""Exhaustive search baseline: BFS over all possible circuits.

For small instances (2 vars, <=4 ops), enumerates all reachable polynomials
layer-by-layer and finds the minimum-operation circuit for each target.

This provides the optimal solution count for NeurIPS gap-to-optimal metrics.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Tuple

from ..config import Config
from ..core.builder import CircuitBuilder
from ..core.poly import Poly, PolyKey, poly_hashkey, add as poly_add, mul as poly_mul


class ExhaustiveSearch:
    """BFS exhaustive search over all possible circuits up to max_ops operations."""

    def __init__(self, config: Config):
        self.config = config
        self.n_vars = config.n_vars
        # Cache: poly_hashkey -> minimum ops to construct
        self._optimal: Dict[PolyKey, int] = {}
        self._built_up_to: int = 0

    def build(self, max_ops: int) -> None:
        """BFS-enumerate all reachable polynomials up to max_ops operations.

        At each layer, combines all pairs of existing polynomials via ADD and MUL
        to discover new polynomials, tracking the minimum ops needed.
        """
        if max_ops <= self._built_up_to:
            return

        # Layer 0: base polynomials (variables + constant 1)
        builder = CircuitBuilder(self.n_vars, eval_points=None)
        polys_at_layer: Dict[int, List[Poly]] = {}
        polys_at_layer[0] = [node.poly for node in builder.nodes]

        all_polys: Dict[PolyKey, Poly] = {}
        for p in polys_at_layer[0]:
            key = poly_hashkey(p)
            if key not in self._optimal:
                self._optimal[key] = 0
            all_polys[key] = p

        for ops in range(1, max_ops + 1):
            new_polys: List[Poly] = []

            # Collect all polynomials reachable in fewer ops
            pool: List[Poly] = []
            for layer in range(ops):
                pool.extend(polys_at_layer.get(layer, []))

            # Try all pairs
            for i, p1 in enumerate(pool):
                for p2 in pool[i:]:
                    for op_fn in (poly_add, poly_mul):
                        result = op_fn(p1, p2)
                        key = poly_hashkey(result)
                        if key not in self._optimal:
                            self._optimal[key] = ops
                            all_polys[key] = result
                            new_polys.append(result)

            polys_at_layer[ops] = new_polys

        self._built_up_to = max_ops

    def find_optimal(self, target: Poly) -> Optional[int]:
        """Find minimum number of operations to construct target.

        Returns the minimum op count, or None if target cannot be
        constructed within the built search depth.
        """
        key = poly_hashkey(target)
        return self._optimal.get(key)

    def find_all_optimal(
        self, targets: List[Poly],
    ) -> Dict[PolyKey, Optional[int]]:
        """Find optimal op counts for a batch of targets."""
        return {
            poly_hashkey(t): self.find_optimal(t)
            for t in targets
        }

    def reachable_count(self) -> int:
        """Number of distinct polynomials reachable within the search depth."""
        return len(self._optimal)

    def gap_to_optimal(self, target: Poly, agent_ops: int) -> Optional[int]:
        """Compute gap between agent's solution and optimal.

        Returns agent_ops - optimal_ops, or None if optimal is unknown.
        """
        opt = self.find_optimal(target)
        if opt is None:
            return None
        return agent_ops - opt
