"""Greedy heuristic baseline: pick the operation minimizing eval-distance.

At each step, evaluates all valid ADD/MUL pairs and chooses the one whose
result polynomial has minimum L1 eval-distance to the target.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional

from ..config import Config
from ..core.builder import CircuitBuilder
from ..core.fingerprints import sample_eval_points, eval_poly_points, EvalPoint
from ..core.poly import Poly, equal


class GreedyBaseline:
    """Greedy heuristic: minimize eval-distance to target at each step."""

    def __init__(self, config: Config, seed: int = 42):
        self.config = config
        self.rng = random.Random(seed)
        self.eval_points: List[EvalPoint] = sample_eval_points(
            self.rng, config.n_vars, config.m, config.eval_low, config.eval_high,
        )

    def solve(
        self,
        target: Poly,
        max_ops: int,
    ) -> Dict:
        """Attempt to construct target using greedy strategy.

        At each step, tries all valid (op, i, j) combinations and picks
        the one whose result has minimum L1 eval-distance to the target.

        Returns dict with 'solved', 'ops_used', 'nodes_built'.
        """
        builder = CircuitBuilder(
            self.config.n_vars,
            eval_points=self.eval_points,
            include_const_one=True,
        )

        target_evals = eval_poly_points(target, self.eval_points)

        for op_num in range(max_ops):
            # Check if we already have the target
            for node in builder.nodes:
                if equal(node.poly, target):
                    return {
                        "solved": True,
                        "ops_used": op_num,
                        "nodes_built": len(builder.nodes),
                    }

            n = len(builder.nodes)
            best_dist = float("inf")
            best_action = None

            for i in range(n):
                for j in range(i, n):
                    for op_name, op_fn in [("add", builder.add_add), ("mul", builder.add_mul)]:
                        # Clone builder to test the operation
                        test_builder = builder.clone()
                        if op_name == "add":
                            result = test_builder.add_add(i, j)
                        else:
                            result = test_builder.add_mul(i, j)

                        if result.reused:
                            continue

                        new_node = test_builder.nodes[result.node_id]
                        new_evals = (
                            new_node.evals
                            if new_node.evals is not None
                            else eval_poly_points(new_node.poly, self.eval_points)
                        )
                        dist = sum(
                            abs(float(a) - float(b))
                            for a, b in zip(new_evals, target_evals)
                        )

                        if dist < best_dist:
                            best_dist = dist
                            best_action = (op_name, i, j)

            if best_action is None:
                break

            op_name, i, j = best_action
            if op_name == "add":
                builder.add_add(i, j)
            else:
                builder.add_mul(i, j)

        # Final check
        for node in builder.nodes:
            if equal(node.poly, target):
                return {
                    "solved": True,
                    "ops_used": max_ops,
                    "nodes_built": len(builder.nodes),
                }

        return {
            "solved": False,
            "ops_used": max_ops,
            "nodes_built": len(builder.nodes),
        }

    def evaluate_batch(
        self,
        targets: List[Poly],
        max_ops: int,
    ) -> Dict:
        """Evaluate greedy baseline on a batch of targets.

        Returns aggregate statistics.
        """
        solved = 0
        total_ops = 0

        for target in targets:
            result = self.solve(target, max_ops)
            if result["solved"]:
                solved += 1
                total_ops += result["ops_used"]

        return {
            "success_rate": solved / max(len(targets), 1),
            "avg_ops": total_ops / max(solved, 1),
            "total_targets": len(targets),
            "total_solved": solved,
        }
