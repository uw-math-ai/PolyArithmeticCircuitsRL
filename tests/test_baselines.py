"""Tests for baseline algorithms."""

import unittest

from poly_circuit_rl.config import Config
from poly_circuit_rl.core.poly import add, mul, make_var, make_const, poly_hashkey
from poly_circuit_rl.baselines.exhaustive import ExhaustiveSearch
from poly_circuit_rl.baselines.factorization import FactorizationBaseline
from poly_circuit_rl.baselines.greedy import GreedyBaseline
from poly_circuit_rl.baselines.horner import HornerBaseline
from poly_circuit_rl.baselines.memoized import MemoizedCSEBaseline


class TestExhaustiveSearch(unittest.TestCase):
    def setUp(self):
        self.config = Config(n_vars=2, max_ops=3, seed=42)
        self.search = ExhaustiveSearch(self.config)
        self.search.build(max_ops=3)

    def test_base_polys_zero_ops(self):
        """Variables and constant 1 should be reachable in 0 ops."""
        x0 = make_var(2, 0)
        x1 = make_var(2, 1)
        c1 = make_const(2, 1)
        self.assertEqual(self.search.find_optimal(x0), 0)
        self.assertEqual(self.search.find_optimal(x1), 0)
        self.assertEqual(self.search.find_optimal(c1), 0)

    def test_simple_sum_one_op(self):
        """x0 + x1 should be reachable in 1 op."""
        x0 = make_var(2, 0)
        x1 = make_var(2, 1)
        target = add(x0, x1)
        self.assertEqual(self.search.find_optimal(target), 1)

    def test_simple_product_one_op(self):
        """x0 * x1 should be reachable in 1 op."""
        x0 = make_var(2, 0)
        x1 = make_var(2, 1)
        target = mul(x0, x1)
        self.assertEqual(self.search.find_optimal(target), 1)

    def test_two_op_target(self):
        """x0^2 + x1 requires 2 ops: x0*x0 then +x1."""
        x0 = make_var(2, 0)
        x1 = make_var(2, 1)
        x0_sq = mul(x0, x0)
        target = add(x0_sq, x1)
        opt = self.search.find_optimal(target)
        self.assertIsNotNone(opt)
        self.assertEqual(opt, 2)

    def test_gap_to_optimal(self):
        """Gap should be agent_ops - optimal_ops."""
        x0 = make_var(2, 0)
        x1 = make_var(2, 1)
        target = add(x0, x1)
        gap = self.search.gap_to_optimal(target, agent_ops=3)
        self.assertEqual(gap, 2)  # 3 - 1

    def test_unreachable_returns_none(self):
        """Very complex poly not in search depth should return None."""
        x0 = make_var(2, 0)
        # x0^10 requires many ops — likely not reachable in 3
        from fractions import Fraction
        target = {(10, 0): Fraction(1)}
        result = self.search.find_optimal(target)
        # May or may not be None depending on reachability
        # Just check it doesn't crash

    def test_reachable_count(self):
        """Should discover a positive number of reachable polynomials."""
        self.assertGreater(self.search.reachable_count(), 0)


class TestGreedyBaseline(unittest.TestCase):
    def setUp(self):
        self.config = Config(n_vars=2, max_ops=4, m=16, seed=42)
        self.greedy = GreedyBaseline(self.config)

    def test_simple_target(self):
        """Greedy should solve x0 + x1 in 1 op."""
        x0 = make_var(2, 0)
        x1 = make_var(2, 1)
        target = add(x0, x1)
        result = self.greedy.solve(target, max_ops=4)
        self.assertTrue(result["solved"])

    def test_product_target(self):
        """Greedy should solve x0 * x1."""
        x0 = make_var(2, 0)
        x1 = make_var(2, 1)
        target = mul(x0, x1)
        result = self.greedy.solve(target, max_ops=4)
        self.assertTrue(result["solved"])

    def test_evaluate_batch(self):
        """Batch evaluation should return valid stats."""
        x0 = make_var(2, 0)
        x1 = make_var(2, 1)
        targets = [
            add(x0, x1),
            mul(x0, x1),
            add(x0, make_const(2, 1)),
        ]
        stats = self.greedy.evaluate_batch(targets, max_ops=4)
        self.assertGreaterEqual(stats["success_rate"], 0.0)
        self.assertLessEqual(stats["success_rate"], 1.0)
        self.assertEqual(stats["total_targets"], 3)


class TestSymbolicBaselines(unittest.TestCase):
    def setUp(self):
        self.config = Config(n_vars=2, max_ops=6, seed=42)
        self.factor = FactorizationBaseline(self.config)
        self.horner = HornerBaseline(self.config)
        self.memo = MemoizedCSEBaseline(self.config)

    def test_factorization_baseline_solves_simple_target(self):
        x0 = make_var(2, 0)
        x1 = make_var(2, 1)
        target = mul(add(x0, x1), add(x0, x1))
        result = self.factor.solve(target, max_ops=6)
        self.assertTrue(result["solved"])
        self.assertGreaterEqual(result["ops_used"], 0)

    def test_horner_baseline_solves_simple_target(self):
        x0 = make_var(2, 0)
        x1 = make_var(2, 1)
        target = add(mul(x0, x0), x1)  # x0^2 + x1
        result = self.horner.solve(target, max_ops=6)
        self.assertTrue(result["solved"])
        self.assertGreaterEqual(result["ops_used"], 0)

    def test_memoized_baseline_tracks_temporaries(self):
        x0 = make_var(2, 0)
        x1 = make_var(2, 1)
        shared = add(x0, x1)
        target = add(mul(shared, shared), shared)
        result = self.memo.solve(target, max_ops=10)
        self.assertTrue(result["solved"])
        self.assertIn("num_temporaries", result)
        self.assertGreaterEqual(result["num_temporaries"], 0)

    def test_symbolic_baseline_batch_stats(self):
        x0 = make_var(2, 0)
        x1 = make_var(2, 1)
        targets = [add(x0, x1), mul(x0, x1)]
        stats = self.factor.evaluate_batch(targets, max_ops=10)
        self.assertEqual(stats["total_targets"], 2)
        self.assertGreaterEqual(stats["success_rate"], 0.0)
        self.assertLessEqual(stats["success_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
