"""Tests for the factor library."""

import unittest
from fractions import Fraction

from poly_circuit_rl.core.poly import (
    Poly, add, mul, make_var, make_const, poly_hashkey, is_scalar,
)
from poly_circuit_rl.core.builder import CircuitBuilder
from poly_circuit_rl.core.factor_library import FactorLibrary
from poly_circuit_rl.core.node import Node, OP_VAR, OP_ADD


class TestPolyHashkey(unittest.TestCase):
    def test_equal_polys_same_key(self):
        p1 = {(1, 0): Fraction(1), (0, 1): Fraction(2)}
        p2 = {(0, 1): Fraction(2), (1, 0): Fraction(1)}
        self.assertEqual(poly_hashkey(p1), poly_hashkey(p2))

    def test_different_polys_different_key(self):
        p1 = {(1, 0): Fraction(1)}
        p2 = {(0, 1): Fraction(1)}
        self.assertNotEqual(poly_hashkey(p1), poly_hashkey(p2))

    def test_zero_poly_key(self):
        self.assertEqual(poly_hashkey({}), ())


class TestIsScalar(unittest.TestCase):
    def test_empty_is_scalar(self):
        self.assertTrue(is_scalar({}))

    def test_constant_is_scalar(self):
        self.assertTrue(is_scalar({(0, 0): Fraction(5)}))

    def test_variable_not_scalar(self):
        self.assertFalse(is_scalar({(1, 0): Fraction(1)}))


class TestFactorLibrary(unittest.TestCase):
    def setUp(self):
        self.lib = FactorLibrary(n_vars=2)

    def test_base_keys_populated(self):
        # x0, x1, and const_1 should be base keys
        x0 = make_var(2, 0)
        x1 = make_var(2, 1)
        c1 = make_const(2, 1)
        self.assertTrue(self.lib.is_base(x0))
        self.assertTrue(self.lib.is_base(x1))
        self.assertTrue(self.lib.is_base(c1))

    def test_non_base_not_base(self):
        # x0 + x1 is not a base node
        x0 = make_var(2, 0)
        x1 = make_var(2, 1)
        p = add(x0, x1)
        self.assertFalse(self.lib.is_base(p))

    def test_register_and_contains(self):
        x0 = make_var(2, 0)
        x1 = make_var(2, 1)
        p = add(x0, x1)  # x0 + x1
        self.assertFalse(self.lib.contains(p))
        self.lib.register(p, step_num=1)
        self.assertTrue(self.lib.contains(p))

    def test_register_base_ignored(self):
        x0 = make_var(2, 0)
        self.lib.register(x0, step_num=1)
        self.assertEqual(len(self.lib), 0)

    def test_register_keeps_min_step(self):
        x0 = make_var(2, 0)
        x1 = make_var(2, 1)
        p = add(x0, x1)
        self.lib.register(p, step_num=5)
        self.lib.register(p, step_num=2)
        self.lib.register(p, step_num=10)
        self.assertEqual(self.lib._known[poly_hashkey(p)], 2)

    def test_factorize_simple_square(self):
        # (x0 + 1)^2 = x0^2 + 2*x0 + 1
        x0 = make_var(2, 0)
        c1 = make_const(2, 1)
        x0_plus_1 = add(x0, c1)
        target = mul(x0_plus_1, x0_plus_1)

        factors = self.lib.factorize_target(target)
        # Should find x0 + 1 as a factor
        factor_keys = {poly_hashkey(f) for f in factors}
        self.assertIn(poly_hashkey(x0_plus_1), factor_keys)

    def test_factorize_irreducible(self):
        # x0 + x1 is irreducible
        x0 = make_var(2, 0)
        x1 = make_var(2, 1)
        p = add(x0, x1)
        factors = self.lib.factorize_target(p)
        self.assertEqual(len(factors), 0)

    def test_factorize_two_factors(self):
        # (x0 + 1) * (x1 + 1)
        x0 = make_var(2, 0)
        x1 = make_var(2, 1)
        c1 = make_const(2, 1)
        f1 = add(x0, c1)
        f2 = add(x1, c1)
        target = mul(f1, f2)

        factors = self.lib.factorize_target(target)
        factor_keys = {poly_hashkey(f) for f in factors}
        # Should find both x0+1 and x1+1
        self.assertIn(poly_hashkey(f1), factor_keys)
        self.assertIn(poly_hashkey(f2), factor_keys)

    def test_factorize_excludes_base_nodes(self):
        # x0 * (x0 + 1) — x0 is a base node, should be excluded
        x0 = make_var(2, 0)
        c1 = make_const(2, 1)
        x0_plus_1 = add(x0, c1)
        target = mul(x0, x0_plus_1)

        factors = self.lib.factorize_target(target)
        factor_keys = {poly_hashkey(f) for f in factors}
        self.assertNotIn(poly_hashkey(x0), factor_keys)
        self.assertIn(poly_hashkey(x0_plus_1), factor_keys)

    def test_exact_quotient_success(self):
        x0 = make_var(2, 0)
        x1 = make_var(2, 1)
        c1 = make_const(2, 1)
        f1 = add(x0, c1)
        f2 = add(x1, c1)
        product = mul(f1, f2)

        quotient = self.lib.exact_quotient(product, f1)
        self.assertIsNotNone(quotient)
        self.assertEqual(poly_hashkey(quotient), poly_hashkey(f2))

    def test_exact_quotient_fail(self):
        x0 = make_var(2, 0)
        x1 = make_var(2, 1)
        c1 = make_const(2, 1)
        dividend = add(x0, x1)
        divisor = add(x0, c1)

        quotient = self.lib.exact_quotient(dividend, divisor)
        self.assertIsNone(quotient)

    def test_filter_known(self):
        x0 = make_var(2, 0)
        c1 = make_const(2, 1)
        f1 = add(x0, c1)  # x0 + 1

        # Register f1
        self.lib.register(f1, step_num=1)

        known = self.lib.filter_known([f1])
        self.assertIn(poly_hashkey(f1), known)

    def test_filter_known_empty(self):
        x0 = make_var(2, 0)
        c1 = make_const(2, 1)
        f1 = add(x0, c1)

        known = self.lib.filter_known([f1])
        self.assertEqual(len(known), 0)

    def test_register_episode_nodes_respects_max_size(self):
        lib = FactorLibrary(n_vars=1, max_size=3)
        builder = CircuitBuilder(n_vars=1, eval_points=None, include_const_one=True)
        n_initial = 2  # x, 1
        r1 = builder.add_add(0, 1)
        r2 = builder.add_mul(0, 0)
        r3 = builder.add_add(r1.node_id, r2.node_id)
        r4 = builder.add_mul(r1.node_id, r2.node_id)

        lib.register_episode_nodes(builder.nodes, n_initial=n_initial)
        self.assertLessEqual(len(lib), 3)
        # Oldest inserted non-base node should have been evicted under LRU cap.
        self.assertFalse(lib.contains(builder.nodes[r1.node_id].poly))

    def test_frozen_view_ignores_write_operations(self):
        lib = FactorLibrary(n_vars=2, max_size=10)
        x0 = make_var(2, 0)
        x1 = make_var(2, 1)
        frozen = lib.frozen_view()

        frozen.register(add(x0, x1), step_num=1)
        self.assertEqual(len(lib), 0)


class TestFactorLibraryOneVar(unittest.TestCase):
    def test_one_var_factorize(self):
        lib = FactorLibrary(n_vars=1)
        # (x + 1)^2 = x^2 + 2x + 1
        x = make_var(1, 0)
        c1 = make_const(1, 1)
        x_plus_1 = add(x, c1)
        target = mul(x_plus_1, x_plus_1)

        factors = lib.factorize_target(target)
        factor_keys = {poly_hashkey(f) for f in factors}
        self.assertIn(poly_hashkey(x_plus_1), factor_keys)


if __name__ == "__main__":
    unittest.main()
