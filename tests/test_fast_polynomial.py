"""Tests for the fast numpy-based polynomial backend."""

import numpy as np
import pytest

from src.environment.fast_polynomial import FastPoly


class TestFastPolyConstruction:
    def test_zero(self):
        p = FastPoly.zero(2, 6, 5)
        assert p.is_zero()
        assert p.coeffs.shape == (7, 7)

    def test_constant(self):
        p = FastPoly.constant(3, 2, 6, 5)
        assert p.coeffs[0, 0] == 3
        assert not p.is_zero()

    def test_constant_mod(self):
        p = FastPoly.constant(7, 2, 6, 5)
        assert p.coeffs[0, 0] == 2  # 7 mod 5 = 2

    def test_variable(self):
        p = FastPoly.variable(0, 2, 6, 5)
        assert p.coeffs[1, 0] == 1  # coefficient of x0
        assert p.coeffs[0, 1] == 0  # coefficient of x1

    def test_variable_1(self):
        p = FastPoly.variable(1, 2, 6, 5)
        assert p.coeffs[0, 1] == 1  # coefficient of x1
        assert p.coeffs[1, 0] == 0


class TestFastPolyArithmetic:
    def setup_method(self):
        self.x0 = FastPoly.variable(0, 2, 6, 5)
        self.x1 = FastPoly.variable(1, 2, 6, 5)
        self.one = FastPoly.constant(1, 2, 6, 5)

    def test_add(self):
        result = self.x0 + self.x1
        assert result.coeffs[1, 0] == 1  # x0
        assert result.coeffs[0, 1] == 1  # x1

    def test_add_mod(self):
        # 3*x0 + 3*x0 = 6*x0 = x0 mod 5
        p = FastPoly.zero(2, 6, 5)
        p.coeffs[1, 0] = 3
        result = p + p
        assert result.coeffs[1, 0] == 1  # 6 mod 5 = 1

    def test_multiply(self):
        result = self.x0 * self.x1
        assert result.coeffs[1, 1] == 1  # x0*x1

    def test_multiply_self(self):
        result = self.x0 * self.x0
        assert result.coeffs[2, 0] == 1  # x0^2

    def test_multiply_mod(self):
        # 3*x0 * 2*x0 = 6*x0^2 = x0^2 mod 5
        p1 = FastPoly.zero(2, 6, 5)
        p1.coeffs[1, 0] = 3
        p2 = FastPoly.zero(2, 6, 5)
        p2.coeffs[1, 0] = 2
        result = p1 * p2
        assert result.coeffs[2, 0] == 1  # 6 mod 5 = 1

    def test_add_constant(self):
        result = self.x0 + self.one
        assert result.coeffs[1, 0] == 1  # x0
        assert result.coeffs[0, 0] == 1  # 1

    def test_square_expansion(self):
        # (x0 + x1)^2 = x0^2 + 2*x0*x1 + x1^2
        s = self.x0 + self.x1
        result = s * s
        assert result.coeffs[2, 0] == 1  # x0^2
        assert result.coeffs[1, 1] == 2  # 2*x0*x1
        assert result.coeffs[0, 2] == 1  # x1^2

    def test_multiply_truncation(self):
        """Multiplying high-degree polynomials truncates beyond max_degree."""
        # x0^6 * x0 would be x0^7, but max_degree=6, so truncated
        p1 = FastPoly.zero(2, 6, 5)
        p1.coeffs[6, 0] = 1  # x0^6
        result = p1 * self.x0  # would be x0^7, truncated
        assert result.coeffs.shape == (7, 7)
        # x0^7 is beyond max_degree, should not appear


class TestFastPolyComparison:
    def setup_method(self):
        self.x0 = FastPoly.variable(0, 2, 6, 5)
        self.x1 = FastPoly.variable(1, 2, 6, 5)

    def test_equal(self):
        a = self.x0 + self.x1
        b = self.x1 + self.x0
        assert a == b

    def test_not_equal(self):
        assert self.x0 != self.x1

    def test_canonical_key_equal(self):
        a = self.x0 + self.x1
        b = self.x1 + self.x0
        assert a.canonical_key() == b.canonical_key()

    def test_canonical_key_different(self):
        assert self.x0.canonical_key() != self.x1.canonical_key()

    def test_hash_consistency(self):
        a = self.x0 + self.x1
        b = self.x1 + self.x0
        assert hash(a) == hash(b)
        # Can use in sets/dicts
        s = {a}
        assert b in s


class TestFastPolyTermSimilarity:
    def setup_method(self):
        self.x0 = FastPoly.variable(0, 2, 6, 5)
        self.x1 = FastPoly.variable(1, 2, 6, 5)

    def test_identical(self):
        p = self.x0 + self.x1
        assert p.term_similarity(p) == 1.0

    def test_completely_different(self):
        assert self.x0.term_similarity(self.x1) == 0.0

    def test_partial_match(self):
        current = self.x0 + self.x1
        # target = x0 + 2*x1 (x0 matches, x1 doesn't)
        target = FastPoly.zero(2, 6, 5)
        target.coeffs[1, 0] = 1
        target.coeffs[0, 1] = 2
        assert current.term_similarity(target) == 0.5

    def test_zero_target(self):
        zero = FastPoly.zero(2, 6, 5)
        assert zero.term_similarity(zero) == 1.0
        assert self.x0.term_similarity(zero) == 0.0


class TestFastPolyVector:
    def test_to_vector_shape(self):
        p = FastPoly.variable(0, 2, 6, 5)
        vec = p.to_vector()
        assert vec.shape == (49,)  # (7*7)

    def test_to_vector_values(self):
        p = FastPoly.variable(0, 2, 6, 5)
        vec = p.to_vector()
        # x0 = coeffs[1,0], which is index 1*7 + 0 = 7 in flattened array
        assert vec[7] == 1.0
        assert vec.sum() == 1.0


class TestFastPolyCopy:
    def test_copy_independence(self):
        p = FastPoly.variable(0, 2, 6, 5)
        q = p.copy()
        q.coeffs[0, 0] = 3
        assert p.coeffs[0, 0] == 0  # original unchanged


class TestFastPoly3Vars:
    """Test with 3 variables to verify n-d generalization."""

    def test_3var_construction(self):
        x0 = FastPoly.variable(0, 3, 4, 5)
        x1 = FastPoly.variable(1, 3, 4, 5)
        x2 = FastPoly.variable(2, 3, 4, 5)
        assert x0.coeffs.shape == (5, 5, 5)

    def test_3var_multiply(self):
        x0 = FastPoly.variable(0, 3, 4, 5)
        x1 = FastPoly.variable(1, 3, 4, 5)
        x2 = FastPoly.variable(2, 3, 4, 5)
        result = x0 * x1 * x2
        assert result.coeffs[1, 1, 1] == 1  # x0*x1*x2

    def test_3var_add(self):
        x0 = FastPoly.variable(0, 3, 4, 5)
        x1 = FastPoly.variable(1, 3, 4, 5)
        x2 = FastPoly.variable(2, 3, 4, 5)
        result = x0 + x1 + x2
        assert result.coeffs[1, 0, 0] == 1
        assert result.coeffs[0, 1, 0] == 1
        assert result.coeffs[0, 0, 1] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
