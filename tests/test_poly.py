import unittest
from fractions import Fraction

from poly_circuit_rl.core.poly import eval_poly


class TestPoly(unittest.TestCase):
    def test_eval_poly_rejects_too_few_values(self):
        poly = {(1, 0): Fraction(1), (0, 1): Fraction(1)}
        with self.assertRaises(AssertionError):
            eval_poly(poly, [2])


if __name__ == "__main__":
    unittest.main()
