"""Polynomial fingerprinting via Schwartz-Zippel evaluation.

Two polynomials are identical iff they agree on a sufficiently large random
sample of evaluation points (Schwartz-Zippel lemma).  We use m fixed random
integer points per environment instance as a fast polynomial identity check.

  sample_eval_points  — sample m random (n_vars,)-tuples once per env
  eval_poly_points    — evaluate a polynomial at all m points
  eval_distance       — L1 distance between two evaluation vectors

Note: exact Fraction arithmetic ensures no rounding errors.  At degree ≤ 4
with integer points in [-3, 3] and m=16, the probability of a false positive
(two distinct polys giving the same eval vector) is negligibly small.
"""

from __future__ import annotations

import random
from fractions import Fraction
from typing import Iterable, List, Sequence, Tuple

from .poly import Poly, eval_poly

# A single evaluation point: one integer value per variable.
EvalPoint = Tuple[int, ...]


def sample_eval_points(
    rng: random.Random,
    n_vars: int,
    m: int,
    low: int = -3,
    high: int = 3,
) -> List[EvalPoint]:
    """Sample m random integer evaluation points for Schwartz-Zippel identity testing.

    These points are fixed for the lifetime of an environment instance so that
    all nodes share a consistent evaluation basis.

    Args:
        rng:    Random source (seeded per env for reproducibility).
        n_vars: Number of polynomial variables.
        m:      Number of evaluation points to sample.
        low:    Minimum value for each coordinate (inclusive).
        high:   Maximum value for each coordinate (inclusive).

    Returns:
        List of m tuples, each of length n_vars.
    """
    points: List[EvalPoint] = []
    for _ in range(m):
        point = tuple(rng.randint(low, high) for _ in range(n_vars))
        points.append(point)
    return points


def eval_poly_points(poly: Poly, points: Iterable[EvalPoint]) -> List[Fraction]:
    """Evaluate a polynomial at each point in the list.

    Returns a list of exact rational values, one per point.
    This is the "eval vector" used as a polynomial fingerprint.
    """
    return [eval_poly(poly, p) for p in points]


def eval_distance(a: Sequence[Fraction], b: Sequence[Fraction]) -> Fraction:
    """Compute L1 distance between two evaluation vectors (sum of |a_i - b_i|).

    A distance of 0 means the two polynomials agree at all sampled points
    (i.e., they are likely identical under Schwartz-Zippel).
    """
    if len(a) != len(b):
        raise ValueError("Evaluation vectors must be same length")
    total = Fraction(0)
    for x, y in zip(a, b):
        total += abs(x - y)
    return total
