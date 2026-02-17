"""Fast polynomial arithmetic using dense numpy coefficient arrays.

Replaces SymPy for all hot-path polynomial operations.
Polynomials over F_p[x0, ..., x_{n-1}] are stored as n-dimensional numpy arrays
of shape (max_degree+1,)^n, where entry [a0, a1, ...] is the coefficient of
x0^a0 * x1^a1 * ... reduced modulo p.

This gives 10-100x speedup over SymPy for the operations used in environment
stepping, MCTS simulation, and BFS game board generation.
"""

import numpy as np
from typing import Optional


class FastPoly:
    """Polynomial with dense coefficient array and mod-p arithmetic.

    Attributes:
        coeffs: numpy int64 array of shape (max_degree+1,)^n_vars
        mod: prime modulus
    """

    __slots__ = ('coeffs', 'mod')

    def __init__(self, coeffs: np.ndarray, mod: int):
        self.coeffs = coeffs.astype(np.int64) % mod
        self.mod = mod

    # ---- Constructors ----

    @classmethod
    def zero(cls, n_vars: int, max_degree: int, mod: int) -> "FastPoly":
        shape = (max_degree + 1,) * n_vars
        return cls(np.zeros(shape, dtype=np.int64), mod)

    @classmethod
    def constant(cls, value: int, n_vars: int, max_degree: int, mod: int) -> "FastPoly":
        p = cls.zero(n_vars, max_degree, mod)
        idx = (0,) * n_vars
        p.coeffs[idx] = value % mod
        return p

    @classmethod
    def variable(cls, var_idx: int, n_vars: int, max_degree: int, mod: int) -> "FastPoly":
        """Create polynomial for a single variable x_{var_idx}."""
        p = cls.zero(n_vars, max_degree, mod)
        idx = tuple(1 if i == var_idx else 0 for i in range(n_vars))
        p.coeffs[idx] = 1
        return p

    # ---- Arithmetic ----

    def __add__(self, other: "FastPoly") -> "FastPoly":
        return FastPoly((self.coeffs + other.coeffs), self.mod)

    def __mul__(self, other: "FastPoly") -> "FastPoly":
        result = _nd_convolve(self.coeffs, other.coeffs)
        # Truncate to original shape (higher-degree terms are lost)
        slices = tuple(slice(0, s) for s in self.coeffs.shape)
        truncated = result[slices] % self.mod
        return FastPoly(truncated, self.mod)

    # ---- Comparison / Hashing ----

    def __eq__(self, other) -> bool:
        if not isinstance(other, FastPoly):
            return NotImplemented
        return np.array_equal(self.coeffs, other.coeffs)

    def __hash__(self):
        return hash(self.coeffs.tobytes())

    def __repr__(self):
        nonzero = list(zip(np.argwhere(self.coeffs != 0).tolist(),
                           self.coeffs[self.coeffs != 0].tolist()))
        if not nonzero:
            return "FastPoly(0)"
        terms = []
        for idx, c in nonzero:
            parts = [f"x{i}^{e}" for i, e in enumerate(idx) if e > 0]
            monomial = "*".join(parts) if parts else "1"
            terms.append(f"{c}*{monomial}" if parts else str(c))
        return f"FastPoly({' + '.join(terms)})"

    # ---- Utility ----

    def canonical_key(self) -> bytes:
        """Unique hashable key for deduplication. O(1) via buffer view."""
        return self.coeffs.tobytes()

    def is_zero(self) -> bool:
        return not np.any(self.coeffs)

    def to_vector(self) -> np.ndarray:
        """Flatten coefficients to 1D vector (for target encoding)."""
        return self.coeffs.flatten().astype(np.float64)

    def term_similarity(self, target: "FastPoly") -> float:
        """Fraction of matching nonzero target coefficients."""
        target_nonzero = target.coeffs != 0
        if not np.any(target_nonzero):
            return 1.0 if self.is_zero() else 0.0
        total = int(np.count_nonzero(target_nonzero))
        matching = int(np.sum((self.coeffs == target.coeffs) & target_nonzero))
        return matching / total

    def copy(self) -> "FastPoly":
        """Return a copy (numpy array is mutable, unlike SymPy exprs)."""
        return FastPoly(self.coeffs.copy(), self.mod)

    @property
    def n_vars(self) -> int:
        return self.coeffs.ndim

    @property
    def max_degree(self) -> int:
        return self.coeffs.shape[0] - 1


# ---- N-dimensional convolution (polynomial multiplication) ----

def _nd_convolve(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """N-dimensional polynomial multiplication via convolution.

    For small arrays (our typical case: 7x7), direct loops are faster than FFT.
    """
    ndim = a.ndim
    assert ndim == b.ndim

    if ndim == 1:
        return np.convolve(a, b)

    if ndim == 2:
        return _convolve_2d(a, b)

    # General n-d case
    return _convolve_nd(a, b)


def _convolve_2d(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Direct 2D convolution optimized for small arrays."""
    sa0, sa1 = a.shape
    sb0, sb1 = b.shape
    result = np.zeros((sa0 + sb0 - 1, sa1 + sb1 - 1), dtype=np.int64)
    for i in range(sa0):
        for j in range(sa1):
            if a[i, j] != 0:
                result[i:i + sb0, j:j + sb1] += int(a[i, j]) * b
    return result


def _convolve_nd(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """General n-dimensional convolution via sparse iteration."""
    result_shape = tuple(sa + sb - 1 for sa, sb in zip(a.shape, b.shape))
    result = np.zeros(result_shape, dtype=np.int64)
    for idx in zip(*np.nonzero(a)):
        slices = tuple(slice(i, i + sb) for i, sb in zip(idx, b.shape))
        result[slices] += int(a[idx]) * b
    return result
