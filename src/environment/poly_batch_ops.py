"""Batched polynomial arithmetic over F_p[x_0, ..., x_{n-1}].

Used by the on-path cache builder to compute all upper-triangular pair
add / mul products per BFS layer in one vectorized op, replacing the
per-pair Python loop in :func:`src.game_board.generator.build_game_board`.

Coefficient layout matches :class:`src.environment.fast_polynomial.FastPoly`:
``int64`` arrays of shape ``(max_degree + 1,) ** n_vars``, then flattened to a
``(N, target_size)`` row layout for indexed batched access. Outputs are
``int64 mod p`` so ``row.tobytes()`` produces the same canonical key as
``FastPoly.canonical_key`` and the existing dict dedup keeps working.
"""

from __future__ import annotations

import itertools
import warnings
from typing import Callable, Literal, Optional

import numpy as np


class PolyBatchOps:
    """All-pairs polynomial add and (truncated) multiply over F_p.

    Args:
        n_vars: number of polynomial variables.
        max_degree: maximum per-variable degree retained (truncation bound).
        mod: prime modulus.
        backend: ``"numpy"`` (default) or ``"jax"``. JAX uses the first
            available JAX GPU device; if JAX or a GPU is unavailable, it warns
            and falls back to NumPy.
    """

    def __init__(
        self,
        n_vars: int,
        max_degree: int,
        mod: int,
        backend: Literal["numpy", "jax"] = "numpy",
    ):
        if backend not in ("numpy", "jax"):
            raise ValueError(f"unknown backend: {backend!r}")
        self.n_vars = int(n_vars)
        self.max_degree = int(max_degree)
        self.mod = int(mod)
        self._D = self.max_degree + 1
        self.grid_shape = (self._D,) * self.n_vars
        self.target_size = int(np.prod(self.grid_shape))
        self._backend = "numpy"
        self._mul_offsets: Optional[tuple[tuple[int, ...], ...]] = None
        self._jax = None
        self._jnp = None
        self._jax_device = None
        self._jax_add: Optional[Callable] = None
        self._jax_mul: Optional[Callable] = None

        if backend == "jax":
            self._try_enable_jax()

    @property
    def backend(self) -> str:
        return self._backend

    def _try_enable_jax(self) -> None:
        max_accumulator = self.mod * self.target_size * self.mod
        if max_accumulator >= 2**31:
            raise ValueError(
                "JAX int32 polynomial multiply may overflow for "
                f"mod={self.mod}, target_size={self.target_size}"
            )

        try:
            import jax
            import jax.numpy as jnp
            from jax import lax
        except Exception as exc:  # pragma: no cover - depends on local install
            warnings.warn(
                f"JAX backend requested but JAX is unavailable ({exc}); "
                "falling back to NumPy.",
                RuntimeWarning,
                stacklevel=2,
            )
            return

        try:
            gpu_devices = [
                d
                for d in jax.devices()
                if d.platform.lower() in ("gpu", "cuda", "rocm", "metal")
            ]
        except Exception:
            gpu_devices = []
        if not gpu_devices:
            warnings.warn(
                "JAX backend requested but no JAX GPU device is available; "
                "falling back to NumPy.",
                RuntimeWarning,
                stacklevel=2,
            )
            return

        D = self._D
        n = self.n_vars
        mod = self.mod
        target_size = self.target_size
        grid_shape = self.grid_shape
        padding = tuple((D - 1, D - 1) for _ in range(n))
        strides = (1,) * n
        dimension_numbers = (
            (0, 1, tuple(range(2, 2 + n))),
            (0, 1, tuple(range(2, 2 + n))),
            (0, 1, tuple(range(2, 2 + n))),
        )

        @jax.jit
        def add_jit(left, right):
            return (left + right) % mod

        @jax.jit
        def mul_jit(left_flat, right_flat):
            pair_count = left_flat.shape[0]
            left = left_flat.reshape((1, pair_count) + grid_shape)
            right = right_flat.reshape((pair_count, 1) + grid_shape)
            # lax.conv is cross-correlation; flipping the per-pair kernels gives
            # polynomial convolution. Grouping keeps each pair independent.
            right = jnp.flip(right, axis=tuple(range(2, 2 + n)))
            conv = lax.conv_general_dilated(
                left,
                right,
                window_strides=strides,
                padding=padding,
                dimension_numbers=dimension_numbers,
                feature_group_count=pair_count,
            )
            truncated = conv[(0, slice(None)) + tuple(slice(0, D) for _ in range(n))]
            return truncated.reshape((pair_count, target_size)) % mod

        self._jax = jax
        self._jnp = jnp
        self._jax_device = gpu_devices[0]
        self._jax_add = add_jit
        self._jax_mul = mul_jit
        self._backend = "jax"

    def add_all_pairs(self, coeffs: np.ndarray, pair_idx: np.ndarray) -> np.ndarray:
        """Vectorized ``(left + right) % mod``.

        Args:
            coeffs: ``(N, target_size)`` int64.
            pair_idx: ``(P, 2)`` int (left, right) node indices.

        Returns:
            ``(P, target_size)`` int64 in ``[0, mod)``.
        """
        left = coeffs[pair_idx[:, 0]]
        right = coeffs[pair_idx[:, 1]]
        if self._backend == "jax":
            assert self._jax is not None
            assert self._jnp is not None
            assert self._jax_add is not None
            left_jax = self._jax.device_put(
                self._jnp.asarray(left, dtype=self._jnp.int32),
                self._jax_device,
            )
            right_jax = self._jax.device_put(
                self._jnp.asarray(right, dtype=self._jnp.int32),
                self._jax_device,
            )
            return np.asarray(self._jax_add(left_jax, right_jax), dtype=np.int64)
        return (left + right) % self.mod

    def mul_all_pairs(self, coeffs: np.ndarray, pair_idx: np.ndarray) -> np.ndarray:
        """Vectorized truncated polynomial multiply mod p.

        Implements ``out[c] = sum_{a + b = c} left[a] * right[b]`` for each
        multi-index ``c`` with all components ``< max_degree + 1`` (truncation).
        The Python loop runs over the ``D ** n_vars`` monomial positions of
        ``left`` (≤ 49 for C6 / n=2); per-pair work is fully vectorized in
        NumPy.

        Args:
            coeffs: ``(N, target_size)`` int64.
            pair_idx: ``(P, 2)`` int.

        Returns:
            ``(P, target_size)`` int64 in ``[0, mod)``.
        """
        P = pair_idx.shape[0]
        D = self._D
        n = self.n_vars
        nd_shape = (-1,) + self.grid_shape
        coeffs_nd = coeffs.reshape(nd_shape)
        left = coeffs_nd[pair_idx[:, 0]]
        right = coeffs_nd[pair_idx[:, 1]]
        if self._backend == "jax":
            assert self._jax is not None
            assert self._jnp is not None
            assert self._jax_mul is not None
            left_jax = self._jax.device_put(
                self._jnp.asarray(
                    left.reshape(P, self.target_size),
                    dtype=self._jnp.int32,
                ),
                self._jax_device,
            )
            right_jax = self._jax.device_put(
                self._jnp.asarray(
                    right.reshape(P, self.target_size),
                    dtype=self._jnp.int32,
                ),
                self._jax_device,
            )
            return np.asarray(self._jax_mul(left_jax, right_jax), dtype=np.int64)

        result = np.zeros((P,) + self.grid_shape, dtype=np.int64)

        if self._mul_offsets is None:
            self._mul_offsets = tuple(
                tuple(a) for a in itertools.product(*[range(D) for _ in range(n)])
            )

        broadcast_shape = (P,) + (1,) * n
        sel_all = (slice(None),)
        for a in self._mul_offsets:
            slices_in = tuple(slice(0, D - ai) for ai in a)
            slices_out = tuple(slice(ai, D) for ai in a)
            coef = left[sel_all + a].reshape(broadcast_shape)
            result[sel_all + slices_out] += coef * right[sel_all + slices_in]

        return (result.reshape(P, self.target_size)) % self.mod
