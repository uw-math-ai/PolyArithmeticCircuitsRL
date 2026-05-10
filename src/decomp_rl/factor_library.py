"""Factorizable polynomial library with JAX-accelerated subset matching.

The library stores polynomials confirmed to have non-trivial multiplicative
factorizations (savings ≥ 1 op over direct construction).  At query time it
finds any library entry that appears as a *term-subset* of the target, up to:
  1. Exact inclusion
  2. F_p scalar scaling
  3. Variable permutation (gated by config; only when n ≤ max_perm_vars)

JAX is used for the exact and scalar matching (batched over all N library
entries at once).  Permutation matching falls back to NumPy loops because the
permutation space is small when enabled.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

from .canonical import modular_inverse
from .cost_model import rebuild_cost
from .factor_fp import FactorizationResult
from .polynomial import SparsePolynomial

try:
    import jax
    import jax.numpy as jnp

    _JAX_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    _JAX_AVAILABLE = False


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LibraryEntry:
    """A polynomial with a confirmed non-trivial factorization."""

    poly: SparsePolynomial
    factorization: FactorizationResult
    rebuild_ops: int
    savings: int


@dataclass(frozen=True)
class LibraryMatch:
    """A library entry found as a scaled/permuted sub-polynomial of a target."""

    entry: LibraryEntry
    matched_poly: SparsePolynomial
    scale: int
    var_perm: tuple[int, ...]
    match_type: str  # "exact" | "scalar" | "permuted" | "scalar_permuted"


# ---------------------------------------------------------------------------
# Main library class
# ---------------------------------------------------------------------------


class FactorizableLibrary:
    """Top-down factorizable polynomial library.

    Usage pattern
    -------------
    1. Create one library per (prime, variables) configuration.
    2. Call ``seed_known_families()`` once to pre-populate with known patterns.
    3. During search pass the library to ``FiniteFieldFactorizer`` so every
       discovered non-trivial factorization is automatically added.
    4. Pass the library to ``propose_splits`` to generate "library_match"
       split candidates.
    5. Check ``is_known(poly)`` in ``DecompEnv.step`` to emit the library
       reward signal.
    """

    def __init__(
        self,
        prime: int,
        variables: tuple[str, ...],
        max_degree: int = 6,
        max_entries: int = 1024,
        match_scalar: bool = True,
        match_permutations: bool = True,
        max_perm_vars: int = 5,
        library_match_score_bonus: float = 2.0,
        library_step_reward: float = 0.5,
    ) -> None:
        self.prime = prime
        self.variables = variables
        self.max_degree = max_degree
        self.max_entries = max_entries
        self.match_scalar = match_scalar
        self.match_permutations = match_permutations
        self.max_perm_vars = max_perm_vars
        self.library_match_score_bonus = library_match_score_bonus
        self.library_step_reward = library_step_reward

        self._entries: dict[str, LibraryEntry] = {}

        # JAX state — rebuilt lazily whenever _dirty is True
        self._dirty = True
        self._monomial_list: list[tuple[int, ...]] = []
        self._monomial_index: dict[tuple[int, ...], int] = {}
        self._jax_lib: "jax.Array | None" = None  # shape (N, M) int32
        self._jax_entry_keys: list[str] = []

        self._build_monomial_index()

    # ------------------------------------------------------------------
    # Monomial index (fixed at construction, based on max_degree)
    # ------------------------------------------------------------------

    def _build_monomial_index(self) -> None:
        """Enumerate all monomials up to total degree max_degree."""
        n = len(self.variables)
        monomials: list[tuple[int, ...]] = []
        for exponent in itertools.product(range(self.max_degree + 1), repeat=n):
            if sum(exponent) <= self.max_degree:
                monomials.append(exponent)
        self._monomial_list = monomials
        self._monomial_index = {exp: i for i, exp in enumerate(monomials)}

    def _poly_to_dense(self, poly: SparsePolynomial) -> np.ndarray:
        """Return a 1-D int32 array of length M (monomial count)."""
        dense = np.zeros(len(self._monomial_index), dtype=np.int32)
        for coeff, exponent in poly.terms:
            idx = self._monomial_index.get(exponent)
            if idx is not None:
                dense[idx] = int(coeff) % self.prime
        return dense

    # ------------------------------------------------------------------
    # Library management
    # ------------------------------------------------------------------

    def add(
        self,
        poly: SparsePolynomial,
        factorization: FactorizationResult,
        rebuild_ops: int,
        savings: int,
    ) -> bool:
        """Add a confirmed factorizable poly.

        Returns True only if the entry was newly inserted.  Rejects:
        - entries with savings ≤ 0
        - entries with degree > max_degree
        - entries whose factorization has no non-constant factors (i.e., unit only)
        - duplicates
        - when the library is already full
        """
        if savings <= 0:
            return False
        if poly.total_degree > self.max_degree:
            return False
        has_poly_factor = any(not f.is_constant for f, _ in factorization.factors)
        if not has_poly_factor:
            return False
        if len(self._entries) >= self.max_entries:
            return False
        key = poly.to_key()
        if key in self._entries:
            return False
        self._entries[key] = LibraryEntry(
            poly=poly,
            factorization=factorization,
            rebuild_ops=rebuild_ops,
            savings=savings,
        )
        self._dirty = True
        return True

    def maybe_add_from_factorization(
        self,
        poly: SparsePolynomial,
        factorization: FactorizationResult,
        direct_cost: int,
    ) -> bool:
        """Convenience wrapper — compute savings and call add()."""
        rb = rebuild_cost(factorization)
        savings = direct_cost - rb
        return self.add(poly, factorization, rb, savings)

    def is_known(self, poly: SparsePolynomial) -> bool:
        """True if an exact canonical key for poly is in the library."""
        return poly.to_key() in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    # ------------------------------------------------------------------
    # JAX state management
    # ------------------------------------------------------------------

    def _rebuild_jax_state(self) -> None:
        self._dirty = False
        if not self._entries:
            self._jax_lib = None
            self._jax_entry_keys = []
            return
        keys = list(self._entries.keys())
        rows = np.stack([self._poly_to_dense(self._entries[k].poly) for k in keys])
        self._jax_entry_keys = keys
        if _JAX_AVAILABLE:
            self._jax_lib = jnp.array(rows)
        else:
            self._jax_lib = rows  # type: ignore[assignment]

    def _ensure_state(self) -> None:
        if self._dirty:
            self._rebuild_jax_state()

    # ------------------------------------------------------------------
    # Core matching API
    # ------------------------------------------------------------------

    def find_matches(self, target: SparsePolynomial) -> list[LibraryMatch]:
        """Return all library entries that appear as a sub-polynomial of target.

        Checks exact, scalar, and (when enabled) variable-permutation matches.
        Results are deduplicated by the key of the matched_poly.
        """
        if not self._entries:
            return []
        self._ensure_state()

        t_dense = self._poly_to_dense(target)

        if _JAX_AVAILABLE and self._jax_lib is not None:
            results = self._jax_find_matches(target, t_dense)
        else:
            results = self._numpy_find_matches(target, t_dense)

        # deduplicate by matched_poly key
        seen: set[str] = set()
        deduped: list[LibraryMatch] = []
        for m in results:
            k = m.matched_poly.to_key()
            if k not in seen:
                seen.add(k)
                deduped.append(m)
        return deduped

    # ------------------------------------------------------------------
    # JAX-backed exact + scalar matching
    # ------------------------------------------------------------------

    def _jax_find_matches(
        self,
        target: SparsePolynomial,
        t_dense: np.ndarray,
    ) -> list[LibraryMatch]:
        t = jnp.array(t_dense, dtype=jnp.int32)
        lib = self._jax_lib  # (N, M)

        # support_mask[i, m] = True iff library entry i uses monomial m
        support_mask = lib > 0  # (N, M)

        # Exact subset: all support terms of row i match target
        exact_mask = jnp.all((lib == t[None, :]) | ~support_mask, axis=1)
        has_support = jnp.any(support_mask, axis=1)
        exact_hits = np.asarray(exact_mask & has_support, dtype=bool)

        results: list[LibraryMatch] = []
        identity = tuple(range(len(self.variables)))

        # --- Exact matches ---
        for i in np.where(exact_hits)[0]:
            entry = self._entries[self._jax_entry_keys[i]]
            if entry.poly == target:
                continue
            results.append(
                LibraryMatch(
                    entry=entry,
                    matched_poly=entry.poly,
                    scale=1,
                    var_perm=identity,
                    match_type="exact",
                )
            )

        # --- Scalar matches ---
        if self.match_scalar:
            lib_np = np.asarray(lib, dtype=np.int32)
            results.extend(
                self._scalar_matches(
                    target, t_dense, lib_np, exact_hits, identity
                )
            )

        # --- Permutation matches ---
        if self.match_permutations and len(self.variables) <= self.max_perm_vars:
            results.extend(self._permutation_matches(target, t_dense))

        return results

    # ------------------------------------------------------------------
    # NumPy fallback
    # ------------------------------------------------------------------

    def _numpy_find_matches(
        self,
        target: SparsePolynomial,
        t_dense: np.ndarray,
    ) -> list[LibraryMatch]:
        if self._jax_lib is None:
            return []
        lib_np = np.asarray(self._jax_lib, dtype=np.int32)
        support_mask = lib_np > 0
        exact_mask = np.all((lib_np == t_dense[None, :]) | ~support_mask, axis=1)
        has_support = np.any(support_mask, axis=1)
        exact_hits = exact_mask & has_support

        results: list[LibraryMatch] = []
        identity = tuple(range(len(self.variables)))

        for i in np.where(exact_hits)[0]:
            entry = self._entries[self._jax_entry_keys[i]]
            if entry.poly == target:
                continue
            results.append(
                LibraryMatch(
                    entry=entry,
                    matched_poly=entry.poly,
                    scale=1,
                    var_perm=identity,
                    match_type="exact",
                )
            )

        if self.match_scalar:
            results.extend(
                self._scalar_matches(target, t_dense, lib_np, exact_hits, identity)
            )
        if self.match_permutations and len(self.variables) <= self.max_perm_vars:
            results.extend(self._permutation_matches(target, t_dense))

        return results

    # ------------------------------------------------------------------
    # Scalar matching helper
    # ------------------------------------------------------------------

    def _scalar_matches(
        self,
        target: SparsePolynomial,
        t_dense: np.ndarray,
        lib_np: np.ndarray,
        already_exact: np.ndarray,
        identity: tuple[int, ...],
    ) -> list[LibraryMatch]:
        p = self.prime
        results: list[LibraryMatch] = []
        N = lib_np.shape[0]

        for i in range(N):
            if already_exact[i]:
                continue
            row = lib_np[i]
            nonzero_cols = np.flatnonzero(row)
            if len(nonzero_cols) == 0:
                continue

            # Infer the unique scalar c from the first support monomial
            m0 = nonzero_cols[0]
            t_val = int(t_dense[m0])
            if t_val == 0:
                continue  # that monomial absent from target → no match
            c = (t_val * modular_inverse(int(row[m0]), p)) % p
            if c <= 1:
                continue  # c=0 invalid; c=1 already handled as exact

            # Verify all support monomials satisfy the same scalar
            scaled = (c * row) % p
            if np.all((scaled == t_dense) | (row == 0)):
                entry = self._entries[self._jax_entry_keys[i]]
                scaled_poly = entry.poly.scale(c)
                if scaled_poly == target:
                    continue
                results.append(
                    LibraryMatch(
                        entry=entry,
                        matched_poly=scaled_poly,
                        scale=c,
                        var_perm=identity,
                        match_type="scalar",
                    )
                )
        return results

    # ------------------------------------------------------------------
    # Permutation matching helper
    # ------------------------------------------------------------------

    def _permutation_matches(
        self,
        target: SparsePolynomial,
        t_dense: np.ndarray,
    ) -> list[LibraryMatch]:
        """Try all variable permutations of each library entry.

        new_exp[j] = old_exp[perm[j]]: position j in the new poly uses the
        degree that position perm[j] had in the original.  Only non-identity
        permutations are tried; exact/scalar covers identity.
        """
        p = self.prime
        n = len(self.variables)
        identity = tuple(range(n))
        results: list[LibraryMatch] = []

        for key, entry in self._entries.items():
            for perm in itertools.permutations(range(n)):
                if perm == identity:
                    continue

                permuted_terms = [
                    (coeff, tuple(exponent[perm[j]] for j in range(n)))
                    for coeff, exponent in entry.poly.terms
                ]
                try:
                    permuted = SparsePolynomial.from_terms(
                        permuted_terms, p, self.variables
                    )
                except Exception:
                    continue

                pd = self._poly_to_dense(permuted)
                support = pd > 0

                # Exact permuted subset
                if np.any(support) and np.all((pd == t_dense) | ~support):
                    if permuted != target:
                        results.append(
                            LibraryMatch(
                                entry=entry,
                                matched_poly=permuted,
                                scale=1,
                                var_perm=perm,
                                match_type="permuted",
                            )
                        )
                    continue

                # Scalar permuted subset
                if not self.match_scalar:
                    continue
                nonzero = np.flatnonzero(pd)
                if len(nonzero) == 0:
                    continue
                m0 = nonzero[0]
                t_val = int(t_dense[m0])
                if t_val == 0:
                    continue
                c = (t_val * modular_inverse(int(pd[m0]), p)) % p
                if c <= 1:
                    continue
                scaled = (c * pd) % p
                if np.all((scaled == t_dense) | (pd == 0)):
                    scaled_poly = permuted.scale(c)
                    if scaled_poly != target:
                        results.append(
                            LibraryMatch(
                                entry=entry,
                                matched_poly=scaled_poly,
                                scale=c,
                                var_perm=perm,
                                match_type="scalar_permuted",
                            )
                        )

        return results

    # ------------------------------------------------------------------
    # Pre-seeding with known factorizable families
    # ------------------------------------------------------------------

    def seed_known_families(self, factorizer=None) -> int:
        """Pre-populate library with known-factorizable polynomial patterns.

        Generates:
          - (x_i + a)^2 for each variable and each nonzero constant a ∈ F_p
          - (x_i + a)(x_j + b) for each variable pair and a, b ∈ F_p
          - Elementary symmetric polynomials of degree 2..min(n, 3)
          - Simple products x_i * x_j

        Returns the count of newly added entries.
        """
        from .baseline_cost import BaselineCostModel
        from .factor_fp import FiniteFieldFactorizer

        own_factorizer = factorizer is None
        factorizer = factorizer or FiniteFieldFactorizer()
        baseline = BaselineCostModel()
        added = 0
        p = self.prime
        vars_ = self.variables
        n = len(vars_)
        zero_exp = (0,) * n

        def _const(a: int) -> SparsePolynomial:
            return SparsePolynomial.from_monomial(a % p, zero_exp, p, vars_)

        def _var(vi: int) -> SparsePolynomial:
            return SparsePolynomial.variable(vars_[vi], p, vars_)

        def try_add(poly: SparsePolynomial) -> None:
            nonlocal added
            if poly.is_zero or poly.is_constant or poly.is_monomial:
                return
            if poly.support_size < 2:
                return
            try:
                fact = factorizer.factor(poly)
            except Exception:
                return
            has_poly_factor = any(not f.is_constant for f, _ in fact.factors)
            if not has_poly_factor:
                return
            direct = baseline.direct_construction_cost(poly)
            rb = rebuild_cost(fact)
            if self.add(poly, fact, rb, direct - rb):
                added += 1

        # (x_i + a)^2 for each variable, each nonzero a
        for vi in range(n):
            xi = _var(vi)
            for a in range(1, p):
                try_add((xi + _const(a)) * (xi + _const(a)))

        # x_i * x_j — monomials; skip since monomial_build_cost handles them,
        # but (x_i + a)(x_j + b) with constants can have savings
        for vi in range(n):
            for vj in range(vi + 1, n):
                xi, xj = _var(vi), _var(vj)
                for a in range(p):
                    for b in range(p):
                        try_add((xi + _const(a)) * (xj + _const(b)))

        # Elementary symmetric polynomials e_k(x0, ..., x_{n-1})
        from itertools import combinations as _combs

        for degree in range(2, min(n + 1, 4)):
            result = SparsePolynomial.zero(p, vars_)
            for combo in _combs(range(n), degree):
                exp = [0] * n
                for idx in combo:
                    exp[idx] = 1
                result = result + SparsePolynomial.from_monomial(
                    1, tuple(exp), p, vars_
                )
            try_add(result)

        if own_factorizer:
            factorizer.close()

        return added
