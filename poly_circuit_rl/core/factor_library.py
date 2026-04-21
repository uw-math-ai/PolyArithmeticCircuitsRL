"""Session-level factor library for polynomial reuse across episodes.

Tracks which intermediate polynomials the agent has successfully built in past
episodes.  At episode reset, the target polynomial is factorized over Q using
SymPy; non-trivial factors become subgoals.  During stepping, matching a
subgoal earns a bonus reward; matching a library-known subgoal earns extra.

Dynamic subgoal discovery runs when a library-known node is built:
  - T - v_new is computed (cheap, no SymPy) and added as an additive subgoal.
  - T / v_new is attempted via exact polynomial division (SymPy).  If exact,
    the quotient is added as a multiplicative subgoal.

Completion bonuses fire when both pieces for one final op exist in the circuit.

Adapted from Repo 1's FactorLibrary for Repo 2's Poly = Dict[Exponent, Fraction].
All arithmetic is exact over Q (no modular reduction needed).
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from fractions import Fraction
from typing import Dict, List, Optional, Set

from .poly import (
    Poly,
    PolyKey,
    canonicalize,
    make_const,
    make_var,
    poly_hashkey,
)

log = logging.getLogger(__name__)


class _FrozenFactorLibrary:
    """Read-only wrapper around a mutable FactorLibrary."""

    def __init__(self, base: "FactorLibrary") -> None:
        self._base = base

    def register(self, poly: Poly, step_num: int = 1) -> None:
        _ = (poly, step_num)
        return None

    def register_episode_nodes(self, nodes, n_initial: int) -> None:
        _ = (nodes, n_initial)
        return None

    def frozen_view(self) -> "_FrozenFactorLibrary":
        return self

    def contains(self, poly: Poly) -> bool:
        return self._base.contains(poly)

    def is_base(self, poly: Poly) -> bool:
        return self._base.is_base(poly)

    def factorize_target(self, target: Poly) -> List[Poly]:
        return self._base.factorize_target(target)

    def factorize_poly(
        self,
        poly: Poly,
        exclude_keys: Optional[Set[PolyKey]] = None,
    ) -> List[Poly]:
        return self._base.factorize_poly(poly, exclude_keys=exclude_keys)

    def exact_quotient(self, dividend: Poly, divisor: Poly) -> Optional[Poly]:
        return self._base.exact_quotient(dividend, divisor)

    def filter_known(self, factor_polys: List[Poly]) -> Set[PolyKey]:
        return self._base.filter_known(factor_polys)

    def __len__(self) -> int:
        return len(self._base)

    def __repr__(self) -> str:
        return f"Frozen{self._base!r}"


class FactorLibrary:
    """In-session cache mapping polynomial canonical keys to construction costs.

    Attributes:
        n_vars: Number of polynomial variables.
    """

    def __init__(self, n_vars: int, max_size: Optional[int] = None) -> None:
        self.n_vars = n_vars
        self._max_size = max_size

        # Maps poly_hashkey -> minimum step index at first construction (1-indexed).
        self._known: "OrderedDict[PolyKey, int]" = OrderedDict()

        # Canonical keys for base nodes (x0, ..., x_{n-1}, const_1).
        self._base_keys: Set[PolyKey] = self._compute_base_keys()

        # SymPy symbols — lazily initialized.
        self._syms: Optional[list] = None

    def _compute_base_keys(self) -> Set[PolyKey]:
        keys: Set[PolyKey] = set()
        for i in range(self.n_vars):
            keys.add(poly_hashkey(make_var(self.n_vars, i)))
        keys.add(poly_hashkey(make_const(self.n_vars, 1)))
        return keys

    def _get_syms(self) -> list:
        if self._syms is None:
            from sympy import symbols as sympy_symbols

            if self.n_vars == 1:
                self._syms = [sympy_symbols("x")]
            else:
                self._syms = list(sympy_symbols(f"x0:{self.n_vars}"))
        return self._syms

    # ------------------------------------------------------------------
    # SymPy conversion helpers
    # ------------------------------------------------------------------

    def _poly_to_sympy(self, poly: Poly):
        """Convert internal Poly dict to a SymPy expression."""
        import sympy

        syms = self._get_syms()
        expr = sympy.Integer(0)
        for mono, coeff in canonicalize(poly).items():
            term = sympy.Rational(coeff)
            for v, e in zip(syms, mono):
                if e:
                    term = term * v ** e
            expr += term
        return expr

    def _sympy_to_poly(self, expr) -> Poly:
        """Convert a SymPy expression to internal Poly dict."""
        import sympy

        syms = self._get_syms()
        sp = sympy.Poly(expr, *syms)
        poly: Poly = {}
        for monom, coeff in sp.as_dict().items():
            poly[monom] = Fraction(coeff)
        return canonicalize(poly)

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def _evict_if_needed(self) -> None:
        if self._max_size is None:
            return
        while len(self._known) > self._max_size:
            self._known.popitem(last=False)

    def register(self, poly: Poly, step_num: int = 1) -> None:
        """Register a polynomial as known, recording its construction cost.

        Base nodes are silently ignored.  Keeps the minimum step_num.
        """
        key = poly_hashkey(poly)
        if key in self._base_keys:
            return
        existing = self._known.get(key)
        if existing is None or step_num < existing:
            self._known[key] = step_num
        self._known.move_to_end(key, last=True)
        self._evict_if_needed()

    def register_episode_nodes(self, nodes, n_initial: int) -> None:
        """Register all agent-built nodes from a successful episode.

        Args:
            nodes: Full ordered list of circuit Node objects.
            n_initial: Number of initial base nodes (= n_vars + 1).
        """
        for step_idx, node in enumerate(nodes[n_initial:], start=1):
            self.register(node.poly, step_num=step_idx)

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def is_base(self, poly: Poly) -> bool:
        """Return True if poly is a free base node (variable or constant 1)."""
        return poly_hashkey(poly) in self._base_keys

    def contains(self, poly: Poly) -> bool:
        """Check if a polynomial has been registered in the library."""
        return poly_hashkey(poly) in self._known

    def filter_known(self, factor_polys: List[Poly]) -> Set[PolyKey]:
        """Return hashkeys from factor_polys that are already in the library."""
        return {
            poly_hashkey(p)
            for p in factor_polys
            if poly_hashkey(p) in self._known
        }

    # ------------------------------------------------------------------
    # Factorization
    # ------------------------------------------------------------------

    def factorize_target(self, target: Poly) -> List[Poly]:
        """Factorize the target and return non-trivial factors.

        Excludes the target itself from results.
        """
        return self.factorize_poly(target, exclude_keys={poly_hashkey(target)})

    def factorize_poly(
        self,
        poly: Poly,
        exclude_keys: Optional[Set[PolyKey]] = None,
    ) -> List[Poly]:
        """Factorize poly over Q and return filtered non-trivial factors.

        A factor is included if it:
          - Has total degree >= 1 (not a scalar constant)
          - Is not a base node (x_i or 1)
          - Is not in exclude_keys
          - Is not zero after canonicalization
          - Has not already appeared in the result (deduped)
        """
        import sympy

        if exclude_keys is None:
            exclude_keys = set()

        if not poly:
            return []

        try:
            expr = self._poly_to_sympy(poly)
        except Exception as e:
            log.debug("sympy fallback in %s.factorize_poly(_poly_to_sympy): %s", __name__, e)
            return []

        if expr == sympy.Integer(0) or expr.is_number:
            return []

        poly_key = poly_hashkey(poly)
        syms = self._get_syms()

        try:
            _content, factors = sympy.factor_list(expr, *syms)
        except Exception as e:
            log.debug("sympy fallback in %s.factorize_poly(factor_list): %s", __name__, e)
            return []

        result: List[Poly] = []
        seen_keys: Set[PolyKey] = set()

        for factor_expr, _mult in factors:
            try:
                if factor_expr.is_number:
                    continue
                fpoly = sympy.Poly(factor_expr, *syms)
                if fpoly.total_degree() == 0:
                    continue
            except Exception as e:
                log.debug("sympy fallback in %s.factorize_poly(total_degree): %s", __name__, e)
                continue

            try:
                factor_poly = self._sympy_to_poly(factor_expr)
            except Exception as e:
                log.debug("sympy fallback in %s.factorize_poly(sympy_to_poly): %s", __name__, e)
                continue

            if not factor_poly:
                continue

            key = poly_hashkey(factor_poly)

            if key == poly_key:
                continue
            if key in self._base_keys:
                continue
            if key in exclude_keys:
                continue
            if key in seen_keys:
                continue
            seen_keys.add(key)

            result.append(factor_poly)

        return result

    def exact_quotient(
        self, dividend: Poly, divisor: Poly,
    ) -> Optional[Poly]:
        """Return exact polynomial quotient dividend / divisor over Q, or None.

        Returns None if divisor does not divide dividend exactly.
        """
        import sympy

        if not divisor:
            return None

        syms = self._get_syms()

        try:
            dividend_expr = self._poly_to_sympy(dividend)
            divisor_expr = self._poly_to_sympy(divisor)
        except Exception as e:
            log.debug("sympy fallback in %s.exact_quotient(poly_to_sympy): %s", __name__, e)
            return None

        if divisor_expr == sympy.Integer(0):
            return None

        try:
            quotient_expr, remainder_expr = sympy.div(
                dividend_expr, divisor_expr, *syms, domain="QQ",
            )
        except Exception as e:
            log.debug("sympy fallback in %s.exact_quotient(div): %s", __name__, e)
            return None

        try:
            remainder_poly = self._sympy_to_poly(remainder_expr)
        except Exception as e:
            log.debug("sympy fallback in %s.exact_quotient(remainder): %s", __name__, e)
            return None

        if remainder_poly:
            return None  # Non-zero remainder — not exact.

        try:
            quotient_poly = self._sympy_to_poly(quotient_expr)
        except Exception as e:
            log.debug("sympy fallback in %s.exact_quotient(quotient): %s", __name__, e)
            return None

        if not quotient_poly:
            return None

        return quotient_poly

    def snapshot(self) -> Dict[PolyKey, int]:
        """Return a copy of the mutable library state."""
        return OrderedDict(self._known)

    def restore(self, known: Dict[PolyKey, int]) -> None:
        """Restore mutable library state from a snapshot."""
        self._known = OrderedDict(known)
        self._evict_if_needed()

    def frozen_view(self) -> _FrozenFactorLibrary:
        """Return a read-only view sharing this library's known-factor state."""
        return _FrozenFactorLibrary(self)

    def __len__(self) -> int:
        return len(self._known)

    def __repr__(self) -> str:
        return f"FactorLibrary(size={len(self)}, n_vars={self.n_vars})"
