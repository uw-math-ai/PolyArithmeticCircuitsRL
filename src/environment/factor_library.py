"""Session-level factor library for polynomial reuse across episodes.

This module implements the FactorLibrary — an in-memory cache that tracks
which intermediate polynomials the agent has successfully built in past
episodes of the current training run.

Motivation
----------
Many target polynomials share common sub-expressions (factors). For example,
(x0 + 1)^2 and (x0 + 1)^3 both require building (x0 + 1) as an intermediate
node. If the agent has already learned to build (x0 + 1) in a previous episode,
it should be guided and rewarded for recognizing and reusing this sub-computation
when it appears as a factor of the current target.

Mechanism
---------
1. At episode reset, the target polynomial is factorized over Z using SymPy's
   factor_list(). Non-trivial polynomial factors (degree >= 1, not a base input
   node) become subgoals for the episode.

2. During stepping, if the agent constructs a node v_new matching a subgoal, it
   receives factor_subgoal_reward. If that factor was also previously built in
   a past successful episode, it additionally receives factor_library_bonus.

   When v_new is library-known, dynamic subgoal discovery also runs:
   - T - v_new is computed (fast, no SymPy) and added as a direct additive
     subgoal (one addition away from T). Its factors are also discovered.
   - T / v_new is attempted via exact polynomial division (SymPy). If exact,
     the quotient is added as a multiplicative subgoal.

3. Completion bonuses fire when the circuit contains both pieces for one final op:
   - Additive: T - v_new already in the circuit → +completion_bonus.
   - Multiplicative: T / v_new is in the circuit or is a scalar → +completion_bonus.

4. After a successful episode, every agent-built node is registered in the library
   so future episodes can benefit from it.

Design Constraints
------------------
- No disk I/O: the library lives entirely in RAM for one training session.
- Fast lookups: all membership tests are O(1) set/dict operations on canonical keys.
- SymPy calls for factorization are gated on library membership to control cost.
- Factorization is over Z (integers), not F_p, for numerical stability. Factors are
  then reduced mod p when converted to FastPoly. Exact division is checked by
  reducing the ZZ remainder mod p (handles cases where ZZ remainder is a multiple
  of p but true F_p remainder is 0).
"""

from typing import Dict, List, Optional, Set

from .fast_polynomial import FastPoly


class FactorLibrary:
    """In-session cache mapping polynomial canonical keys to construction costs.

    The library serves two roles:

    1. **Subgoal generation**: Given a target polynomial, factorize it over Z
       and return the non-trivial polynomial factors as FastPoly subgoals for
       the current episode. The agent is rewarded when it builds these factors
       as intermediate nodes.

    2. **Reuse bonus**: Track which polynomials have been successfully built in
       previous episodes. When a factor subgoal is also in this cache, the agent
       earns an additional reward to encourage rediscovery of previously learned
       sub-computations.

    Attributes:
        mod (int): Prime modulus for polynomial arithmetic (e.g., 5).
        n_vars (int): Number of polynomial variables (e.g., 2 for F_p[x0, x1]).
        max_degree (int): Maximum degree per variable in the dense representation.
    """

    def __init__(self, mod: int, n_vars: int, max_degree: int) -> None:
        """Initialize an empty factor library.

        Args:
            mod: Prime modulus for coefficient arithmetic (e.g., 5). Must be prime.
            n_vars: Number of variables in the polynomial ring (e.g., 2 for x0, x1).
            max_degree: Maximum degree per variable in the dense FastPoly layout.
        """
        self.mod = mod
        self.n_vars = n_vars
        self.max_degree = max_degree

        # Maps canonical_key (bytes) -> minimum step index at which this polynomial
        # was first constructed in a successful episode (1-indexed). Lower = better.
        self._known: Dict[bytes, int] = {}

        # Canonical keys of the initial/base nodes (x0, x1, ..., x_{n-1}, 1).
        # These are available at the start of every episode and are never treated
        # as subgoals or stored in the library.
        self._base_keys: Set[bytes] = self._compute_base_keys()

        # SymPy symbol list — lazily initialized on first call to factorize_target()
        # to avoid importing SymPy at module load time (it's slow to import).
        self._syms: Optional[list] = None

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _compute_base_keys(self) -> Set[bytes]:
        """Compute and return canonical keys for all initial/base nodes.

        Base nodes are x0, x1, ..., x_{n-1} (input variables) and the constant
        polynomial 1. They are always present at episode start and are excluded
        from the library and from subgoal lists because the agent never needs to
        'discover' them — they are given for free.

        Returns:
            Frozen set of canonical key bytes (one per base node).
        """
        keys: Set[bytes] = set()
        for i in range(self.n_vars):
            poly = FastPoly.variable(i, self.n_vars, self.max_degree, self.mod)
            keys.add(poly.canonical_key())
        keys.add(
            FastPoly.constant(1, self.n_vars, self.max_degree, self.mod).canonical_key()
        )
        return keys

    def _get_syms(self) -> list:
        """Lazily initialize and return the list of SymPy symbols [x0, ..., x_{n-1}].

        Deferred import avoids paying SymPy's heavy import cost until the first
        factorization is actually needed.

        Returns:
            List of SymPy Symbol objects corresponding to each polynomial variable.
        """
        if self._syms is None:
            from sympy import symbols as sympy_symbols
            self._syms = list(sympy_symbols(f"x0:{self.n_vars}"))
        return self._syms

    # -------------------------------------------------------------------------
    # Library write operations
    # -------------------------------------------------------------------------

    def register(self, poly: FastPoly, step_num: int = 1) -> None:
        """Register a polynomial as known, recording its construction cost.

        If the polynomial is already in the library with a lower or equal step
        count, the existing entry is preserved (we keep the cheapest known
        construction). Base nodes (variables, constant 1) are silently ignored
        since they do not need to be 'learned'.

        Args:
            poly: The FastPoly polynomial to register.
            step_num: The 1-indexed step at which this polynomial was constructed
                      within its episode (1 = first operation result, 2 = second,
                      etc.). Used as a proxy for construction cost; lower is better.
        """
        key = poly.canonical_key()
        if key in self._base_keys:
            return  # Base nodes are never stored in the library.
        existing = self._known.get(key)
        if existing is None or step_num < existing:
            self._known[key] = step_num

    def register_episode_nodes(self, nodes: List[FastPoly], n_initial: int) -> None:
        """Register all agent-built nodes from a just-completed successful episode.

        Called immediately after an episode ends with is_success=True. Adds every
        node that was constructed by the agent during the episode (i.e., all nodes
        after the initial base nodes) to the library.

        The step number assigned to each node is its 1-indexed position in the
        construction sequence (first operation result = step 1, second = step 2,
        ...). This is a lower-bound on how many operations were needed to reach it.

        Args:
            nodes: Full ordered list of circuit nodes (base nodes + constructed nodes).
                   nodes[0 : n_initial] are the initial base nodes; nodes[n_initial:]
                   are the nodes the agent built during the episode.
            n_initial: Number of initial base nodes (= n_vars + 1). Nodes at
                       indices [n_initial:] are the agent-built ones to register.
        """
        for step_idx, poly in enumerate(nodes[n_initial:], start=1):
            self.register(poly, step_num=step_idx)

    # -------------------------------------------------------------------------
    # Library read operations
    # -------------------------------------------------------------------------

    def is_base(self, poly: FastPoly) -> bool:
        """Return True if poly is one of the initial free base nodes.

        Base nodes are x0, x1, ..., x_{n-1} (input variables) and the constant
        polynomial 1. They are never useful subgoals because the agent has them
        for free at every episode start.

        Args:
            poly: The FastPoly polynomial to test.

        Returns:
            True if poly's canonical key matches any base node, False otherwise.
        """
        return poly.canonical_key() in self._base_keys

    def factorize_target(self, target: FastPoly) -> List[FastPoly]:
        """Factorize the target polynomial and return its non-trivial factors mod p.

        Convenience wrapper around factorize_poly() that automatically excludes
        the target's own canonical key (so an irreducible target does not produce
        itself as a subgoal).

        See factorize_poly() for full details on filtering rules and examples.

        Args:
            target: The target FastPoly polynomial for the current episode.

        Returns:
            List of distinct non-trivial FastPoly factors of the target. Returns an
            empty list if the target is irreducible, constant, or if SymPy
            factorization raises an exception.
        """
        return self.factorize_poly(target, exclude_keys={target.canonical_key()})

    def factorize_poly(
        self,
        poly: FastPoly,
        exclude_keys: Optional[Set[bytes]] = None,
    ) -> List[FastPoly]:
        """Factorize poly over Z and return filtered non-trivial factors mod p.

        Factorization is performed over Z (integers) using SymPy's factor_list()
        for numerical stability. Each factor is then converted to a FastPoly, which
        automatically reduces all coefficients modulo p.

        A factor is included in the result if and only if:
          - It is a polynomial of total degree >= 1 (not a pure scalar constant).
          - It is not identical to any base input node (x0, ..., x_{n-1}, or 1).
          - It is not the zero polynomial after mod-p reduction.
          - Its canonical key is not in exclude_keys (caller-supplied set of keys
            to skip — typically already-known subgoal keys).
          - It is not poly itself (irreducible case: skipped via exclude_keys or
            the poly_key guard).
          - It has not already appeared in the result (deduplicated by canonical key,
            so (x0+1)^3 yields a single result entry rather than three copies).

        Factorization examples (mod p):
            poly = (x0 + 1)^3   →   [FastPoly(x0 + 1)]    (multiplicity collapsed)
            poly = 3*(x0+1)^2   →   [FastPoly(x0+1)]      (scalar 3 filtered)
            poly = x0*(x0+1)    →   [FastPoly(x0+1)]      (x0 is base node, filtered)
            poly = (x0+1)*(x1+2) →  [FastPoly(x0+1), FastPoly(x1+2)]
            poly = x0+x1 (irreducible) →  []               (only factor = poly itself)

        Args:
            poly: The FastPoly polynomial to factorize.
            exclude_keys: Optional set of canonical key bytes to skip. Keys in this
                          set will be excluded from the result even if they are valid
                          factors. Typically set to self._subgoal_keys to avoid
                          re-adding already-known subgoals. Base keys are always
                          excluded regardless of this parameter.

        Returns:
            List of distinct non-trivial FastPoly factors, with excluded keys and
            base nodes removed. Returns an empty list on any failure.
        """
        import sympy
        from .polynomial_utils import fast_to_sympy, sympy_to_fast

        if exclude_keys is None:
            exclude_keys = set()

        syms = self._get_syms()

        # Handle trivial degenerate cases quickly (no SymPy overhead).
        if poly.is_zero():
            return []

        # Convert to SymPy. O(nnz) — fast for small polynomials.
        try:
            expr = fast_to_sympy(poly, syms)
        except Exception:
            return []

        if expr == sympy.Integer(0) or expr.is_number:
            return []

        # Canonical key of the input poly — always excluded from results.
        poly_key = poly.canonical_key()

        # Factorize over Z. Returns (content, [(factor_expr, multiplicity), ...]).
        # Using ZZ (integers) is more stable than factoring directly over F_p.
        try:
            _content, factors = sympy.factor_list(expr, *syms)
        except Exception:
            # Gracefully degrade: return no subgoals rather than crashing.
            return []

        result: List[FastPoly] = []
        seen_keys: Set[bytes] = set()

        for factor_expr, _mult in factors:
            # Skip pure scalar constants (degree-0 over Z, e.g. 3 or -1).
            try:
                if factor_expr.is_number:
                    continue
                fpoly_sympy = sympy.Poly(factor_expr, *syms)
                if fpoly_sympy.total_degree() == 0:
                    continue
            except (sympy.PolynomialError, Exception):
                continue

            # Convert to FastPoly (reduces coefficients mod p automatically).
            try:
                fast_factor = sympy_to_fast(
                    factor_expr, syms, self.mod, self.max_degree
                )
            except Exception:
                continue

            # Skip zero polynomial (all coefficients vanish mod p).
            if fast_factor.is_zero():
                continue

            key = fast_factor.canonical_key()

            # Skip poly itself (irreducible polynomial returns itself as its factor).
            if key == poly_key:
                continue

            # Skip base nodes (x0, x1, ..., 1) — always available for free.
            if key in self._base_keys:
                continue

            # Skip caller-excluded keys (e.g. already-known subgoals).
            if key in exclude_keys:
                continue

            # Deduplicate: (x0+1, mult=3) → one entry only.
            if key in seen_keys:
                continue
            seen_keys.add(key)

            result.append(fast_factor)

        return result

    def exact_quotient(
        self, dividend: FastPoly, divisor: FastPoly
    ) -> Optional[FastPoly]:
        """Return the exact polynomial quotient dividend / divisor mod p, or None.

        Performs pseudo-division over Z (integers) using SymPy. Checks if the
        remainder is zero modulo p — this correctly handles cases where the ZZ
        remainder is a non-zero multiple of p (e.g., T = x^2 + 4 and v = x + 1
        over F_5: ZZ remainder is 5, which reduces to 0 mod 5).

        If the remainder vanishes mod p, returns the quotient as a FastPoly
        (with coefficients reduced mod p). The quotient may be a scalar
        (constant polynomial); callers can test this via quotient.is_scalar().

        Returns None if:
          - divisor is the zero polynomial.
          - The remainder is non-zero mod p (divisor does not divide dividend).
          - Any SymPy conversion or division step raises an exception.

        Args:
            dividend: The polynomial to be divided (typically the target T).
            divisor: The candidate divisor (typically the newly built node v).

        Returns:
            FastPoly quotient if divisor | dividend exactly over F_p, else None.
        """
        import sympy
        from .polynomial_utils import fast_to_sympy, sympy_to_fast

        if divisor.is_zero():
            return None

        syms = self._get_syms()

        try:
            dividend_expr = fast_to_sympy(dividend, syms)
            divisor_expr = fast_to_sympy(divisor, syms)
        except Exception:
            return None

        if divisor_expr == sympy.Integer(0):
            return None

        # Pseudo-division over ZZ. f = g * q + r where r is reduced w.r.t. g.
        # If r ≡ 0 (mod p), divisor divides dividend over F_p.
        try:
            quotient_expr, remainder_expr = sympy.div(
                dividend_expr, divisor_expr, *syms, domain="ZZ"
            )
        except Exception:
            return None

        # Check remainder mod p. sympy_to_fast reduces coefficients mod p.
        try:
            remainder_poly = sympy_to_fast(
                remainder_expr, syms, self.mod, self.max_degree
            )
        except Exception:
            return None

        if not remainder_poly.is_zero():
            return None  # Not an exact factor over F_p.

        try:
            quotient_poly = sympy_to_fast(
                quotient_expr, syms, self.mod, self.max_degree
            )
        except Exception:
            return None

        if quotient_poly.is_zero():
            return None

        return quotient_poly

    def filter_known(self, factor_polys: List[FastPoly]) -> Set[bytes]:
        """Return canonical keys from factor_polys that are already in the library.

        Used at episode reset to identify which factor subgoals the agent has
        previously built in a past successful episode. These earn a library_bonus
        reward (on top of the standard subgoal reward) when rediscovered during
        the current episode.

        Args:
            factor_polys: List of factor polynomials returned by factorize_target().

        Returns:
            Set of canonical key bytes for the subset of factor_polys that are
            currently registered in the library. May be empty if the library is
            empty or no factor is known.
        """
        return {
            p.canonical_key()
            for p in factor_polys
            if p.canonical_key() in self._known
        }

    def contains(self, poly: FastPoly) -> bool:
        """Check if a polynomial has been registered in the library.

        Args:
            poly: The FastPoly polynomial to look up.

        Returns:
            True if poly has been registered (i.e., was built in at least one
            past successful episode), False otherwise.
        """
        return poly.canonical_key() in self._known

    def __len__(self) -> int:
        """Return the number of distinct polynomials currently in the library.

        Returns:
            Integer count of registered polynomials (excludes base nodes, which
            are never stored).
        """
        return len(self._known)

    def __repr__(self) -> str:
        return f"FactorLibrary(size={len(self)}, n_vars={self.n_vars}, mod={self.mod})"
