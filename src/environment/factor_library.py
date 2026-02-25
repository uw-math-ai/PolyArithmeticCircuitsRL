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

2. During stepping, if the agent constructs a node matching a subgoal factor, it
   receives a factor_subgoal_reward. If that factor was also previously built in
   a past successful episode (and is thus in the library), it additionally receives
   a factor_library_bonus.

3. After a successful episode, every agent-built node is registered in the library
   so future episodes can benefit from it.

Design Constraints
------------------
- No disk I/O: the library lives entirely in RAM for one training session.
- Fast lookups: all membership tests are O(1) set/dict operations on canonical keys.
- SymPy is called only once per episode (at reset), never during per-step execution.
- Factorization is over Z (integers), not F_p, for numerical stability. Factors are
  then reduced mod p when converted to FastPoly.
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

    def factorize_target(self, target: FastPoly) -> List[FastPoly]:
        """Factorize the target polynomial and return its non-trivial factors mod p.

        Factorization is performed over Z (integers) using SymPy's factor_list()
        for numerical stability. Each factor is then converted to a FastPoly, which
        automatically reduces all coefficients modulo p.

        A factor is included in the result if and only if:
          - It is a polynomial of total degree >= 1 (not a pure scalar constant).
          - It is not identical to any base input node (x0, ..., x_{n-1}, or 1),
            since those are already available and building them is not a meaningful
            sub-goal.
          - It is not the zero polynomial after mod-p reduction.
          - It has not already appeared in the result list (deduplicated by canonical
            key, so (x0+1)^3 yields a single subgoal [x0+1] rather than three copies).

        Factorization examples:
            target = (x0 + 1)^3
                factor_list -> (1, [(x0 + 1, 3)])
                Returns [FastPoly(x0 + 1)]          (multiplicity collapsed)

            target = 3 * (x0 + 1)^2
                factor_list -> (3, [(x0 + 1, 2)])
                Returns [FastPoly(x0 + 1)]          (scalar 3 filtered out)

            target = x0 * (x0 + 1)
                factor_list -> (1, [(x0, 1), (x0 + 1, 1)])
                Returns [FastPoly(x0 + 1)]          (x0 is a base node, filtered)

            target = (x0 + 1) * (x1 + 2)
                factor_list -> (1, [(x0+1, 1), (x1+2, 1)])
                Returns [FastPoly(x0+1), FastPoly(x1+2)]

            target = x0 + x1  (irreducible)
                factor_list -> (1, [(x0+x1, 1)])
                Returns []                          (single factor = target itself, skipped below)

        Note: If the target is itself irreducible, factor_list returns it as its
        own single factor. We skip this case (no useful subgoal is produced).

        Args:
            target: The target FastPoly polynomial for the current episode.

        Returns:
            List of distinct non-trivial FastPoly factors of the target. Returns an
            empty list if the target is irreducible, constant, or if SymPy
            factorization raises an exception.
        """
        import sympy
        from .polynomial_utils import fast_to_sympy, sympy_to_fast

        syms = self._get_syms()

        # Convert target to SymPy expression. This is O(nnz) in the number of
        # nonzero coefficients — fast in practice for small polynomials.
        expr = fast_to_sympy(target, syms)

        # Handle trivial degenerate cases quickly.
        if expr == sympy.Integer(0) or expr.is_number:
            return []

        # Factorize over Z. Returns (content, [(factor_expr, multiplicity), ...])
        # where content is the integer GCD and each factor_expr is an irreducible
        # polynomial factor over Z. Using ZZ (integers) is more stable and
        # interpretable than factoring over F_p directly.
        try:
            content, factors = sympy.factor_list(expr, *syms)
        except Exception:
            # Gracefully degrade: if factorization fails (unusual expression type,
            # SymPy bug, etc.), return no subgoals rather than crashing training.
            return []

        # Precompute the canonical key of the target itself so we can skip it
        # (a polynomial is not a useful factor of itself).
        target_key = target.canonical_key()

        result: List[FastPoly] = []
        seen_keys: Set[bytes] = set()

        for factor_expr, mult in factors:
            # Skip pure scalar constants (degree-0 factors like 3 or -1).
            try:
                if factor_expr.is_number:
                    continue
                fpoly_sympy = sympy.Poly(factor_expr, *syms)
                if fpoly_sympy.total_degree() == 0:
                    continue  # Constant polynomial after variable extraction.
            except (sympy.PolynomialError, Exception):
                # Poly() can fail for exotic expressions; skip them.
                continue

            # Convert to FastPoly (reduces all coefficients mod p automatically).
            try:
                fast_factor = sympy_to_fast(
                    factor_expr, syms, self.mod, self.max_degree
                )
            except Exception:
                continue

            # Skip the zero polynomial (can happen when all coefficients vanish mod p).
            if fast_factor.is_zero():
                continue

            key = fast_factor.canonical_key()

            # Skip factors that are identical to the target itself (irreducible case).
            if key == target_key:
                continue

            # Skip base nodes (xi or the constant 1) — always free at episode start.
            if key in self._base_keys:
                continue

            # Deduplicate: (x0+1)^3 should yield one subgoal [x0+1], not three.
            if key in seen_keys:
                continue
            seen_keys.add(key)

            result.append(fast_factor)

        return result

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
