"""Top-down polynomial decomposition environment (``BreakdownGame``).

This module provides ``BreakdownGame`` — a Gymnasium-style environment in
which the agent decomposes a target polynomial into smaller sub-polynomials
(rather than building a circuit bottom-up from variables).

Episode loop
------------
1. The environment is reset with a target polynomial ``T`` over F_p[x_0, ...].
2. ``T`` is placed on a *frontier* (FIFO queue) of polynomials still to be
   decomposed.
3. At each step, the polynomial at the head of the frontier is the *focus*;
   the agent picks one of up to ``max_options`` candidate decompositions
   ``focus = A op B`` (with ``op`` in {add, mul}). The two children
   ``A``, ``B`` are appended to the frontier unless they are base nodes
   (x_0, ..., x_{n-1}, or the constant 1).
4. The episode terminates when the frontier is empty (success) or when
   ``max_steps`` decomposition steps have been taken (timeout).

Decomposition candidates
------------------------
The candidate list at each step is generated using:

* The existing :class:`FactorLibrary` for **multiplicative** splits — its
  ``factorize_poly`` and ``exact_quotient`` helpers yield ``focus = f * (focus / f)``
  splits over Z reduced mod p.
* A library of **leaf-residual** splits — for each polynomial already known
  to be a leaf or previously-built node ``v``, if ``focus - v`` is non-zero
  we propose ``focus = v + (focus - v)`` as an additive split.
* A small **monomial peel-off** fallback that removes a single monomial of
  ``focus`` at a time. This guarantees that *every* non-base polynomial has
  at least one valid decomposition, so the environment can never deadlock
  and the agent can always reach a successful terminal state.

The environment is purely classical / numpy-based; no torch or jax is
imported at module load time, so importing it does not affect the existing
PPO/SAC training loops.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Set, Tuple

import numpy as np

from ...config import Config
from ...environment.factor_library import FactorLibrary
from ...environment.fast_polynomial import FastPoly


# Default maximum number of decomposition options exposed to the agent at
# each step. The candidate list is padded with masked-out slots up to this
# size so the action space is fixed.
DEFAULT_MAX_OPTIONS: int = 32

# Default maximum number of decomposition steps per episode (frontier pops).
# Matched roughly to the existing forward game's ``max_steps`` so episodes
# remain comparable.
DEFAULT_MAX_BREAKDOWN_STEPS: int = 16

# Number of "peel one monomial" additive candidates to add as a guaranteed
# fallback. Keeping this small avoids drowning out the more useful
# library-driven splits but still ensures the action set is non-empty.
DEFAULT_MONOMIAL_PEELS: int = 4


@dataclass
class BreakdownCandidate:
    """One concrete decomposition option ``focus = left op right``.

    Attributes:
        op: ``"add"`` or ``"mul"``.
        left: First child sub-polynomial.
        right: Second child sub-polynomial.
        kind: Provenance tag for logging / debugging
            (``"factor"``, ``"leaf-add"``, or ``"monomial"``).
    """

    op: str
    left: FastPoly
    right: FastPoly
    kind: str = "unknown"


@dataclass
class BreakdownObservation:
    """Numerical observation exposed to the policy at one decomposition step.

    All tensors are plain numpy arrays so the env stays torch-free.

    Attributes:
        focus_vec: Flattened coefficient vector of the current focus poly,
            normalised by ``mod`` to lie in ``[0, 1]``.
        target_vec: Flattened coefficient vector of the original target.
        context_vec: A handful of scalar episode-level features (steps taken,
            frontier size, etc.) — useful for the value head.
        candidate_features: Stacked numerical features for each of the
            ``max_options`` candidate decompositions (zeros where masked).
        mask: Boolean validity mask over the ``max_options`` candidate slots.
    """

    focus_vec: np.ndarray
    target_vec: np.ndarray
    context_vec: np.ndarray
    candidate_features: np.ndarray
    mask: np.ndarray


def _peel_monomials(
    poly: FastPoly, max_count: int = DEFAULT_MONOMIAL_PEELS
) -> List[Tuple[FastPoly, FastPoly]]:
    """Return up to ``max_count`` ``(monomial, residual)`` pairs of ``poly``.

    The monomials are taken in descending total-degree order so the largest
    chunk of the polynomial is peeled first. ``residual = poly - monomial``
    is computed via FastPoly arithmetic (numpy, no SymPy).
    """
    nonzero = list(zip(*np.nonzero(poly.coeffs)))
    if not nonzero:
        return []
    nonzero.sort(key=lambda idx: (-sum(idx), idx))
    out: List[Tuple[FastPoly, FastPoly]] = []
    for idx in nonzero[:max_count]:
        mono_coeffs = np.zeros_like(poly.coeffs)
        mono_coeffs[idx] = poly.coeffs[idx]
        mono = FastPoly(mono_coeffs, poly.mod)
        residual = poly - mono
        out.append((mono, residual))
    return out


def _candidate_features(cand: BreakdownCandidate, target: FastPoly) -> np.ndarray:
    """Compact per-candidate feature vector used by the policy network.

    Currently we expose:

    * a flag for additive vs. multiplicative,
    * the term-similarity of each child to the target,
    * coarse "size" metrics (count of non-zero coefficients) of each child.

    These are enough for a small MLP to learn to prefer well-shaped splits
    without us having to embed full polynomials per candidate.
    """
    is_mul = 1.0 if cand.op == "mul" else 0.0
    sim_l = cand.left.term_similarity(target) if not cand.left.is_zero() else 0.0
    sim_r = cand.right.term_similarity(target) if not cand.right.is_zero() else 0.0
    nnz_l = float(np.count_nonzero(cand.left.coeffs))
    nnz_r = float(np.count_nonzero(cand.right.coeffs))
    return np.asarray([is_mul, sim_l, sim_r, nnz_l, nnz_r], dtype=np.float32)


CANDIDATE_FEATURE_DIM: int = 5
CONTEXT_FEATURE_DIM: int = 5


class BreakdownGame:
    """Top-down polynomial decomposition game.

    The environment maintains an evolving decomposition tree rooted at the
    target polynomial. Each step grows the tree by one internal node by
    applying one of up to ``max_options`` candidate splits to the next
    polynomial in the frontier.

    The class is deliberately self-contained: it does **not** subclass or
    import the existing :class:`CircuitGame`. It only borrows the
    :class:`FactorLibrary` (for guided splits) and :class:`FastPoly`
    (for fast modular polynomial arithmetic). This makes it safe to coexist
    with the original training loops without risk of accidental side
    effects.

    Attributes:
        config (Config): Shared configuration (``mod``, ``n_variables``,
            ``effective_max_degree``, ``max_complexity``...).
        factor_library (FactorLibrary): Required — used for multiplicative
            splits. A new library is constructed by the caller; this env
            does not try to share it with the forward CircuitGame.
        max_options (int): Size of the fixed action space at each step.
        max_steps (int): Maximum number of decomposition steps per episode.
        target_poly (FastPoly | None): The current episode's target.
        tree (Dict[bytes, Tuple[str, bytes, bytes]]): Mapping from a
            polynomial's canonical key to its chosen split as
            ``(op, left_key, right_key)``.
        leaf_keys (Set[bytes]): Canonical keys of polynomials we treat as
            leaves of the tree (base inputs + the constant 1).
        steps_taken (int): Number of decomposition steps applied this
            episode.
        done (bool): Whether the episode has ended (success or timeout).
    """

    def __init__(
        self,
        config: Config,
        factor_library: Optional[FactorLibrary] = None,
        max_options: int = DEFAULT_MAX_OPTIONS,
        max_steps: int = DEFAULT_MAX_BREAKDOWN_STEPS,
        success_reward: Optional[float] = None,
        step_penalty: Optional[float] = None,
        factor_subgoal_reward: Optional[float] = None,
        factor_library_bonus: Optional[float] = None,
        size_penalty_per_node: float = 0.05,
    ) -> None:
        """Construct a fresh breakdown environment.

        Args:
            config: Shared :class:`Config` object. Only the polynomial-ring
                fields (``n_variables``, ``mod``, ``effective_max_degree``,
                ``max_complexity``) are used here.
            factor_library: Optional :class:`FactorLibrary`. If ``None``,
                a fresh in-memory library is created. The library is *only*
                used for proposing splits; it is **not** mutated unless the
                caller explicitly registers nodes after a successful
                episode (a helper is provided below).
            max_options: Size of the fixed-width discrete action space the
                agent sees each step. Excess candidates are dropped, missing
                ones are padded with mask=False.
            max_steps: Hard limit on decomposition steps per episode.
            success_reward: Override for the terminal success reward
                (defaults to ``config.success_reward``).
            step_penalty: Override for the per-step penalty
                (defaults to ``config.step_penalty``).
            factor_subgoal_reward: Override for the bonus given when a
                child of the chosen split matches a known factor of the
                target (defaults to ``config.factor_subgoal_reward``).
            factor_library_bonus: Override for the additional bonus when
                that child is also already in the library
                (defaults to ``config.factor_library_bonus``).
            size_penalty_per_node: Soft penalty applied at episode end for
                every internal tree node *beyond* ``config.max_complexity``.
                Encourages compact decomposition trees.
        """
        self.config = config
        self.n_vars = config.n_variables
        self.mod = config.mod
        self.max_deg = config.effective_max_degree
        self.target_size = config.target_size
        self.max_complexity = config.max_complexity
        self.max_options = int(max_options)
        self.max_steps = int(max_steps)

        # Reward overrides default to the existing forward-game settings so
        # the two games live on a comparable scale.
        self.success_reward = (
            success_reward if success_reward is not None else config.success_reward
        )
        self.step_penalty = (
            step_penalty if step_penalty is not None else config.step_penalty
        )
        self.factor_subgoal_reward = (
            factor_subgoal_reward
            if factor_subgoal_reward is not None
            else config.factor_subgoal_reward
        )
        self.factor_library_bonus = (
            factor_library_bonus
            if factor_library_bonus is not None
            else config.factor_library_bonus
        )
        self.size_penalty_per_node = float(size_penalty_per_node)

        # Always have a factor library — we rely on it for proposing splits.
        if factor_library is None:
            factor_library = FactorLibrary(
                mod=self.mod, n_vars=self.n_vars, max_degree=self.max_deg
            )
        self.factor_library = factor_library

        # Pre-compute the canonical keys (and FastPolys) for the base nodes
        # x_0, ..., x_{n-1} and the constant 1. These are *always* leaves.
        self._base_polys: List[FastPoly] = []
        for i in range(self.n_vars):
            self._base_polys.append(
                FastPoly.variable(i, self.n_vars, self.max_deg, self.mod)
            )
        self._base_polys.append(
            FastPoly.constant(1, self.n_vars, self.max_deg, self.mod)
        )
        self._base_keys: Set[bytes] = {p.canonical_key() for p in self._base_polys}

        # Mutable per-episode state — initialised properly in ``reset``.
        self.target_poly: Optional[FastPoly] = None
        self._target_factor_keys: Set[bytes] = set()
        self._library_known_factor_keys: Set[bytes] = set()
        self._claimed_factor_keys: Set[bytes] = set()
        self.tree: Dict[bytes, Tuple[str, bytes, bytes]] = {}
        self.poly_by_key: Dict[bytes, FastPoly] = {}
        self.leaf_keys: Set[bytes] = set()
        self._frontier: Deque[bytes] = deque()
        self._candidates: List[BreakdownCandidate] = []
        self.steps_taken: int = 0
        self.done: bool = True
        self._last_reward_terms: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------

    def reset(self, target_poly: FastPoly) -> BreakdownObservation:
        """Begin a new decomposition episode for the given target polynomial.

        The target is placed on the frontier and its non-trivial factors are
        recorded as "subgoal" canonical keys for the duration of the
        episode (mirroring the forward CircuitGame's behaviour). The library
        is consulted for which of those factors are already known.

        Args:
            target_poly: The polynomial whose decomposition tree the agent
                must construct.

        Returns:
            The initial observation. If the target is itself a base node the
            episode terminates immediately with success and the returned
            observation is a fully-masked dummy state — callers should
            check ``done`` after ``reset``.
        """
        self.target_poly = target_poly
        self.tree = {}
        self.poly_by_key = {}
        self.leaf_keys = set(self._base_keys)
        self._frontier = deque()
        self._candidates = []
        self.steps_taken = 0
        self.done = False
        self._last_reward_terms = {}

        for poly in self._base_polys:
            self.poly_by_key[poly.canonical_key()] = poly

        # Subgoal factors for the *original* target polynomial — used to
        # award shaping bonuses when one of them shows up as a child of the
        # chosen split.
        self._target_factor_keys = set()
        self._library_known_factor_keys = set()
        self._claimed_factor_keys = set()
        if self.config.factor_library_enabled:
            try:
                factors = self.factor_library.factorize_target(target_poly)
            except Exception:
                factors = []
            for f in factors:
                self._target_factor_keys.add(f.canonical_key())
            self._library_known_factor_keys = self.factor_library.filter_known(factors)

        target_key = target_poly.canonical_key()
        self.poly_by_key[target_key] = target_poly

        if target_key in self._base_keys or target_poly.is_zero():
            # Trivial target — no decomposition needed.
            self.done = True
            return self._empty_observation()

        self._frontier.append(target_key)
        self._advance_frontier()  # in case target was already a leaf above
        return self._build_observation()

    def step(
        self, action_idx: int
    ) -> Tuple[BreakdownObservation, float, bool, dict]:
        """Apply one decomposition action.

        Args:
            action_idx: Index in ``[0, max_options)`` selecting one of the
                pre-computed candidates for the current focus polynomial.

        Returns:
            A tuple ``(obs, reward, done, info)``:

            * ``obs`` — observation for the next focus polynomial, or a
              dummy zero observation when the episode is finished.
            * ``reward`` — scalar reward composed of step penalty, optional
              factor subgoal bonus, and (on terminal step) success reward
              minus an oversize-tree penalty.
            * ``done`` — True iff the episode has just ended.
            * ``info`` — diagnostic dict with keys ``is_success``,
              ``factor_hit``, ``library_hit``, ``op``, ``left_kind``,
              ``frontier_size``, ``tree_size``, ``steps_taken``,
              ``num_internal_nodes``.

        Raises:
            AssertionError: if the episode is already done or the chosen
                slot is invalid (mask is False).
        """
        assert not self.done, "Episode already done; call reset() first."
        assert 0 <= action_idx < self.max_options, (
            f"action_idx {action_idx} out of range [0, {self.max_options})"
        )
        assert action_idx < len(self._candidates), (
            f"action_idx {action_idx} selects a padded slot (no candidate)"
        )

        cand = self._candidates[action_idx]
        focus_key = self._frontier.popleft()
        focus_poly = self.poly_by_key[focus_key]

        # Sanity-check: the candidate must reconstruct the focus polynomial.
        if cand.op == "add":
            recomposed = cand.left + cand.right
        else:
            recomposed = cand.left * cand.right
        assert recomposed == focus_poly, (
            "Internal error: candidate does not reproduce focus polynomial"
        )

        # Register children in the lookup tables.
        left_key = cand.left.canonical_key()
        right_key = cand.right.canonical_key()
        self.poly_by_key[left_key] = cand.left
        self.poly_by_key[right_key] = cand.right
        self.tree[focus_key] = (cand.op, left_key, right_key)

        reward = self.step_penalty
        factor_hit = False
        library_hit = False

        # Reward shaping: if either child matches a known factor of the
        # *original* target, hand out the same subgoal bonuses the forward
        # CircuitGame uses.
        for child_key in (left_key, right_key):
            if (
                self.config.factor_library_enabled
                and child_key in self._target_factor_keys
                and child_key not in self._claimed_factor_keys
            ):
                self._claimed_factor_keys.add(child_key)
                reward += self.factor_subgoal_reward
                factor_hit = True
                if child_key in self._library_known_factor_keys:
                    reward += self.factor_library_bonus
                    library_hit = True

        # Push non-leaf children onto the frontier; leaves are absorbed.
        for child_poly, child_key in ((cand.left, left_key), (cand.right, right_key)):
            if child_key in self._base_keys or child_poly.is_zero():
                self.leaf_keys.add(child_key)
                continue
            if child_key in self.tree or child_key in self._frontier:
                # Already decomposed (or queued) — DAG sharing, no-op.
                continue
            self._frontier.append(child_key)

        self.steps_taken += 1
        self._advance_frontier()

        is_success = False
        if not self._frontier:
            # Frontier empty → every leaf of the tree is a base node, the
            # decomposition terminates successfully.
            self.done = True
            is_success = True
            reward += self.success_reward
            # Soft penalty for oversized trees: the eventual circuit needs
            # one operation per *internal* tree node.
            num_internal = self._count_distinct_internal_nodes()
            overshoot = max(0, num_internal - self.max_complexity)
            reward -= self.size_penalty_per_node * overshoot

        elif self.steps_taken >= self.max_steps:
            # Out of step budget — terminate without success.
            self.done = True
            reward += -1.0  # mild timeout penalty (kept small on purpose)

        info = {
            "is_success": is_success,
            "factor_hit": factor_hit,
            "library_hit": library_hit,
            "op": cand.op,
            "left_kind": cand.kind,
            "frontier_size": len(self._frontier),
            "tree_size": len(self.tree),
            "steps_taken": self.steps_taken,
            "num_internal_nodes": self._count_distinct_internal_nodes(),
        }
        self._last_reward_terms = {
            "step": self.step_penalty,
            "factor_hit": float(factor_hit),
            "library_hit": float(library_hit),
            "success": float(is_success),
            "reward": reward,
        }

        if self.done:
            return self._empty_observation(), reward, True, info
        return self._build_observation(), reward, False, info

    # ------------------------------------------------------------------
    # Helpers — frontier and candidates
    # ------------------------------------------------------------------

    def _advance_frontier(self) -> None:
        """Skip frontier entries that are already decomposed or are leaves.

        The frontier may contain duplicate or now-leaf entries due to DAG
        sharing or post-step changes. This method drops those so that the
        next observed focus is always a fresh non-leaf polynomial.
        """
        while self._frontier:
            head = self._frontier[0]
            if head in self.tree or head in self._base_keys:
                self._frontier.popleft()
                continue
            poly = self.poly_by_key.get(head)
            if poly is None or poly.is_zero():
                self._frontier.popleft()
                continue
            self._refresh_candidates(poly)
            if not self._candidates:
                # No valid decomposition possible — treat as leaf and skip.
                self.leaf_keys.add(head)
                self._frontier.popleft()
                continue
            return

        # Fallthrough: frontier exhausted. Caller will set done=True.
        self._candidates = []

    def _refresh_candidates(self, focus: FastPoly) -> None:
        """Populate ``self._candidates`` for the given focus polynomial.

        Combines three sources:

        1. Multiplicative splits ``focus = f * (focus / f)`` where ``f`` is
           a non-trivial factor returned by :meth:`FactorLibrary.factorize_poly`
           and the exact quotient exists over F_p.
        2. Leaf-residual splits ``focus = v + (focus - v)`` for every
           current leaf or registered library polynomial ``v`` such that
           ``focus - v`` is non-zero and not equal to ``focus``.
        3. Monomial peel-offs ``focus = m + (focus - m)`` for the top few
           monomials of ``focus``. Always applicable, guarantees progress.

        The list is deduplicated by the unordered key
        ``(op, frozenset({left_key, right_key}))`` and truncated to
        ``self.max_options``.
        """
        candidates: List[BreakdownCandidate] = []
        seen: Set[Tuple[str, frozenset]] = set()

        def _try_add(op: str, left: FastPoly, right: FastPoly, kind: str) -> None:
            if left.is_zero() or right.is_zero():
                return
            l_key = left.canonical_key()
            r_key = right.canonical_key()
            focus_key = focus.canonical_key()
            if l_key == focus_key or r_key == focus_key:
                # Trivial split focus = 1*focus or focus = focus + 0 — useless.
                return
            sig = (op, frozenset({l_key, r_key}))
            if sig in seen:
                return
            seen.add(sig)
            candidates.append(BreakdownCandidate(op=op, left=left, right=right, kind=kind))

        # 1. Multiplicative via factor library.
        try:
            factors = self.factor_library.factorize_poly(focus)
        except Exception:
            factors = []
        for f in factors:
            if len(candidates) >= self.max_options:
                break
            try:
                cofactor = self.factor_library.exact_quotient(focus, f)
            except Exception:
                cofactor = None
            if cofactor is None or cofactor.is_zero():
                continue
            _try_add("mul", f, cofactor, kind="factor")

        # 2. Leaf-residual additive splits, preferring leaves first.
        for v_key in list(self.leaf_keys):
            if len(candidates) >= self.max_options:
                break
            v_poly = self.poly_by_key.get(v_key)
            if v_poly is None or v_poly.is_zero():
                continue
            residual = focus - v_poly
            if residual.is_zero():
                continue
            _try_add("add", v_poly, residual, kind="leaf-add")

        # 3. Top monomial peel-offs (always include — guarantees progress).
        for mono, residual in _peel_monomials(focus, max_count=DEFAULT_MONOMIAL_PEELS):
            if len(candidates) >= self.max_options:
                break
            _try_add("add", mono, residual, kind="monomial")

        self._candidates = candidates[: self.max_options]

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------

    def _build_observation(self) -> BreakdownObservation:
        """Construct an observation for the current focus polynomial.

        All numeric arrays are float32 to keep downstream tensor conversion
        cheap. The mask is bool with True for valid candidate slots.
        """
        focus_key = self._frontier[0]
        focus = self.poly_by_key[focus_key]

        focus_vec = (focus.to_vector() / self.mod).astype(np.float32)
        target_vec = (self.target_poly.to_vector() / self.mod).astype(np.float32)

        # Context features: scalar episode-level stats normalised to [0, 1]
        # to keep the network input well-conditioned.
        ctx = np.asarray(
            [
                self.steps_taken / max(self.max_steps, 1),
                len(self._frontier) / max(self.max_steps, 1),
                len(self.tree) / max(self.max_steps, 1),
                len(self.leaf_keys) / max(self.max_steps + len(self._base_keys), 1),
                focus.term_similarity(self.target_poly),
            ],
            dtype=np.float32,
        )

        feats = np.zeros((self.max_options, CANDIDATE_FEATURE_DIM), dtype=np.float32)
        mask = np.zeros(self.max_options, dtype=bool)
        for i, cand in enumerate(self._candidates):
            feats[i] = _candidate_features(cand, self.target_poly)
            mask[i] = True

        return BreakdownObservation(
            focus_vec=focus_vec,
            target_vec=target_vec,
            context_vec=ctx,
            candidate_features=feats,
            mask=mask,
        )

    def _empty_observation(self) -> BreakdownObservation:
        """Zero-filled observation for terminal / trivial states."""
        return BreakdownObservation(
            focus_vec=np.zeros(self.target_size, dtype=np.float32),
            target_vec=(self.target_poly.to_vector() / self.mod).astype(np.float32)
            if self.target_poly is not None
            else np.zeros(self.target_size, dtype=np.float32),
            context_vec=np.zeros(CONTEXT_FEATURE_DIM, dtype=np.float32),
            candidate_features=np.zeros(
                (self.max_options, CANDIDATE_FEATURE_DIM), dtype=np.float32
            ),
            mask=np.zeros(self.max_options, dtype=bool),
        )

    # ------------------------------------------------------------------
    # Bookkeeping helpers
    # ------------------------------------------------------------------

    def _count_distinct_internal_nodes(self) -> int:
        """Number of distinct *non-leaf* polynomials in the decomposition tree.

        Each such node corresponds to one arithmetic operation in the final
        bottom-up circuit, since identical polynomials only need to be
        constructed once.
        """
        return len(set(self.tree.keys()) - self._base_keys)

    def get_candidates(self) -> List[BreakdownCandidate]:
        """Return a *copy* of the current candidate list (for inspection)."""
        return list(self._candidates)

    @property
    def num_valid_options(self) -> int:
        """Number of currently valid (non-padded) action slots."""
        return len(self._candidates)

    @property
    def current_focus(self) -> Optional[FastPoly]:
        """The polynomial the next ``step`` will decompose, or ``None``."""
        if not self._frontier:
            return None
        return self.poly_by_key[self._frontier[0]]

    def clone(self) -> "BreakdownGame":
        """Return a deep-enough copy of this game for use in tree search.

        The factor library reference is shared (it is a session-level cache,
        same convention as :meth:`CircuitGame.clone`). All mutable per-
        episode state is duplicated.
        """
        new = BreakdownGame.__new__(BreakdownGame)
        new.config = self.config
        new.n_vars = self.n_vars
        new.mod = self.mod
        new.max_deg = self.max_deg
        new.target_size = self.target_size
        new.max_complexity = self.max_complexity
        new.max_options = self.max_options
        new.max_steps = self.max_steps
        new.success_reward = self.success_reward
        new.step_penalty = self.step_penalty
        new.factor_subgoal_reward = self.factor_subgoal_reward
        new.factor_library_bonus = self.factor_library_bonus
        new.size_penalty_per_node = self.size_penalty_per_node

        new._base_polys = self._base_polys
        new._base_keys = self._base_keys
        new.factor_library = self.factor_library

        new.target_poly = self.target_poly
        new._target_factor_keys = set(self._target_factor_keys)
        new._library_known_factor_keys = set(self._library_known_factor_keys)
        new._claimed_factor_keys = set(self._claimed_factor_keys)

        new.tree = dict(self.tree)
        new.poly_by_key = dict(self.poly_by_key)
        new.leaf_keys = set(self.leaf_keys)
        new._frontier = deque(self._frontier)
        new._candidates = list(self._candidates)
        new.steps_taken = self.steps_taken
        new.done = self.done
        new._last_reward_terms = dict(self._last_reward_terms)
        return new

    # ------------------------------------------------------------------
    # Library update on success
    # ------------------------------------------------------------------

    def register_decomposition_in_library(self) -> None:
        """Register every distinct internal polynomial of the tree.

        Mirrors :meth:`CircuitGame.register_episode_nodes` so successful
        decompositions enrich the same factor library across episodes.
        Each polynomial is registered with a step count derived from a
        topological ordering (depth-first post-order) of the tree, so
        smaller / shallower sub-polynomials are credited with lower cost.
        """
        if self.target_poly is None:
            return

        order: List[bytes] = []
        seen: Set[bytes] = set()

        def _dfs(key: bytes) -> None:
            if key in seen:
                return
            seen.add(key)
            split = self.tree.get(key)
            if split is None:
                return
            _, lkey, rkey = split
            _dfs(lkey)
            _dfs(rkey)
            order.append(key)

        _dfs(self.target_poly.canonical_key())
        for step_idx, key in enumerate(order, start=1):
            poly = self.poly_by_key.get(key)
            if poly is None:
                continue
            self.factor_library.register(poly, step_num=step_idx)


# ----------------------------------------------------------------------
# Observation conversion helpers (numpy <-> torch). Imported lazily by
# the trainer files so that this module stays torch-free if it is
# imported standalone (e.g. for unit testing the env).
# ----------------------------------------------------------------------


def observation_to_tensors(obs: BreakdownObservation, device: str = "cpu") -> dict:
    """Convert a :class:`BreakdownObservation` into a torch tensor dict.

    The returned dict has the exact keys the breakdown policy networks
    expect: ``focus``, ``target``, ``context``, ``cand_feats``, ``mask``.
    """
    import torch  # local import keeps env importable without torch

    return {
        "focus": torch.from_numpy(np.ascontiguousarray(obs.focus_vec)).to(device),
        "target": torch.from_numpy(np.ascontiguousarray(obs.target_vec)).to(device),
        "context": torch.from_numpy(np.ascontiguousarray(obs.context_vec)).to(device),
        "cand_feats": torch.from_numpy(
            np.ascontiguousarray(obs.candidate_features)
        ).to(device),
        "mask": torch.from_numpy(np.ascontiguousarray(obs.mask)).to(device),
    }


def stack_observations(obs_list: List[BreakdownObservation], device: str = "cpu") -> dict:
    """Batch-stack a list of breakdown observations into tensors.

    Returns a dict with the same keys as :func:`observation_to_tensors` but
    each value carries an extra leading batch dimension.
    """
    import torch

    return {
        "focus": torch.from_numpy(
            np.stack([o.focus_vec for o in obs_list], axis=0)
        ).to(device),
        "target": torch.from_numpy(
            np.stack([o.target_vec for o in obs_list], axis=0)
        ).to(device),
        "context": torch.from_numpy(
            np.stack([o.context_vec for o in obs_list], axis=0)
        ).to(device),
        "cand_feats": torch.from_numpy(
            np.stack([o.candidate_features for o in obs_list], axis=0)
        ).to(device),
        "mask": torch.from_numpy(
            np.stack([o.mask for o in obs_list], axis=0)
        ).to(device),
    }
