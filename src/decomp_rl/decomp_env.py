"""Split-based decomposition environment."""

from __future__ import annotations

from dataclasses import dataclass, field

from .baseline_cost import BaselineCostModel
from .config import DecompEnvConfig
from .cost_model import rebuild_cost, unresolved_children
from .factor_fp import FactorizationResult, FiniteFieldFactorizer
from .polynomial import SparsePolynomial
from .split_proposals import SplitAction, propose_splits


@dataclass(frozen=True)
class StepInfo:
    active_poly: SparsePolynomial
    action_kind: str
    baseline_before: int
    baseline_after: int
    split: SplitAction | None = None
    g_factorization: FactorizationResult | None = None
    h_factorization: FactorizationResult | None = None
    rebuild_g: int = 0
    rebuild_h: int = 0
    children: tuple[SparsePolynomial, ...] = ()
    cache_hits: tuple[str, ...] = ()
    immediate_reward: int = 0
    direct_cost: int = 0
    library_hits: tuple[str, ...] = ()
    library_reward: float = 0.0


@dataclass
class EnvState:
    frontier: list[SparsePolynomial]
    memo: dict[str, int] = field(default_factory=dict)
    acc_cost: int = 0
    history: list[StepInfo] = field(default_factory=list)


class DecompEnv:
    """Environment that expands a frontier of unresolved polynomials."""

    def __init__(
        self,
        config: DecompEnvConfig | None = None,
        factorizer: FiniteFieldFactorizer | None = None,
        baseline_model: BaselineCostModel | None = None,
        library=None,  # FactorizableLibrary | None
    ) -> None:
        self.config = config or DecompEnvConfig()
        self.library = library
        self.factorizer = factorizer or FiniteFieldFactorizer(library=self.library)
        self.baseline_model = baseline_model or BaselineCostModel(
            exact_support_limit=self.config.exact_support_limit
        )

    def reset(self, target_poly: SparsePolynomial) -> EnvState:
        state = EnvState(frontier=[target_poly], memo={}, acc_cost=0, history=[])
        key = target_poly.to_key()
        if self.baseline_model.is_base_case(target_poly):
            state.memo[key] = self.baseline_model.exact_base_cost(target_poly)
        return state

    def get_active_items(self, state: EnvState) -> list[SparsePolynomial]:
        return list(state.frontier)

    def get_candidate_splits(
        self,
        state: EnvState,
        poly_handle: int,
        k: int,
    ) -> list[SplitAction]:
        target = state.frontier[poly_handle]
        return propose_splits(
            target=target,
            k=k,
            config=self.config.proposal,
            baseline_model=self.baseline_model,
        )

    def _child_resolution_cost(
        self,
        state: EnvState,
        children: tuple[SparsePolynomial, ...],
    ) -> tuple[list[SparsePolynomial], int, tuple[str, ...]]:
        unresolved: list[SparsePolynomial] = []
        resolved_cost = 0
        cache_hits: list[str] = []
        frontier_keys = {poly.to_key() for poly in state.frontier}

        for child in children:
            key = child.to_key()
            if key in state.memo:
                resolved_cost += state.memo[key]
                cache_hits.append(key)
                continue
            if self.baseline_model.is_base_case(child):
                cost = self.baseline_model.exact_base_cost(child)
                state.memo[key] = cost
                resolved_cost += cost
                cache_hits.append(key)
                continue
            if self.config.dedup_frontier and key in frontier_keys:
                continue
            unresolved.append(child)
            frontier_keys.add(key)
        return unresolved, resolved_cost, tuple(cache_hits)

    def step(self, state: EnvState, poly_handle: int, action: SplitAction) -> tuple[EnvState, int, bool, StepInfo]:
        active = state.frontier[poly_handle]
        if action.g + action.h != active:
            raise ValueError("Invalid split: g + h must equal the active polynomial")

        next_state = EnvState(
            frontier=list(state.frontier),
            memo=dict(state.memo),
            acc_cost=state.acc_cost,
            history=list(state.history),
        )
        next_state.frontier.pop(poly_handle)

        g_factors = self.factorizer.factor(action.g)
        h_factors = self.factorizer.factor(action.h)
        rebuild_g = rebuild_cost(g_factors)
        rebuild_h = rebuild_cost(h_factors)

        child_map: dict[str, SparsePolynomial] = {}
        for child in unresolved_children(g_factors) + unresolved_children(h_factors):
            child_map.setdefault(child.to_key(), child)
        children = tuple(child_map.values())
        new_children, resolved_cost, cache_hits = self._child_resolution_cost(next_state, children)

        baseline_before = self.baseline_model.direct_construction_cost(active)
        baseline_after = 1 + rebuild_g + rebuild_h + sum(
            self.baseline_model.direct_construction_cost(child) for child in new_children
        )
        reward = baseline_before - baseline_after

        # Library reward: bonus when either split piece is a known factorizable poly
        library_hits: list[str] = []
        library_reward = 0.0
        if self.library is not None:
            lib_reward_per_hit = self.library.library_step_reward
            for piece, label in ((action.g, "g"), (action.h, "h")):
                if self.library.is_known(piece):
                    library_hits.append(label)
                    library_reward += lib_reward_per_hit

        next_state.frontier.extend(new_children)
        next_state.acc_cost += 1 + rebuild_g + rebuild_h + resolved_cost
        done = len(next_state.frontier) == 0

        info = StepInfo(
            active_poly=active,
            action_kind="split",
            split=action,
            g_factorization=g_factors,
            h_factorization=h_factors,
            rebuild_g=rebuild_g,
            rebuild_h=rebuild_h,
            children=children,
            cache_hits=cache_hits,
            immediate_reward=reward,
            baseline_before=baseline_before,
            baseline_after=baseline_after,
            library_hits=tuple(library_hits),
            library_reward=library_reward,
        )
        next_state.history.append(info)
        return next_state, reward, done, info

    def solve_direct(self, state: EnvState, poly_handle: int) -> tuple[EnvState, int, bool, StepInfo]:
        active = state.frontier[poly_handle]
        next_state = EnvState(
            frontier=list(state.frontier),
            memo=dict(state.memo),
            acc_cost=state.acc_cost,
            history=list(state.history),
        )
        next_state.frontier.pop(poly_handle)
        direct_cost = self.baseline_model.direct_construction_cost(active)
        next_state.memo[active.to_key()] = direct_cost
        next_state.acc_cost += direct_cost
        done = len(next_state.frontier) == 0
        info = StepInfo(
            active_poly=active,
            action_kind="direct",
            baseline_before=direct_cost,
            baseline_after=direct_cost,
            direct_cost=direct_cost,
        )
        next_state.history.append(info)
        return next_state, 0, done, info
