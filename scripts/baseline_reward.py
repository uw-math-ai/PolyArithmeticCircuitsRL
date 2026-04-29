#!/usr/bin/env python3
"""Shared utilities for pure-Python baseline search.

The shaped evaluator mirrors this branch's ``clean_onpath`` reward:

  - step penalty plus terminal success reward
  - cached on-path potential shaping
  - optional positive progress bonus on coherent-route phi increase

It deliberately does not use factor-library rewards.  Targets can still be
passed as bare ``FastPoly`` objects for sparse baselines, but on-path shaping
requires an ``OnPathTargetContext`` from ``OnPathCache``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import FrozenSet, Iterable, List, Optional, Tuple

import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Config  # noqa: E402
from src.environment.fast_polynomial import FastPoly  # noqa: E402
from src.game_board.on_path import OnPathTargetContext  # noqa: E402

Action = Tuple[int, int, int]
ALL_ROUTE_BITS = (1 << 32) - 1


def initial_nodes(config: Config) -> List[FastPoly]:
    """Return the base circuit nodes [x0, ..., x{n-1}, 1]."""
    n_vars = config.n_variables
    max_deg = config.effective_max_degree
    mod = config.mod
    nodes = [FastPoly.variable(i, n_vars, max_deg, mod) for i in range(n_vars)]
    nodes.append(FastPoly.constant(1, n_vars, max_deg, mod))
    return nodes


def enumerate_actions(num_nodes: int) -> List[Action]:
    """Return all valid upper-triangular add/mul actions for num_nodes nodes."""
    return [
        (op, i, j)
        for op in (0, 1)
        for i in range(num_nodes)
        for j in range(i, num_nodes)
    ]


def apply_action(nodes: List[FastPoly], action: Action) -> FastPoly:
    """Apply an action to existing nodes and return the resulting polynomial."""
    op, i, j = action
    return nodes[i] + nodes[j] if op == 0 else nodes[i] * nodes[j]


def active_node_limit(config: Config, context: Optional[OnPathTargetContext]) -> int:
    """Node cap matching JAX env's per-target build-complexity slack rule."""
    static_limit = config.max_nodes
    if config.build_complexity_slack < 0 or context is None:
        return static_limit

    base_nodes = config.n_variables + 1
    max_build = config.effective_max_build_complexity
    target_build = int(context.target_board_step)
    active_build = min(max_build, target_build + config.build_complexity_slack)
    if target_build <= 0:
        active_build = max_build
    active_build = max(1, active_build)
    return min(static_limit, base_nodes + active_build)


@dataclass(frozen=True)
class RewardState:
    """Per-trajectory on-path bookkeeping."""

    on_path_hit_keys: FrozenSet[bytes] = frozenset()
    on_path_count: int = 0
    on_path_deepest_step: int = 0
    on_path_active_route_mask: int = ALL_ROUTE_BITS


@dataclass(frozen=True)
class RewardResult:
    """Reward transition result returned by BaselineRewardEvaluator."""

    reward: float
    next_state: RewardState
    on_path_hit: bool
    phi_before: float
    phi_after: float
    success: bool


class BaselineRewardEvaluator:
    """Per-target reward evaluator for pure Python search baselines."""

    def __init__(
        self,
        config: Config,
        target_or_context: FastPoly | OnPathTargetContext,
    ):
        self.config = config
        self.context: Optional[OnPathTargetContext]
        if isinstance(target_or_context, OnPathTargetContext):
            self.context = target_or_context
            self.target = target_or_context.target_poly
            self.on_path_steps = target_or_context.on_path_keys
            self.on_path_route_masks = target_or_context.on_path_route_keys
            self.on_path_target_step = int(target_or_context.target_board_step)
        else:
            self.context = None
            self.target = target_or_context
            self.on_path_steps = {}
            self.on_path_route_masks = {}
            self.on_path_target_step = 0

        self.on_path_total = len(self.on_path_steps)
        self._route_bits = tuple(self._iter_route_bits())

    def initial_state(self) -> RewardState:
        active_mask = 0
        for mask in self.on_path_route_masks.values():
            active_mask |= int(mask)
        if active_mask == 0:
            active_mask = ALL_ROUTE_BITS
        return RewardState(on_path_active_route_mask=active_mask)

    def best_similarity(self, nodes: List[FastPoly]) -> float:
        return max(node.term_similarity(self.target) for node in nodes)

    def is_on_path_enabled(self) -> bool:
        return self.config.reward_mode == "clean_onpath" and self.context is not None

    def _iter_route_bits(self) -> Iterable[int]:
        route_mask = 0
        for mask in self.on_path_route_masks.values():
            route_mask |= int(mask)
        if route_mask == 0:
            route_mask = 1
        for route_idx in range(32):
            bit = 1 << route_idx
            if route_mask & bit:
                yield bit

    def _route_mode(self) -> str:
        if not self.config.on_path_route_consistency:
            return "off"
        return self.config.on_path_route_consistency_mode

    def _step_weight(self, key: bytes) -> float:
        step = max(0, int(self.on_path_steps.get(key, 0)))
        if step == 0:
            return 0.0
        return float(step) ** float(self.config.on_path_depth_weight_power)

    def _best_route_count_phi(self, state: RewardState) -> float:
        best = 0.0
        for bit in self._route_bits:
            total = 0
            hits = 0
            for key, mask in self.on_path_route_masks.items():
                if int(mask) & bit:
                    total += 1
                    if key in state.on_path_hit_keys:
                        hits += 1
            if total > 0:
                best = max(best, hits / total)
        return best

    def _best_route_max_step_phi(self, state: RewardState) -> float:
        if self.on_path_target_step <= 0:
            return 0.0
        best_step = 0
        for bit in self._route_bits:
            route_has_nodes = False
            deepest = 0
            for key, mask in self.on_path_route_masks.items():
                if int(mask) & bit:
                    route_has_nodes = True
                    if key in state.on_path_hit_keys:
                        deepest = max(deepest, int(self.on_path_steps[key]))
            if route_has_nodes:
                best_step = max(best_step, deepest)
        return best_step / self.on_path_target_step

    def _best_route_depth_weighted_phi(self, state: RewardState) -> float:
        best = 0.0
        for bit in self._route_bits:
            total = 0.0
            hits = 0.0
            for key, mask in self.on_path_route_masks.items():
                if int(mask) & bit:
                    weight = self._step_weight(key)
                    total += weight
                    if key in state.on_path_hit_keys:
                        hits += weight
            if total > 0.0:
                best = max(best, hits / total)
        return best

    def _union_depth_weighted_phi(self, state: RewardState) -> float:
        total = 0.0
        hits = 0.0
        use_active_mask = self._route_mode() == "lock_on_first_hit"
        for key, mask in self.on_path_route_masks.items():
            if use_active_mask and not (int(mask) & state.on_path_active_route_mask):
                continue
            weight = self._step_weight(key)
            total += weight
            if key in state.on_path_hit_keys:
                hits += weight
        if total <= 0.0:
            return 0.0
        return hits / total

    def on_path_phi(self, state: RewardState) -> float:
        """Potential matching CircuitGame/JAX clean_onpath phi."""
        if not self.is_on_path_enabled():
            return 0.0

        if self._route_mode() == "best_route_phi":
            if self.config.on_path_phi_mode == "count":
                return self._best_route_count_phi(state)
            if self.config.on_path_phi_mode == "max_step":
                return self._best_route_max_step_phi(state)
            if self.config.on_path_phi_mode == "depth_weighted":
                return self._best_route_depth_weighted_phi(state)
            raise ValueError(f"Unknown on_path_phi_mode: {self.config.on_path_phi_mode}")

        if self.config.on_path_phi_mode == "count":
            if self.on_path_total <= 0:
                return 0.0
            return state.on_path_count / self.on_path_total
        if self.config.on_path_phi_mode == "max_step":
            if self.on_path_target_step <= 0:
                return 0.0
            return state.on_path_deepest_step / self.on_path_target_step
        if self.config.on_path_phi_mode == "depth_weighted":
            return self._union_depth_weighted_phi(state)
        raise ValueError(f"Unknown on_path_phi_mode: {self.config.on_path_phi_mode}")

    def _record_on_path_hit(
        self,
        new_poly: FastPoly,
        state: RewardState,
    ) -> tuple[bool, RewardState]:
        key = new_poly.canonical_key()
        if key not in self.on_path_steps or key in state.on_path_hit_keys:
            return False, state

        route_mask = int(self.on_path_route_masks.get(key, ALL_ROUTE_BITS))
        active_mask = state.on_path_active_route_mask
        if self._route_mode() == "lock_on_first_hit" and not (route_mask & active_mask):
            return False, state

        next_active_mask = active_mask
        if self._route_mode() == "lock_on_first_hit":
            next_active_mask &= route_mask

        next_keys = frozenset((*state.on_path_hit_keys, key))
        next_state = RewardState(
            on_path_hit_keys=next_keys,
            on_path_count=state.on_path_count + 1,
            on_path_deepest_step=max(
                state.on_path_deepest_step,
                int(self.on_path_steps[key]),
            ),
            on_path_active_route_mask=next_active_mask,
        )
        return True, next_state

    def step_reward(
        self,
        nodes_before: List[FastPoly],
        new_poly: FastPoly,
        state: RewardState,
        *,
        is_terminal: bool = False,
    ) -> RewardResult:
        """Compute clean reward for appending new_poly to nodes_before."""
        is_success = new_poly == self.target
        reward = float(self.config.step_penalty)
        if is_success:
            if self.config.reward_mode == "legacy":
                reward += float(self.config.success_reward)
            else:
                reward += float(self.config.terminal_success_reward)

        phi_before = self.on_path_phi(state)
        on_path_hit = False
        next_state = state
        phi_after = phi_before

        if self.is_on_path_enabled():
            on_path_hit, next_state = self._record_on_path_hit(new_poly, state)
            phi_after = self.on_path_phi(next_state)
            phi_after_for_reward = (
                0.0
                if self.config.on_path_terminal_zero and is_terminal
                else phi_after
            )
            reward += float(self.config.graph_onpath_shaping_coeff) * (
                float(self.config.gamma) * phi_after_for_reward - phi_before
            )
            delta_phi_pos = max(0.0, phi_after - phi_before)
            reward += float(self.config.on_route_bonus_coeff) * delta_phi_pos

        return RewardResult(
            reward=float(reward),
            next_state=next_state,
            on_path_hit=on_path_hit,
            phi_before=float(phi_before),
            phi_after=float(phi_after),
            success=bool(is_success),
        )
