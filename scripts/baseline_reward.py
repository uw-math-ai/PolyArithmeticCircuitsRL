#!/usr/bin/env python3
"""Shared reward utilities for non-learning search baselines.

The richer baseline reward mirrors the cheap parts of CircuitGame's shaped
reward while staying per-target and stateless across evaluation examples:

  - step penalty and success reward
  - PBRS term-similarity shaping
  - one-time rewards for target factor subgoals
  - one-time additive/multiplicative completion bonuses

It intentionally does not use cross-target library memory or learned values.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import FrozenSet, List, Tuple

import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Config  # noqa: E402
from src.environment.factor_library import FactorLibrary  # noqa: E402
from src.environment.fast_polynomial import FastPoly  # noqa: E402

Action = Tuple[int, int, int]


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


@dataclass(frozen=True)
class RewardState:
    """Per-trajectory one-time reward bookkeeping."""

    subgoals_hit: FrozenSet[bytes] = frozenset()
    additive_complete_hit: bool = False
    mult_complete_hit: bool = False


@dataclass(frozen=True)
class RewardResult:
    """Reward transition result returned by BaselineRewardEvaluator."""

    reward: float
    next_state: RewardState
    factor_hit: bool
    additive_complete: bool
    mult_complete: bool


class BaselineRewardEvaluator:
    """Per-target rich reward evaluator for pure Python search baselines."""

    def __init__(self, config: Config, target: FastPoly):
        self.config = config
        self.target = target
        self.factor_library = FactorLibrary(
            mod=config.mod,
            n_vars=config.n_variables,
            max_degree=config.effective_max_degree,
        )
        if config.factor_library_enabled:
            factors = self.factor_library.factorize_target(target)
            self.subgoal_keys = {factor.canonical_key() for factor in factors}
        else:
            self.subgoal_keys = set()

    def initial_state(self) -> RewardState:
        return RewardState()

    def best_similarity(self, nodes: List[FastPoly]) -> float:
        return max(node.term_similarity(self.target) for node in nodes)

    def step_reward(
        self,
        nodes_before: List[FastPoly],
        new_poly: FastPoly,
        state: RewardState,
    ) -> RewardResult:
        """Compute rich reward for appending new_poly to nodes_before."""
        is_success = new_poly == self.target
        nodes_after = nodes_before + [new_poly]

        reward = self.config.step_penalty
        if is_success:
            reward += self.config.success_reward
        elif self.config.use_reward_shaping:
            phi_before = self.best_similarity(nodes_before)
            phi_after = self.best_similarity(nodes_after)
            reward += self.config.gamma * phi_after - phi_before

        subgoals_hit = state.subgoals_hit
        additive_complete_hit = state.additive_complete_hit
        mult_complete_hit = state.mult_complete_hit
        factor_hit = False
        additive_complete = False
        mult_complete = False

        if self.config.factor_library_enabled:
            new_key = new_poly.canonical_key()
            if new_key in self.subgoal_keys and new_key not in subgoals_hit:
                subgoals_hit = frozenset((*subgoals_hit, new_key))
                factor_hit = True
                reward += self.config.factor_subgoal_reward

            if not is_success:
                existing_keys = {node.canonical_key() for node in nodes_before}
                residual = self.target - new_poly
                if (
                    not additive_complete_hit
                    and not residual.is_zero()
                    and residual.canonical_key() in existing_keys
                ):
                    additive_complete = True
                    additive_complete_hit = True
                    reward += self.config.completion_bonus

                if not mult_complete_hit:
                    quotient = self.factor_library.exact_quotient(
                        self.target, new_poly
                    )
                    if (
                        quotient is not None
                        and quotient.canonical_key() in existing_keys
                    ):
                        mult_complete = True
                        mult_complete_hit = True
                        reward += self.config.completion_bonus

        next_state = RewardState(
            subgoals_hit=subgoals_hit,
            additive_complete_hit=additive_complete_hit,
            mult_complete_hit=mult_complete_hit,
        )
        return RewardResult(
            reward=float(reward),
            next_state=next_state,
            factor_hit=factor_hit,
            additive_complete=additive_complete,
            mult_complete=mult_complete,
        )
