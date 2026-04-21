from __future__ import annotations

import math
import random
import logging
import warnings
from fractions import Fraction
from typing import Any, Dict, List, Optional, Set, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..core.builder import CircuitBuilder
from ..core.action_codec import (
    ACTION_ADD, ACTION_MUL, ACTION_SET_OUTPUT, ACTION_STOP,
    decode_action, encode_action,
)
from ..core.fingerprints import (
    sample_eval_points, eval_poly_points, EvalPoint,
)
from ..core.poly import Poly, PolyKey, equal, poly_hashkey, is_scalar, sub as poly_sub
from ..core.factor_library import FactorLibrary
from ..config import Config
from .obs import encode_obs
from .samplers import RandomCircuitSampler

log = logging.getLogger(__name__)


class PolyCircuitEnv(gym.Env):
    """
    Goal-conditioned environment for polynomial circuit construction.

    Observation: flat float32 vector + action mask.
    Reward: +1 on solve, -step_cost per ADD/MUL operation.
    """

    metadata = {"render_modes": []}

    def __init__(self, config: Config, target_sampler=None, factor_library: Optional[FactorLibrary] = None):
        super().__init__()
        self.config = config
        self.rng = random.Random(config.seed)
        self.factor_library = factor_library

        if target_sampler is None:
            self.target_sampler = RandomCircuitSampler(
                n_vars=config.n_vars,
                max_steps=config.max_ops,
            )
        else:
            self.target_sampler = target_sampler

        # Fixed eval points for lifetime of this env
        self.eval_points: List[EvalPoint] = sample_eval_points(
            self.rng, config.n_vars, config.m, config.eval_low, config.eval_high,
        )

        obs_dim = config.obs_dim
        act_dim = config.action_dim

        self.observation_space = spaces.Dict({
            "obs": spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32),
            "action_mask": spaces.Box(0, 1, shape=(act_dim,), dtype=np.int8),
        })
        self.action_space = spaces.Discrete(act_dim)

        # Episode state (set in reset)
        self.builder: Optional[CircuitBuilder] = None
        self.target_poly: Optional[Poly] = None
        self.target_evals: Optional[Tuple[Fraction, ...]] = None
        self.steps_left: int = 0
        self.max_ops_this_ep: int = config.max_ops
        self._episode_step_count: int = 0  # total steps taken this episode
        self._episode_step_limit: int = config.max_ops + config.max_nodes + 5

        # Trajectory for HER
        self._trajectory: List[Dict[str, Any]] = []

        # Track whether last action was SET_OUTPUT (to prevent spam)
        self._last_was_set_output = False

        # Simulation mode: skip expensive shaping during MCTS
        self._simulation = False

        # Factor library per-episode state
        self._subgoal_keys: Set[PolyKey] = set()
        self._library_known_keys: Set[PolyKey] = set()
        self._subgoals_hit: Set[PolyKey] = set()
        self._completion_hit: bool = False

        # Oracle mask helper (diagnostic, set externally)
        self._oracle_helper = None  # Optional[OracleMaskHelper]
        self._oracle_empty_warned_targets: Set[PolyKey] = set()

        # Factor shaping depends on SymPy and should not fail silently.
        self._factor_shaping_available = config.factor_shaping_coeff <= 0.0
        if config.factor_shaping_coeff > 0.0:
            try:
                import sympy  # noqa: F401

                self._factor_shaping_available = True
            except ImportError:
                warnings.warn(
                    "factor_shaping_coeff > 0 but sympy is not installed; "
                    "factor shaping is disabled. Install the 'interesting' extra "
                    "to enable it.",
                    stacklevel=2,
                )

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict]:
        """Reset the environment to a new episode.

        Args:
            seed:    Optionally re-seed the internal RNG.
            options: Optional dict with:
                       "max_ops"     — override max operations for this episode
                       "target_poly" — use a specific Poly as the goal instead
                                       of sampling from the target sampler

        Returns:
            (obs_dict, info) where obs_dict has keys "obs" and "action_mask".
        """
        if seed is not None:
            self.rng.seed(seed)
        options = options or {}

        self.max_ops_this_ep = options.get("max_ops", self.config.max_ops)

        # Sample target polynomial (or use one provided via options)
        target_poly = options.get("target_poly")
        if target_poly is None:
            old_max = self.target_sampler.max_steps
            try:
                self.target_sampler.max_steps = self.max_ops_this_ep
                target_poly, _ = self.target_sampler.sample(self.rng)
            finally:
                self.target_sampler.max_steps = old_max

        self.builder = CircuitBuilder(
            self.config.n_vars,
            eval_points=self.eval_points,
            include_const_one=True,
        )
        self.target_poly = target_poly
        self.target_evals = tuple(eval_poly_points(target_poly, self.eval_points))
        self.steps_left = self.max_ops_this_ep
        self._trajectory = []
        self._episode_step_count = 0
        self._episode_step_limit = (
            self.config.max_episode_steps
            if self.config.max_episode_steps is not None
            else self.max_ops_this_ep + self.config.max_nodes + 5
        )

        # Reward shaping: track best eval distance to target across all nodes
        self._best_eval_dist = self._compute_best_eval_dist()
        self._last_was_set_output = False
        self._simulation = False

        # Factor library: compute subgoals for this episode's target
        self._subgoal_keys = set()
        self._library_known_keys = set()
        self._subgoals_hit = set()
        self._completion_hit = False
        if self.factor_library is not None and self.config.factor_library_enabled:
            try:
                factors = self.factor_library.factorize_target(self.target_poly)
                for f in factors:
                    self._subgoal_keys.add(poly_hashkey(f))
                self._library_known_keys = self.factor_library.filter_known(factors)
            except Exception as e:
                log.debug("sympy fallback in %s: %s", __name__, e)

        obs_dict = self._obs_dict()
        return obs_dict, {"solved": False}

    def step(
        self, action: int,
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict]:
        """Execute one action in the circuit environment.

        Reward structure:
          - ADD or MUL:      -step_cost  (default -0.05)
          - SET_OUTPUT:       0.0
          - STOP:             0.0 (triggers truncation)
          - Solve (output matches target): +1.0 additional reward

        Termination:
          - terminated=True when SET_OUTPUT is called and the output poly
            matches the target (solved).
          - truncated=True when steps_left reaches 0 or STOP is chosen.

        Returns:
            (obs_dict, reward, terminated, truncated, info)
        """
        assert self.builder is not None, "Call reset() first"

        self._episode_step_count += 1
        decoded = decode_action(action, self.config.L)
        mask = self._build_mask()

        # Invalid action: small penalty and immediately truncate the episode
        if mask[action] == 0:
            return self._obs_dict(), -1.0, False, True, {"solved": False, "invalid": True}

        terminated = False
        truncated = False
        base_reward = 0.0
        shaping_reward = 0.0
        budget_just_exhausted = False

        if decoded.kind == ACTION_ADD:
            result = self.builder.add_add(decoded.i, decoded.j)
            self.steps_left -= 1
            base_reward = -self.config.step_cost
            budget_just_exhausted = (self.steps_left <= 0)
            if self.config.reward_mode != "sparse":
                shaping_reward += self._shaping_reward(result)
                if not self._simulation:
                    shaping_reward += self._factor_shaping_reward(result)
                if self.config.reward_mode == "full" and not self._simulation:
                    shaping_reward += self._factor_library_reward(result)
            self._last_was_set_output = False
        elif decoded.kind == ACTION_MUL:
            result = self.builder.add_mul(decoded.i, decoded.j)
            self.steps_left -= 1
            base_reward = -self.config.step_cost
            budget_just_exhausted = (self.steps_left <= 0)
            if self.config.reward_mode != "sparse":
                shaping_reward += self._shaping_reward(result)
                if self.config.reward_mode == "full" and not self._simulation:
                    shaping_reward += self._factor_library_reward(result)
            self._last_was_set_output = False
        elif decoded.kind == ACTION_SET_OUTPUT:
            self.builder.set_output(decoded.i)
            self._last_was_set_output = True
        elif decoded.kind == ACTION_STOP:
            truncated = True

        solved = self._is_solved()
        solve_bonus = 1.0 if solved else 0.0
        reward = base_reward + shaping_reward + solve_bonus
        if solved:
            terminated = True
            # Register successful episode nodes in factor library
            if self.factor_library is not None and self.config.factor_library_enabled:
                n_initial = self.config.n_vars + 1
                self.factor_library.register_episode_nodes(
                    self.builder.nodes, n_initial,
                )
        elif self.steps_left <= 0 and not truncated and not budget_just_exhausted:
            truncated = True
        elif self._episode_step_count >= self._episode_step_limit and not terminated:
            # Hard cap: prevent infinite loops from repeated SET_OUTPUT calls
            truncated = True

        obs_dict = self._obs_dict()

        # Store for HER
        self._trajectory.append({
            "obs": obs_dict["obs"].copy(),
            "action_mask": obs_dict["action_mask"].copy(),
            "action": action,
            "reward": reward,
            "base_reward": base_reward,
            "shaping_reward": shaping_reward,
            "solve_bonus": solve_bonus,
            "solved": solved,
            "terminated": terminated,
            "truncated": truncated,
            "achieved_goal": self._achieved_goal(),
            "achieved_goal_key": self._achieved_goal_key(),
            "node_evals": self._all_node_evals(),
        })

        info = {
            "solved": solved,
            "num_nodes": len(self.builder.nodes),
            "steps_left": self.steps_left,
        }
        return obs_dict, reward, terminated, truncated, info

    def get_trajectory(self) -> List[Dict[str, Any]]:
        """Full episode trajectory for HER processing."""
        return self._trajectory

    def get_state(self) -> Dict[str, Any]:
        """Snapshot current env state for MCTS simulation."""
        return {
            "builder": self.builder.clone(),
            "steps_left": self.steps_left,
            "episode_step_count": self._episode_step_count,
            "best_eval_dist": self._best_eval_dist,
            "last_was_set_output": self._last_was_set_output,
            # Factor library per-episode state (shallow copy of sets)
            "subgoal_keys": set(self._subgoal_keys),
            "library_known_keys": set(self._library_known_keys),
            "subgoals_hit": set(self._subgoals_hit),
            "completion_hit": self._completion_hit,
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore env state from snapshot (enters simulation mode)."""
        self.builder = state["builder"]
        self.steps_left = state["steps_left"]
        self._episode_step_count = state["episode_step_count"]
        self._best_eval_dist = state["best_eval_dist"]
        self._last_was_set_output = state["last_was_set_output"]
        self._subgoal_keys = state["subgoal_keys"]
        self._library_known_keys = state["library_known_keys"]
        self._subgoals_hit = state["subgoals_hit"]
        self._completion_hit = state["completion_hit"]
        self._simulation = True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_solved(self) -> bool:
        if self.builder.output_node is None:
            return False
        return equal(
            self.builder.nodes[self.builder.output_node].poly,
            self.target_poly,
        )

    def _obs_dict(self) -> Dict[str, np.ndarray]:
        obs = encode_obs(
            nodes=self.builder.nodes,
            target_evals=self.target_evals,
            eval_points=self.eval_points,
            steps_left=self.steps_left,
            max_ops=self.max_ops_this_ep,
            config=self.config,
        )
        return {"obs": obs, "action_mask": self._build_mask()}

    def _build_mask(self) -> np.ndarray:
        """Build action mask inline (avoids buggy masks.py)."""
        L = self.config.L
        mask = np.zeros(self.config.action_dim, dtype=np.int8)

        num_nodes = len(self.builder.nodes)
        visible = min(num_nodes, L)
        # Do not create hidden nodes that exceed visible observation capacity.
        can_expand = (
            self.steps_left > 0
            and num_nodes < self.config.max_nodes
            and num_nodes < L
        )

        if can_expand and visible > 0:
            for i in range(visible):
                for j in range(i, visible):
                    mask[encode_action(ACTION_ADD, i, j, L)] = 1
                    mask[encode_action(ACTION_MUL, i, j, L)] = 1

        # Only allow SET_OUTPUT if the last action was NOT a SET_OUTPUT
        if not self._last_was_set_output:
            for i in range(visible):
                mask[encode_action(ACTION_SET_OUTPUT, i, None, L)] = 1

        mask[encode_action(ACTION_STOP, None, None, L)] = 1

        # Oracle mask: restrict ADD/MUL to optimal-path actions only
        if self._oracle_helper is not None and not self._simulation and can_expand:
            oracle_actions = self._oracle_helper.get_optimal_actions(
                self.target_poly,
                [node.poly for node in self.builder.nodes[:visible]],
            )
            if oracle_actions is not None:
                target_key = poly_hashkey(self.target_poly)
                if len(oracle_actions) == 0:
                    if target_key not in self._oracle_empty_warned_targets:
                        warnings.warn(
                            "Oracle mask returned no actions; falling back to base action mask.",
                            stacklevel=2,
                        )
                        self._oracle_empty_warned_targets.add(target_key)
                    return mask
                oracle_mask = np.zeros(self.config.action_dim, dtype=np.int8)
                for op, i, j in oracle_actions:
                    kind = ACTION_ADD if op == "add" else ACTION_MUL
                    oracle_mask[encode_action(kind, i, j, L)] = 1
                # Keep SET_OUTPUT and STOP from the original mask
                if not self._last_was_set_output:
                    for i in range(visible):
                        oracle_mask[encode_action(ACTION_SET_OUTPUT, i, None, L)] = 1
                oracle_mask[encode_action(ACTION_STOP, None, None, L)] = 1
                # Intersect: only allow actions valid in BOTH masks
                combined = mask & oracle_mask
                # Fallback: if no action survives, keep original mask and warn once per target.
                if combined.sum() > 0:
                    mask = combined
                elif target_key not in self._oracle_empty_warned_targets:
                    warnings.warn(
                        "Oracle mask eliminated all actions; falling back to base action mask.",
                        stacklevel=2,
                    )
                    self._oracle_empty_warned_targets.add(target_key)

        return mask

    def _factor_library_reward(self, result) -> float:
        """Factor library subgoal and completion rewards for a new node."""
        if (
            self.factor_library is None
            or not self.config.factor_library_enabled
            or result.reused
        ):
            return 0.0

        node = self.builder.nodes[result.node_id]
        new_key = poly_hashkey(node.poly)
        reward = 0.0

        # Subgoal match: new node matches a factor of the target
        if new_key in self._subgoal_keys and new_key not in self._subgoals_hit:
            reward += self.config.factor_subgoal_reward
            self._subgoals_hit.add(new_key)
            # Library bonus: subgoal was previously built in a past episode
            if new_key in self._library_known_keys:
                reward += self.config.factor_library_bonus

        # Completion bonuses: check if one operation away from target
        existing_keys = {
            poly_hashkey(n.poly) for n in self.builder.nodes
        }

        # Additive completion: T - v_new already in circuit
        if not self._completion_hit:
            try:
                residual = poly_sub(self.target_poly, node.poly)
                residual_key = poly_hashkey(residual)
                if residual_key in existing_keys:
                    reward += self.config.completion_bonus
                    self._completion_hit = True
            except Exception as e:
                log.debug("sympy fallback in %s: %s", __name__, e)

        # Dynamic subgoal discovery when node is library-known
        if self.factor_library.contains(node.poly):
            # Additive subgoal: T - v_new
            try:
                residual = poly_sub(self.target_poly, node.poly)
                if residual:
                    res_key = poly_hashkey(residual)
                    if res_key not in self._subgoal_keys and not self.factor_library.is_base(residual):
                        self._subgoal_keys.add(res_key)
                    # Also factorize the residual
                    new_factors = self.factor_library.factorize_poly(
                        residual, exclude_keys=self._subgoal_keys,
                    )
                    for f in new_factors:
                        self._subgoal_keys.add(poly_hashkey(f))
            except Exception as e:
                log.debug("sympy fallback in %s: %s", __name__, e)

            # Multiplicative subgoal: T / v_new
            try:
                quotient = self.factor_library.exact_quotient(
                    self.target_poly, node.poly,
                )
                if quotient is not None:
                    q_key = poly_hashkey(quotient)
                    # Multiplicative completion: quotient is scalar or in circuit
                    if not self._completion_hit:
                        if is_scalar(quotient) or q_key in existing_keys:
                            reward += self.config.completion_bonus
                            self._completion_hit = True
                    if q_key not in self._subgoal_keys and not self.factor_library.is_base(quotient):
                        self._subgoal_keys.add(q_key)
                    # Factorize quotient
                    q_factors = self.factor_library.factorize_poly(
                        quotient, exclude_keys=self._subgoal_keys,
                    )
                    for f in q_factors:
                        self._subgoal_keys.add(poly_hashkey(f))
            except Exception as e:
                log.debug("sympy fallback in %s: %s", __name__, e)

        return reward

    def _compute_best_eval_dist(self) -> float:
        """Minimum L1 eval-distance between any circuit node and the target."""
        best = float("inf")
        for node in self.builder.nodes:
            if node.evals is not None:
                evals = node.evals
            else:
                evals = eval_poly_points(node.poly, self.eval_points)
            dist = sum(abs(float(a) - float(b)) for a, b in zip(evals, self.target_evals))
            best = min(best, dist)
        return best

    def _factor_shaping_reward(self, result) -> float:
        """Penalize ADD when the result is factorizable (should have used MUL)."""
        if (
            self.config.factor_shaping_coeff <= 0
            or result.reused
            or not self._factor_shaping_available
        ):
            return 0.0
        node = self.builder.nodes[result.node_id]
        poly = node.poly
        if len(poly) <= 1:
            return 0.0
        try:
            from sympy import factor, expand, symbols as sym_symbols
            from functools import reduce
            n = self.config.n_vars
            var_syms = (
                [sym_symbols("x")]
                if n == 1
                else list(sym_symbols(f"x0:{n}"))
            )
            # Convert internal Poly dict to SymPy expression
            terms = []
            for mono, coeff in poly.items():
                term = int(coeff)
                for v, e in zip(var_syms, mono):
                    if e > 0:
                        term = term * v ** e
                terms.append(term)
            expr = reduce(lambda a, b: a + b, terms)
            factored = factor(expr)
            if factored != expand(expr):
                return -self.config.factor_shaping_coeff
        except Exception as e:
            log.debug("sympy fallback in %s: %s", __name__, e)
        return 0.0

    def _shaping_reward(self, result) -> float:
        """Eval-distance reward shaping for a newly created node."""
        if self.config.shaping_coeff <= 0 or result.reused:
            return 0.0
        node = self.builder.nodes[result.node_id]
        if node.evals is not None:
            evals = node.evals
        else:
            evals = eval_poly_points(node.poly, self.eval_points)
        new_dist = sum(abs(float(a) - float(b)) for a, b in zip(evals, self.target_evals))
        if new_dist < self._best_eval_dist:
            progress = (self._best_eval_dist - new_dist) / (self._best_eval_dist + 1e-10)
            self._best_eval_dist = new_dist
            return self.config.shaping_coeff * progress
        return 0.0

    def _all_node_evals(self) -> List[np.ndarray]:
        """Eval vectors for all current nodes (used by HER).

        Values are tanh-normalised (same transform as obs encoding) so that
        HER goals are in the same space as the target eval vector in obs.
        """
        scale = self.config.eval_norm_scale
        result = []
        for node in self.builder.nodes:
            raw = node.evals if node.evals is not None else eval_poly_points(node.poly, self.eval_points)
            if scale > 0.0:
                arr = np.array([math.tanh(float(e) / scale) for e in raw], dtype=np.float32)
            else:
                arr = np.array([float(e) for e in raw], dtype=np.float32)
            result.append(arr)
        return result

    def _achieved_goal(self) -> Optional[np.ndarray]:
        """Return the current output-node goal in normalized goal space."""
        if self.builder.output_node is None:
            return None

        node = self.builder.nodes[self.builder.output_node]
        raw = node.evals if node.evals is not None else eval_poly_points(node.poly, self.eval_points)
        scale = self.config.eval_norm_scale
        if scale > 0.0:
            return np.array([math.tanh(float(e) / scale) for e in raw], dtype=np.float32)
        return np.array([float(e) for e in raw], dtype=np.float32)

    def _achieved_goal_key(self) -> Optional[PolyKey]:
        """Exact achieved-goal identity key for HER solved gating."""
        if self.builder.output_node is None:
            return None
        return poly_hashkey(self.builder.nodes[self.builder.output_node].poly)
