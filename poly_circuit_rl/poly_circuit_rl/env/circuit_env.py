from __future__ import annotations

import math
import random
from fractions import Fraction
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..core.builder import CircuitBuilder
from ..core.action_codec import (
    ACTION_ADD, ACTION_MUL, ACTION_SET_OUTPUT, ACTION_STOP,
    action_space_size, decode_action, encode_action,
)
from ..core.fingerprints import (
    sample_eval_points, eval_poly_points, EvalPoint,
)
from ..core.poly import Poly, equal
from ..config import Config
from .obs import encode_obs
from .samplers import RandomCircuitSampler


class PolyCircuitEnv(gym.Env):
    """
    Goal-conditioned environment for polynomial circuit construction.

    Observation: flat float32 vector + action mask.
    Reward: +1 on solve, -step_cost per ADD/MUL operation.
    """

    metadata = {"render_modes": []}

    def __init__(self, config: Config, target_sampler=None):
        super().__init__()
        self.config = config
        self.rng = random.Random(config.seed)

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
            self.target_sampler.max_steps = self.max_ops_this_ep
            target_poly, _ = self.target_sampler.sample(self.rng)
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
        reward = 0.0
        budget_just_exhausted = False

        if decoded.kind == ACTION_ADD:
            result = self.builder.add_add(decoded.i, decoded.j)
            self.steps_left -= 1
            reward = -self.config.step_cost
            budget_just_exhausted = (self.steps_left <= 0)
            reward += self._shaping_reward(result)
        elif decoded.kind == ACTION_MUL:
            result = self.builder.add_mul(decoded.i, decoded.j)
            self.steps_left -= 1
            reward = -self.config.step_cost
            budget_just_exhausted = (self.steps_left <= 0)
            reward += self._shaping_reward(result)
        elif decoded.kind == ACTION_SET_OUTPUT:
            self.builder.set_output(decoded.i)
        elif decoded.kind == ACTION_STOP:
            truncated = True

        solved = self._is_solved()
        if solved:
            reward += 1.0
            terminated = True
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
            "terminated": terminated,
            "truncated": truncated,
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
        can_expand = self.steps_left > 0 and num_nodes < self.config.max_nodes

        if can_expand and visible > 0:
            for i in range(visible):
                for j in range(i, visible):
                    mask[encode_action(ACTION_ADD, i, j, L)] = 1
                    mask[encode_action(ACTION_MUL, i, j, L)] = 1

        for i in range(visible):
            mask[encode_action(ACTION_SET_OUTPUT, i, None, L)] = 1

        mask[encode_action(ACTION_STOP, None, None, L)] = 1
        return mask

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
