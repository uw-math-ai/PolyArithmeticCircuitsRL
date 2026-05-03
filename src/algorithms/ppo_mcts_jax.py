"""MCTS-guided policy/value trainer using JAX and mctx.

This implements the same Expert Iteration loop as ppo_mcts.py but with
batched MCTS via Google DeepMind's mctx library. All B environments are
searched in parallel on GPU using either mctx.muzero_policy (PUCT) or
mctx.gumbel_muzero_policy (Gumbel MuZero), giving a massive speedup over the
sequential PyTorch implementation.

Architecture:
  - Environment states are pure JAX arrays (jax_env.EnvState).
  - Policy-value network is Flax (jax_net.PolicyValueNet).
    - Search uses mctx with a learned dynamics model replaced by the true
        environment dynamics (perfect simulator).
    - Policy/value updates use optax for gradient clipping and Adam.

Flow per iteration:
  1. Reset B parallel environments with sampled targets.
  2. For each step (up to max_steps):
     a. Batch-evaluate policy+value for all B states.
    b. Run the configured mctx search policy for all B states in parallel.
     c. Execute chosen actions in all B environments.
     d. Store transitions.
  3. Compute GAE advantages.
    4. Distill the MCTS visit distribution and fit values.
  5. Adjust curriculum.
"""

import functools
import math
import os
import pickle
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import List, Optional, Tuple

import jax
import jax.numpy as jnp
import mctx
import numpy as np
import optax
import flax.linen as nn
from flax.core import freeze, unfreeze
from flax.core.frozen_dict import FrozenDict
from flax.training import train_state
from tqdm import tqdm

from ..config import Config
from ..environment.fast_polynomial import FastPoly
from ..environment.factor_library import FactorLibrary
from ..game_board.generator import generate_random_circuit
from .jax_env import (
    EnvConfig, EnvState, make_env_config,
    reset as env_reset, step as env_step,
    get_observation, make_empty_subgoal_arrays,
)
from .jax_net import PolicyValueNet, create_network, init_params


@dataclass
class Transition:
    """Single transition for PPO update."""
    obs: dict           # JAX observation arrays
    env_index: int
    action: int
    reward: float
    policy_target: np.ndarray
    value: float
    done: bool


class PPOMCTSJAXTrainer:
    """PPO + batched MCTS trainer using JAX/mctx.

    Runs B parallel environments, uses mctx.muzero_policy for batched tree
    search, and trains with PPO using optax.

    Supports training on multiple fixed complexities simultaneously by splitting
    the batch across complexities. When ``fixed_complexities`` is provided,
    curriculum learning is disabled and targets are sampled uniformly from each
    specified complexity level.

    Attributes:
        config: Shared hyperparameter configuration.
        env_config: JAX-compatible environment config.
        network: Flax policy-value network.
        tx: optax optimizer.
        train_state: Flax TrainState with params and optimizer state.
        batch_size: Number of parallel environments for data collection.
        fixed_complexities: If set, train on these complexities in parallel
            (disables curriculum).
    """

    def __init__(self, config: Config, device: str = "cpu",
                 batch_size: int = 256,
                 fixed_complexities: Optional[List[int]] = None) -> None:
        """Initialize the JAX PPO+MCTS trainer.

        Args:
            config: Configuration dataclass with all hyperparameters.
            device: Ignored (JAX auto-detects GPU/TPU).
            batch_size: Number of parallel environments during rollouts.
            fixed_complexities: List of complexity levels to train on in
                parallel (e.g. [5, 6, 7, 8]). Disables curriculum when set.
                The batch is split evenly across these levels.
        """
        self.config = config
        if config.search not in {"puct", "gumbel"}:
            raise ValueError(f"Unsupported search backend: {config.search}")
        self.fixed_complexities = fixed_complexities
        if fixed_complexities:
            config.curriculum_enabled = False
        self.env_config = make_env_config(config)
        self.batch_size = batch_size
        self.library_cache_size = 512

        # Network.
        self.network = create_network(config)
        rng = jax.random.PRNGKey(config.seed)
        params = init_params(self.network, self.env_config, rng)

        # Optimizer with gradient clipping.
        self.tx = optax.chain(
            optax.clip_by_global_norm(config.max_grad_norm),
            optax.adam(config.ppo_lr),
        )
        self.train_state = train_state.TrainState.create(
            apply_fn=self.network.apply,
            params=params,
            tx=self.tx,
        )

        # Curriculum state (only used when fixed_complexities is None).
        self.current_complexity = (
            config.starting_complexity if config.curriculum_enabled
            else config.max_complexity
        )
        self.success_history: List[bool] = []

        factor_library: Optional[FactorLibrary] = None
        if config.factor_library_enabled:
            factor_library = FactorLibrary(
                mod=config.mod,
                n_vars=config.n_variables,
                max_degree=config.effective_max_degree,
            )
        self.factor_library = factor_library

        if config.search == "gumbel" and config.gumbel_root_only:
            print(
                "JAX Gumbel search uses full mctx Gumbel MuZero; "
                "gumbel_root_only is ignored on this backend."
            )

        # JIT-compile core functions.
        self._jit_apply = jax.jit(self.network.apply)
        self._jit_env_step = jax.jit(
            functools.partial(env_step, self.env_config)
        )
        self._jit_env_reset = jax.jit(
            functools.partial(env_reset, self.env_config)
        )
        self._jit_get_obs = jax.jit(
            functools.partial(get_observation, self.env_config)
        )
        self._jit_batched_mcts = jax.jit(self._batched_mcts_search)

    def _tree_all_finite(self, tree) -> bool:
        """Return True when every floating-point leaf in a pytree is finite."""

        def leaf_is_finite(x):
            arr = np.asarray(x)
            if arr.dtype.kind == 'f':
                return bool(np.isfinite(arr).all())
            return True

        return bool(
            jax.tree_util.tree_all(
                jax.tree_util.tree_map(leaf_is_finite, tree)
            )
        )

    def _assert_finite_train_state(self, context: str) -> None:
        """Fail fast instead of silently continuing with corrupted parameters."""
        if not self._tree_all_finite(self.train_state.params):
            raise RuntimeError(f"Non-finite model parameters detected {context}")
        if not self._tree_all_finite(self.train_state.opt_state):
            raise RuntimeError(f"Non-finite optimizer state detected {context}")

    def _sanitize_policy_target(
        self,
        action_weights: np.ndarray,
        valid_mask: np.ndarray,
        sampled_action: int,
    ) -> np.ndarray:
        """Return a finite, normalized policy target over valid actions only."""
        target = np.asarray(action_weights, dtype=np.float32).copy()
        mask = np.asarray(valid_mask, dtype=bool)
        if target.shape != mask.shape:
            raise RuntimeError(
                f"Policy target shape {target.shape} did not match mask {mask.shape}"
            )

        target[~mask] = 0.0
        total = float(target.sum()) if np.isfinite(target).all() else float("nan")
        if not math.isfinite(total) or total <= 0.0:
            target.fill(0.0)
            if 0 <= sampled_action < target.shape[0] and mask[sampled_action]:
                target[sampled_action] = 1.0
            else:
                valid_indices = np.flatnonzero(mask)
                if valid_indices.size == 0:
                    raise RuntimeError("Encountered state with no valid actions")
                target[valid_indices] = 1.0 / valid_indices.size
            return target

        target /= total
        return target

    def _search_num_simulations(self) -> int:
        """Return the configured number of search simulations."""
        if self.config.search == "gumbel":
            return self.config.gumbel_num_simulations
        return self.config.mcts_simulations

    def _gumbel_qtransform(self):
        """Return the configured q-transform for mctx Gumbel MuZero."""
        if not self.config.gumbel_use_completed_q:
            return functools.partial(mctx.qtransform_by_parent_and_siblings)

        return functools.partial(
            mctx.qtransform_completed_by_mix_value,
            value_scale=self.config.gumbel_c_scale,
            maxvisit_init=self.config.gumbel_c_visit,
            rescale_values=self.config.gumbel_q_normalize,
            use_mixed_value=self.config.gumbel_use_mixed_value,
        )

    def _extract_gumbel_policy_target(self, policy_output, search_summary):
        """Choose which Gumbel root policy target to train against."""
        if self.config.gumbel_policy_target == "visits":
            return search_summary.visit_probs
        return policy_output.action_weights

    def _extract_gumbel_search_stats(self, policy_output, obs_batch, search_summary):
        """Compute per-environment Gumbel root diagnostics from the search tree."""
        qtransform = self._gumbel_qtransform()
        root_completed_q = jax.vmap(
            qtransform, in_axes=[0, None]
        )(policy_output.search_tree, policy_output.search_tree.ROOT_INDEX)
        target_policy = self._extract_gumbel_policy_target(policy_output, search_summary)
        valid_mask = obs_batch["mask"]
        clipped_policy = jnp.where(valid_mask, target_policy, 0.0)
        policy_entropy = -jnp.sum(
            clipped_policy * jnp.log(jnp.clip(clipped_policy, 1e-8, 1.0)),
            axis=-1,
        )

        masked_q = jnp.where(valid_mask, root_completed_q, jnp.nan)
        selected_visits = jnp.take_along_axis(
            search_summary.visit_counts,
            policy_output.action[:, None],
            axis=-1,
        ).squeeze(-1)

        return {
            "policy_entropy": policy_entropy,
            "considered_actions": jnp.sum(search_summary.visit_counts > 0, axis=-1),
            "selected_action_visit_count": selected_visits,
            "max_root_q": jnp.nanmax(masked_q, axis=-1),
            "min_root_q": jnp.nanmin(masked_q, axis=-1),
        }

    # ------------------------------------------------------------------
    # MCTS via mctx
    # ------------------------------------------------------------------

    def _root_fn(self, params, rng_key, obs_batch):
        """Compute root node values for mctx.

        Args:
            params: Network parameters.
            rng_key: JAX PRNG key.
            obs_batch: Batched observation dict, each leaf has shape (B, ...).

        Returns:
            mctx.RootFnOutput with prior_logits, value, embedding (=state).
        """
        # vmap the network over batch dimension.
        logits, values = jax.vmap(
            lambda obs: self.network.apply(params, obs)
        )(obs_batch)
        # logits: (B, max_actions), values: (B,)

        return mctx.RootFnOutput(
            prior_logits=logits,
            value=values,
            embedding=obs_batch,  # Pass full obs as embedding for recurrent_fn.
        )

    def _recurrent_fn(self, params, rng_key, action, embedding,
                      library_coeffs, library_mask):
        """Simulate one environment step for mctx tree expansion.

        This is the "dynamics model" — since we have a perfect simulator,
        we use the actual environment step function.

        Args:
            params: Network parameters.
            rng_key: JAX PRNG key.
            action: (B,) int32 actions.
            embedding: Dict of batched env state arrays from the parent node.

        Returns:
            (mctx.RecurrentFnOutput, new_embedding)
        """
        # We need to carry the EnvState through the tree. The embedding
        # stores the full state needed to reconstruct observations.
        # For mctx, embedding is whatever we pass — here it's the obs dict
        # plus hidden env state. We use a combined state dict.

        env_states = embedding['_env_state']  # Batched EnvState

        # vmap env_step over batch.
        (
            next_states, rewards, dones, successes,
            _factor_hits, _library_hits, _additive_complete, _mult_complete
        ) = jax.vmap(
            lambda s, a: env_step(
                self.env_config, s, a, library_coeffs, library_mask
            )
        )(env_states, action)

        # Get observations for next states.
        next_obs = jax.vmap(
            lambda s: get_observation(self.env_config, s)
        )(next_states)

        # Evaluate network at next states.
        logits, values = jax.vmap(
            lambda obs: self.network.apply(params, obs)
        )(next_obs)

        # Discount: 0 if done, gamma otherwise.
        discount = jnp.where(dones, 0.0, self.config.gamma)

        # Build new embedding (carries state forward in tree).
        new_embedding = {**next_obs, '_env_state': next_states}

        recurrent_output = mctx.RecurrentFnOutput(
            reward=rewards,
            discount=discount,
            prior_logits=logits,
            value=values,
        )
        return recurrent_output, new_embedding

    def _batched_mcts_search(self, params, rng_key, obs_batch, env_states,
                             library_coeffs, library_mask):
        """Run batched search using the configured mctx backend.

        Args:
            params: Network parameters.
            rng_key: JAX PRNG key.
            obs_batch: Batched observations, each leaf shape (B, ...).
            env_states: Batched EnvState for dynamics simulation.

        Returns:
            mctx.PolicyOutput with action_weights (B, max_actions).
        """
        # Prepare root with env state embedded.
        embedding_with_state = {**obs_batch, '_env_state': env_states}

        root = self._root_fn(params, rng_key, obs_batch)
        # Override embedding to include env state.
        root = mctx.RootFnOutput(
            prior_logits=root.prior_logits,
            value=root.value,
            embedding=embedding_with_state,
        )

        # Invalid action masking: mctx uses -inf in prior_logits.
        # Our logits already have -1e9 for invalid actions, which works.

        recurrent_fn = functools.partial(
            self._recurrent_fn,
            library_coeffs=library_coeffs,
            library_mask=library_mask,
        )

        if self.config.search == "gumbel":
            policy_output = mctx.gumbel_muzero_policy(
                params=params,
                rng_key=rng_key,
                root=root,
                recurrent_fn=recurrent_fn,
                num_simulations=self._search_num_simulations(),
                max_depth=self.config.max_steps,
                invalid_actions=~obs_batch['mask'],
                qtransform=self._gumbel_qtransform(),
                max_num_considered_actions=self.config.gumbel_max_num_considered_actions,
                gumbel_scale=self.config.gumbel_scale,
            )
        else:
            policy_output = mctx.muzero_policy(
                params=params,
                rng_key=rng_key,
                root=root,
                recurrent_fn=recurrent_fn,
                num_simulations=self._search_num_simulations(),
                max_depth=self.config.max_steps,
                invalid_actions=~obs_batch['mask'],  # True = invalid
                temperature=self.config.temperature_init,
            )
        return policy_output

    def _key_to_fastpoly(self, key: bytes) -> FastPoly:
        """Reconstruct a FastPoly from a canonical key."""
        shape = (self.config.effective_max_degree + 1,) * self.config.n_variables
        coeffs = np.frombuffer(key, dtype=np.int64).copy().reshape(shape)
        return FastPoly(coeffs, self.config.mod)

    def _export_library_cache(self):
        """Export the session factor library as a dense JAX cache."""
        if self.factor_library is None or len(self.factor_library) == 0:
            return (
                jnp.zeros((self.library_cache_size, self.env_config.target_size), dtype=jnp.int32),
                jnp.zeros((self.library_cache_size,), dtype=jnp.bool_),
            )

        items = sorted(
            self.factor_library._known.items(),
            key=lambda kv: (kv[1], len(kv[0])),
        )[: self.library_cache_size]
        coeff_rows = []
        for key, _step_num in items:
            poly = self._key_to_fastpoly(key)
            coeff_rows.append(self._fastpoly_to_jax(poly))

        lib_coeffs = np.zeros(
            (self.library_cache_size, self.env_config.target_size), dtype=np.int32
        )
        lib_mask = np.zeros((self.library_cache_size,), dtype=bool)
        if coeff_rows:
            arr = np.stack([np.array(r, dtype=np.int32) for r in coeff_rows], axis=0)
            lib_coeffs[: arr.shape[0]] = arr
            lib_mask[: arr.shape[0]] = True
        return jnp.array(lib_coeffs), jnp.array(lib_mask)

    def _prepare_initial_subgoals(self, targets: List[FastPoly]):
        """Factorize targets on the host and build fixed-size JAX subgoal arrays."""
        B = len(targets)
        max_subgoals = self.env_config.max_subgoals
        target_size = self.env_config.target_size

        coeffs = np.zeros((B, max_subgoals, target_size), dtype=np.int32)
        active = np.zeros((B, max_subgoals), dtype=bool)
        library_known = np.zeros((B, max_subgoals), dtype=bool)

        if self.factor_library is None or not self.config.factor_library_enabled:
            return jnp.array(coeffs), jnp.array(active), jnp.array(library_known)

        for i, target in enumerate(targets):
            factors = self.factor_library.factorize_target(target)
            known = self.factor_library.filter_known(factors)
            for j, factor in enumerate(factors[:max_subgoals]):
                coeffs[i, j] = np.array(self._fastpoly_to_jax(factor), dtype=np.int32)
                active[i, j] = True
                library_known[i, j] = factor.canonical_key() in known

        return jnp.array(coeffs), jnp.array(active), jnp.array(library_known)

    def _state_to_fastpolys(self, state_np) -> List[FastPoly]:
        """Convert a single JAX env state snapshot to the node list needed by FactorLibrary."""
        num_nodes = int(state_np.num_nodes)
        shape = (self.config.effective_max_degree + 1,) * self.config.n_variables
        nodes = []
        for idx in range(num_nodes):
            coeffs = np.array(state_np.node_coeffs[idx], dtype=np.int64).reshape(shape)
            nodes.append(FastPoly(coeffs, self.config.mod))
        return nodes

    # ------------------------------------------------------------------
    # Data collection
    # ------------------------------------------------------------------

    def _sample_one_target(self, complexity: int) -> FastPoly:
        """Sample a target polynomial by generating a random circuit.

        Uses ``generate_random_circuit`` which is O(complexity) — instant
        compared to the BFS game board which is combinatorially explosive
        for complexity >= 4.  The true minimum complexity of the resulting
        target may be lower than ``complexity``, but this is fine for
        training: the agent still needs to discover a valid circuit.
        """
        poly, _ = generate_random_circuit(self.config, complexity)
        return poly

    def _sample_targets_multi(self, n: int) -> Tuple[List[FastPoly], List[int]]:
        """Sample n target polynomials, distributing across complexities.

        When ``fixed_complexities`` is set, the batch is split evenly across
        all specified complexity levels. Otherwise falls back to the single
        current_complexity used by curriculum.

        Args:
            n: Total number of targets to sample.

        Returns:
            (targets, complexity_labels) — lists of length n.
        """
        if self.fixed_complexities:
            targets = []
            labels = []
            per_c = n // len(self.fixed_complexities)
            remainder = n % len(self.fixed_complexities)
            for idx, c in enumerate(self.fixed_complexities):
                count = per_c + (1 if idx < remainder else 0)
                for _ in range(count):
                    targets.append(self._sample_one_target(c))
                    labels.append(c)
            return targets, labels
        else:
            targets = []
            for _ in range(n):
                targets.append(self._sample_one_target(self.current_complexity))
            return targets, [self.current_complexity] * n

    def _fastpoly_to_jax(self, poly: FastPoly) -> jnp.ndarray:
        """Convert FastPoly to flat JAX int32 coefficient array."""
        return jnp.array(poly.coeffs.flatten(), dtype=jnp.int32)

    def collect_rollouts(self) -> Tuple[List[Transition], dict]:
        """Collect rollout data using batched MCTS for action selection.

        Runs self.batch_size parallel episodes. At each step, all B
        environments are searched simultaneously via mctx. When
        ``fixed_complexities`` is set, the batch is split across complexity
        levels and per-complexity metrics are tracked.

        Returns:
            (transitions, rollout_info) where transitions is a flat list
            and rollout_info has episode statistics (including per-complexity
            breakdowns when using fixed_complexities).
        """
        B = self.batch_size
        params = self.train_state.params
        rng = jax.random.PRNGKey(
            np.random.randint(0, 2**31)
        )

        # Sample targets and reset environments.
        targets, complexity_labels = self._sample_targets_multi(B)
        complexity_labels_np = np.array(complexity_labels)
        target_arrays = jnp.stack(
            [self._fastpoly_to_jax(t) for t in targets], axis=0
        )  # (B, target_size)
        subgoal_coeffs, subgoal_active, subgoal_library_known = (
            self._prepare_initial_subgoals(targets)
        )
        library_coeffs, library_mask = self._export_library_cache()

        # Batch reset.
        states = jax.vmap(
            lambda tc, sgc, sga, sgl: env_reset(self.env_config, tc, sgc, sga, sgl)
        )(target_arrays, subgoal_coeffs, subgoal_active, subgoal_library_known)

        transitions = []
        episode_rewards = np.zeros(B)
        episode_successes = np.zeros(B, dtype=bool)
        active = np.ones(B, dtype=bool)  # Track which envs are still running.
        factor_hits = 0
        library_hits = 0
        successful_final_states = []
        gumbel_policy_entropies = []
        gumbel_considered_actions = []
        gumbel_selected_visits = []
        gumbel_max_root_q = []
        gumbel_min_root_q = []

        for step_idx in range(self.config.max_steps):
            if not active.any():
                break

            # Get observations for all envs.
            obs_batch = jax.vmap(
                lambda s: get_observation(self.env_config, s)
            )(states)

            # Run batched MCTS.
            rng, search_rng = jax.random.split(rng)
            policy_output = self._jit_batched_mcts(
                params, search_rng, obs_batch, states,
                library_coeffs, library_mask,
            )
            search_summary = policy_output.search_tree.summary()

            if self.config.search == "gumbel":
                action_weights = self._extract_gumbel_policy_target(
                    policy_output, search_summary
                )
                actions = policy_output.action
                gumbel_stats = self._extract_gumbel_search_stats(
                    policy_output, obs_batch, search_summary
                )
            else:
                # Sample actions from MCTS visit counts (with temperature).
                action_weights = policy_output.action_weights  # (B, max_actions)
                rng, sample_rng = jax.random.split(rng)
                actions = jax.random.categorical(
                    sample_rng, jnp.log(action_weights + 1e-8)
                )  # (B,)
                gumbel_stats = None

            # Evaluate the network at the current states.
            _logits, values = jax.vmap(
                lambda obs: self.network.apply(params, obs)
            )(obs_batch)

            # Step all environments.
            (
                next_states, rewards, dones, successes,
                factor_hits_step, library_hits_step,
                _additive_complete, _mult_complete,
            ) = jax.vmap(
                lambda s, a: env_step(
                    self.env_config, s, a, library_coeffs, library_mask
                )
            )(states, actions)

            # Convert to numpy for storage.
            actions_np = np.array(actions)
            rewards_np = np.array(rewards)
            dones_np = np.array(dones)
            successes_np = np.array(successes)
            factor_hits_np = np.array(factor_hits_step)
            library_hits_np = np.array(library_hits_step)
            values_np = np.array(values)
            action_weights_np = np.array(action_weights, dtype=np.float32)
            masks_np = np.array(obs_batch['mask'], dtype=bool)
            if gumbel_stats is not None:
                gumbel_stats_np = jax.tree.map(np.array, gumbel_stats)
            else:
                gumbel_stats_np = None

            # Store transitions for active environments.
            for i in range(B):
                if active[i]:
                    # Extract single-env observation (numpy-ify for storage).
                    obs_i = jax.tree.map(lambda x: np.array(x[i]), obs_batch)
                    transitions.append(Transition(
                        obs=obs_i,
                        env_index=i,
                        action=int(actions_np[i]),
                        reward=float(rewards_np[i]),
                        policy_target=self._sanitize_policy_target(
                            action_weights_np[i],
                            masks_np[i],
                            int(actions_np[i]),
                        ),
                        value=float(values_np[i]),
                        done=bool(dones_np[i]),
                    ))
                    episode_rewards[i] += rewards_np[i]
                    if successes_np[i]:
                        episode_successes[i] = True
                    if factor_hits_np[i]:
                        factor_hits += 1
                    if library_hits_np[i]:
                        library_hits += 1
                    if dones_np[i]:
                        if successes_np[i]:
                            successful_final_states.append(
                                jax.tree.map(lambda x: np.array(x[i]), next_states)
                            )
                        active[i] = False
                    if gumbel_stats_np is not None:
                        gumbel_policy_entropies.append(
                            float(gumbel_stats_np["policy_entropy"][i])
                        )
                        gumbel_considered_actions.append(
                            int(gumbel_stats_np["considered_actions"][i])
                        )
                        gumbel_selected_visits.append(
                            float(gumbel_stats_np["selected_action_visit_count"][i])
                        )
                        gumbel_max_root_q.append(
                            float(gumbel_stats_np["max_root_q"][i])
                        )
                        gumbel_min_root_q.append(
                            float(gumbel_stats_np["min_root_q"][i])
                        )

            states = next_states

        if self.factor_library is not None:
            for state_np in successful_final_states:
                nodes = self._state_to_fastpolys(state_np)
                self.factor_library.register_episode_nodes(
                    nodes, self.config.n_variables + 1
                )

        # Track success history for curriculum.
        for s in episode_successes:
            self.success_history.append(bool(s))

        success_rate = episode_successes.sum() / B
        avg_reward = episode_rewards.mean()

        rollout_info = {
            "episodes": B,
            "success_rate": float(success_rate),
            "avg_reward": float(avg_reward),
            "complexity": self.current_complexity,
            "factor_hits": factor_hits,
            "library_hits": library_hits,
            "library_size": len(self.factor_library) if self.factor_library else 0,
            "search_type": self.config.search,
        }
        if self.config.search == "gumbel":
            rollout_info.update({
                "gumbel_root_policy_entropy": float(np.mean(gumbel_policy_entropies)) if gumbel_policy_entropies else 0.0,
                "gumbel_considered_actions": float(np.mean(gumbel_considered_actions)) if gumbel_considered_actions else 0.0,
                "gumbel_selected_action_visit_count": float(np.mean(gumbel_selected_visits)) if gumbel_selected_visits else 0.0,
                "gumbel_max_root_q": float(np.mean(gumbel_max_root_q)) if gumbel_max_root_q else 0.0,
                "gumbel_min_root_q": float(np.mean(gumbel_min_root_q)) if gumbel_min_root_q else 0.0,
            })

        # Per-complexity breakdown.
        if self.fixed_complexities:
            for c in self.fixed_complexities:
                mask_c = complexity_labels_np == c
                n_c = mask_c.sum()
                if n_c > 0:
                    rollout_info[f"success_rate_C{c}"] = float(
                        episode_successes[mask_c].sum() / n_c
                    )
                    rollout_info[f"avg_reward_C{c}"] = float(
                        episode_rewards[mask_c].mean()
                    )
                    rollout_info[f"episodes_C{c}"] = int(n_c)
            rollout_info["complexity"] = str(self.fixed_complexities)

        return transitions, rollout_info

    # ------------------------------------------------------------------
    # GAE + PPO update
    # ------------------------------------------------------------------

    def compute_gae(self, transitions: List[Transition]):
        """Compute GAE advantages and returns."""
        n = len(transitions)
        advantages = np.zeros(n, dtype=np.float32)
        returns = np.zeros(n, dtype=np.float32)

        trajectories = [[] for _ in range(self.batch_size)]
        for idx, transition in enumerate(transitions):
            trajectories[transition.env_index].append(idx)

        for env_indices in trajectories:
            last_gae = 0.0
            for pos in reversed(range(len(env_indices))):
                idx = env_indices[pos]
                transition = transitions[idx]
                if transition.done:
                    next_value = 0.0
                    last_gae = 0.0
                elif pos + 1 < len(env_indices):
                    next_value = transitions[env_indices[pos + 1]].value
                else:
                    next_value = 0.0

                delta = (
                    transition.reward
                    + self.config.gamma * next_value
                    - transition.value
                )
                last_gae = (
                    delta
                    + self.config.gamma * self.config.gae_lambda * last_gae
                )
                advantages[idx] = last_gae
                returns[idx] = advantages[idx] + transition.value

        return advantages, returns

    def update(self, transitions: List[Transition],
               advantages: np.ndarray, returns: np.ndarray) -> dict:
        """Run the MCTS-distillation policy/value update.

        Args:
            transitions: Collected transitions.
            advantages: GAE advantages, shape (N,).
            returns: Target returns, shape (N,).

        Returns:
            Dict with mean policy loss, value loss, and entropy.
        """
        n = len(transitions)
        if n == 0:
            raise RuntimeError("PPO update received an empty rollout")

        self._assert_finite_train_state("before PPO update")

        if not np.isfinite(advantages).all() or not np.isfinite(returns).all():
            raise RuntimeError("Non-finite GAE targets detected before PPO update")

        ret_jax = jnp.array(returns)
        policy_targets_np = np.stack(
            [np.asarray(t.policy_target, dtype=np.float32) for t in transitions],
            axis=0,
        )
        policy_target_mass = policy_targets_np.sum(axis=1, keepdims=True)
        if (
            not np.isfinite(policy_targets_np).all()
            or not np.isfinite(policy_target_mass).all()
            or np.any(policy_target_mass <= 0.0)
        ):
            raise RuntimeError("Non-finite MCTS policy targets detected before PPO update")
        policy_targets_np = policy_targets_np / policy_target_mass
        policy_targets = jnp.array(policy_targets_np)

        # Pre-stack all observations into batched arrays.
        obs_keys = transitions[0].obs.keys()
        obs_stacked = {}
        for key in obs_keys:
            obs_stacked[key] = jnp.stack(
                [jnp.array(t.obs[key]) for t in transitions], axis=0
            )

        total_pg_loss = 0.0
        total_vf_loss = 0.0
        total_entropy = 0.0
        total_grad_norm = 0.0
        num_updates = 0
        skipped_updates = 0

        bs = self.config.batch_size

        for epoch in range(self.config.ppo_epochs):
            perm = np.random.permutation(n)

            for start in range(0, n, bs):
                end = min(start + bs, n)
                idx = perm[start:end]
                idx_jax = jnp.array(idx)

                batch_obs = jax.tree.map(lambda x: x[idx_jax], obs_stacked)
                batch_ret = ret_jax[idx_jax]
                batch_policy_targets = policy_targets[idx_jax]

                self.train_state, loss_info = self._ppo_step(
                    self.train_state,
                    batch_obs,
                    batch_ret,
                    batch_policy_targets,
                )

                pg_loss = float(loss_info['pg_loss'])
                vf_loss = float(loss_info['vf_loss'])
                entropy = float(loss_info['entropy'])
                grad_norm = float(loss_info['grad_norm'])
                applied = bool(loss_info['applied'])

                if applied:
                    total_pg_loss += pg_loss
                    total_vf_loss += vf_loss
                    total_entropy += entropy
                skipped_updates += int(not applied)
                total_grad_norm += grad_norm
                num_updates += 1

        self._assert_finite_train_state("after PPO update")

        return {
            "pg_loss": total_pg_loss / max(num_updates, 1),
            "vf_loss": total_vf_loss / max(num_updates, 1),
            "entropy": total_entropy / max(num_updates, 1),
            "grad_norm": total_grad_norm / max(num_updates, 1),
            "skipped_updates": skipped_updates,
        }

    @functools.partial(jax.jit, static_argnums=(0,))
    def _ppo_step(self, state, obs, returns, policy_targets):
        """Single JIT-compiled MCTS-distillation gradient step."""

        def loss_fn(params):
            # vmap network over batch.
            logits, values = jax.vmap(
                lambda o: self.network.apply(params, o)
            )(obs)

            log_probs = jax.nn.log_softmax(logits)

            # Entropy.
            probs = jax.nn.softmax(logits)
            entropy = -(probs * log_probs).sum(axis=-1).mean()

            # Distill the improved MCTS policy instead of applying PPO off-policy.
            pg_loss = -jnp.sum(
                jax.lax.stop_gradient(policy_targets) * log_probs,
                axis=-1,
            ).mean()

            # Value loss.
            vf_loss = jnp.mean((values - returns) ** 2)

            total = (pg_loss
                     + self.config.vf_coef * vf_loss
                     - self.config.ent_coef * entropy)

            return total, {'pg_loss': pg_loss, 'vf_loss': vf_loss, 'entropy': entropy}

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (_, loss_info), grads = grad_fn(state.params)
        grad_norm = optax.global_norm(grads)
        applied = jnp.all(jnp.array([
            jnp.isfinite(loss_info['pg_loss']),
            jnp.isfinite(loss_info['vf_loss']),
            jnp.isfinite(loss_info['entropy']),
            jnp.isfinite(grad_norm),
        ]))
        state = jax.lax.cond(
            applied,
            lambda s: s.apply_gradients(grads=grads),
            lambda s: s,
            state,
        )
        loss_info = {
            **loss_info,
            'grad_norm': grad_norm,
            'applied': applied,
        }
        return state, loss_info

    # ------------------------------------------------------------------
    # Curriculum
    # ------------------------------------------------------------------

    def _maybe_advance_curriculum(self):
        if not self.config.curriculum_enabled:
            return
        window = 50
        if len(self.success_history) < window:
            return
        recent = self.success_history[-window:]
        rate = sum(recent) / len(recent)
        if (rate >= self.config.advance_threshold
                and self.current_complexity < self.config.max_complexity):
            self.current_complexity += 1
            self.success_history.clear()
            print(f"[Curriculum] Advanced to complexity {self.current_complexity}")
        elif (rate <= self.config.backoff_threshold
              and self.current_complexity > self.config.starting_complexity):
            self.current_complexity -= 1
            self.success_history.clear()
            print(f"[Curriculum] Backed off to complexity {self.current_complexity}")

    # ------------------------------------------------------------------
    # Checkpoint save / load
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str) -> None:
        """Save model params, optimizer state, and config to a pickle file.

        Args:
            path: File path (e.g. 'results/.../checkpoint_00050.pkl').
        """
        self._assert_finite_train_state(f"before saving checkpoint to {path}")
        state_dict = {
            "params": self.train_state.params,
            "opt_state": self.train_state.opt_state,
            "step": int(self.train_state.step),
            "config": self.config,
            "fixed_complexities": self.fixed_complexities,
            "current_complexity": self.current_complexity,
            "algorithm": "ppo-mcts-jax",
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(state_dict, f)

    def _merge_checkpoint_params(self, loaded_params):
        """Warm-start current params from a checkpoint with possible shape growth.

        When the architecture expands (for example, a larger action head after
        increasing ``max_complexity``), copy the overlapping slice from the
        checkpoint into the freshly initialized parameter tensor and leave new
        rows/columns at their random initialization.
        """

        adapted_paths: List[str] = []

        def _merge(current, loaded, path: Tuple[str, ...]):
            if isinstance(current, Mapping):
                merged = {}
                loaded_mapping = loaded if isinstance(loaded, Mapping) else {}
                for key, current_value in current.items():
                    child_path = path + (str(key),)
                    if key in loaded_mapping:
                        merged[key] = _merge(current_value, loaded_mapping[key], child_path)
                    else:
                        adapted_paths.append(".".join(child_path))
                        merged[key] = current_value
                extra_keys = set(loaded_mapping.keys()) - set(current.keys())
                for key in extra_keys:
                    adapted_paths.append(".".join(path + (str(key),)))
                return merged

            current_arr = jnp.asarray(current)
            loaded_arr = jnp.asarray(loaded, dtype=current_arr.dtype)

            if current_arr.shape == loaded_arr.shape:
                return loaded_arr

            adapted_paths.append(".".join(path))
            if current_arr.ndim != loaded_arr.ndim:
                return current_arr

            overlap = tuple(
                slice(0, min(curr_dim, load_dim))
                for curr_dim, load_dim in zip(current_arr.shape, loaded_arr.shape)
            )
            return current_arr.at[overlap].set(loaded_arr[overlap])

        current_params = self.train_state.params
        current_tree = (
            unfreeze(current_params)
            if isinstance(current_params, FrozenDict)
            else current_params
        )
        loaded_tree = (
            unfreeze(loaded_params)
            if isinstance(loaded_params, FrozenDict)
            else loaded_params
        )
        merged_tree = _merge(current_tree, loaded_tree, ())
        if isinstance(current_params, FrozenDict):
            return freeze(merged_tree), adapted_paths
        return merged_tree, adapted_paths

    def load_checkpoint(self, path: str) -> None:
        """Load model params and optimizer state from a pickle checkpoint.

        Args:
            path: Path to a checkpoint saved by save_checkpoint().
        """
        with open(path, "rb") as f:
            state_dict = pickle.load(f)
        merged_params, adapted_paths = self._merge_checkpoint_params(
            state_dict["params"]
        )
        if adapted_paths:
            preview = ", ".join(adapted_paths[:4])
            if len(adapted_paths) > 4:
                preview += ", ..."
            print(
                "Warm-starting checkpoint into the current architecture; "
                "resetting optimizer state for adapted params: "
                f"{preview}"
            )
            self.train_state = train_state.TrainState.create(
                apply_fn=self.network.apply,
                params=merged_params,
                tx=self.tx,
            ).replace(
                step=state_dict["step"],
            )
        else:
            self.train_state = self.train_state.replace(
                params=state_dict["params"],
                opt_state=state_dict["opt_state"],
                step=state_dict["step"],
            )
        self.current_complexity = state_dict.get(
            "current_complexity", self.current_complexity
        )
        self._assert_finite_train_state(f"after loading checkpoint from {path}")

    # ------------------------------------------------------------------
    # Main train loop
    # ------------------------------------------------------------------

    def train(self, num_iterations: int,
              results_dir: str = "results") -> dict:
        """Run the full MCTS-guided JAX training loop.

        Each iteration:
          1. Collect batch_size episodes using batched MCTS.
          2. Compute GAE.
          3. Policy/value update.
          4. Adjust curriculum.
          5. Save checkpoint every 50 iterations.

        Args:
            num_iterations: Number of collect + update cycles.
            results_dir: Directory for saving checkpoints.

        Returns:
            History dict with metric lists.
        """
        checkpoint_interval = 50
        history = {
            "pg_loss": [], "vf_loss": [], "entropy": [],
            "success_rate": [], "avg_reward": [], "complexity": [],
            "grad_norm": [], "skipped_updates": [],
        }
        if self.config.search == "gumbel":
            history.update({
                "gumbel_root_policy_entropy": [],
                "gumbel_considered_actions": [],
                "gumbel_selected_action_visit_count": [],
                "gumbel_max_root_q": [],
                "gumbel_min_root_q": [],
            })
        # Per-complexity history when using fixed_complexities.
        if self.fixed_complexities:
            for c in self.fixed_complexities:
                history[f"success_rate_C{c}"] = []
                history[f"avg_reward_C{c}"] = []

        print("Iteration 1: JIT-compiling MCTS + env (this may take a few minutes)...",
              flush=True)

        pbar = tqdm(range(1, num_iterations + 1), desc="Training", unit="iter")
        for iteration in pbar:
            iter_start = time.time()

            transitions, rollout_info = self.collect_rollouts()

            if iteration == 1:
                    print(f"  Rollout done ({time.time() - iter_start:.1f}s). "
                        "Running policy/value update...", flush=True)

            advantages, returns = self.compute_gae(transitions)
            loss_info = self.update(transitions, advantages, returns)
            self._maybe_advance_curriculum()

            iter_time = time.time() - iter_start
            if iteration == 1:
                print(f"  Iteration 1 complete ({iter_time:.1f}s). "
                      "Subsequent iterations should be much faster.", flush=True)

            # Update tqdm postfix with key metrics.
            pbar.set_postfix({
                "success": f"{rollout_info['success_rate']:.1%}",
                "reward": f"{rollout_info['avg_reward']:.2f}",
                "entropy": f"{loss_info['entropy']:.2f}",
                "skip": int(loss_info['skipped_updates']),
                "s/iter": f"{iter_time:.1f}",
            })

            history["pg_loss"].append(loss_info["pg_loss"])
            history["vf_loss"].append(loss_info["vf_loss"])
            history["entropy"].append(loss_info["entropy"])
            history["success_rate"].append(rollout_info["success_rate"])
            history["avg_reward"].append(rollout_info["avg_reward"])
            history["complexity"].append(rollout_info["complexity"])
            history["grad_norm"].append(loss_info["grad_norm"])
            history["skipped_updates"].append(loss_info["skipped_updates"])
            if self.config.search == "gumbel":
                history["gumbel_root_policy_entropy"].append(
                    rollout_info["gumbel_root_policy_entropy"]
                )
                history["gumbel_considered_actions"].append(
                    rollout_info["gumbel_considered_actions"]
                )
                history["gumbel_selected_action_visit_count"].append(
                    rollout_info["gumbel_selected_action_visit_count"]
                )
                history["gumbel_max_root_q"].append(
                    rollout_info["gumbel_max_root_q"]
                )
                history["gumbel_min_root_q"].append(
                    rollout_info["gumbel_min_root_q"]
                )

            if self.fixed_complexities:
                for c in self.fixed_complexities:
                    history[f"success_rate_C{c}"].append(
                        rollout_info.get(f"success_rate_C{c}", 0.0)
                    )
                    history[f"avg_reward_C{c}"].append(
                        rollout_info.get(f"avg_reward_C{c}", 0.0)
                    )

            if self.config.wandb_enabled:
                import wandb
                log_dict = {
                    "iteration": iteration,
                    "pg_loss": loss_info["pg_loss"],
                    "vf_loss": loss_info["vf_loss"],
                    "entropy": loss_info["entropy"],
                    "success_rate": rollout_info["success_rate"],
                    "avg_reward": rollout_info["avg_reward"],
                    "episodes": rollout_info["episodes"],
                    "factor_hits": rollout_info["factor_hits"],
                    "library_hits": rollout_info["library_hits"],
                    "library_size": rollout_info["library_size"],
                    "search/is_gumbel": float(self.config.search == "gumbel"),
                }
                if self.config.search == "gumbel":
                    log_dict.update({
                        "gumbel/root_policy_entropy": rollout_info["gumbel_root_policy_entropy"],
                        "gumbel/considered_actions": rollout_info["gumbel_considered_actions"],
                        "gumbel/selected_action_visit_count": rollout_info["gumbel_selected_action_visit_count"],
                        "gumbel/max_root_q": rollout_info["gumbel_max_root_q"],
                        "gumbel/min_root_q": rollout_info["gumbel_min_root_q"],
                    })
                if self.fixed_complexities:
                    for c in self.fixed_complexities:
                        log_dict[f"success_rate_C{c}"] = rollout_info.get(
                            f"success_rate_C{c}", 0.0
                        )
                        log_dict[f"avg_reward_C{c}"] = rollout_info.get(
                            f"avg_reward_C{c}", 0.0
                        )
                else:
                    log_dict["complexity"] = rollout_info["complexity"]
                log_dict["grad_norm"] = loss_info["grad_norm"]
                log_dict["skipped_updates"] = loss_info["skipped_updates"]
                wandb.log(log_dict, step=iteration)

            if loss_info["skipped_updates"]:
                tqdm.write(
                    f"[PPO+MCTS-JAX iter {iteration}] skipped "
                    f"{loss_info['skipped_updates']} non-finite minibatch updates"
                )

            if iteration % self.config.log_interval == 0:
                line = (
                    f"[PPO+MCTS-JAX iter {iteration}] "
                    f"episodes={rollout_info['episodes']} "
                    f"lib={rollout_info['library_size']} "
                    f"fhits={rollout_info['factor_hits']} "
                    f"lhits={rollout_info['library_hits']} "
                    f"success={rollout_info['success_rate']:.2%} "
                    f"reward={rollout_info['avg_reward']:.3f} "
                    f"pg_loss={loss_info['pg_loss']:.4f} "
                    f"vf_loss={loss_info['vf_loss']:.4f} "
                    f"entropy={loss_info['entropy']:.4f} "
                    f"grad_norm={loss_info['grad_norm']:.4f} "
                    f"skipped={loss_info['skipped_updates']} "
                    f"({iter_time:.1f}s/iter)"
                )
                if self.fixed_complexities:
                    parts = []
                    for c in self.fixed_complexities:
                        sr = rollout_info.get(f"success_rate_C{c}", 0.0)
                        parts.append(f"C{c}={sr:.1%}")
                    line += " | " + " ".join(parts)
                else:
                    line = line.replace(
                        f"episodes=",
                        f"complexity={rollout_info['complexity']} episodes=",
                    )
                if self.config.search == "gumbel":
                    line += (
                        f" | g_entropy={rollout_info['gumbel_root_policy_entropy']:.4f}"
                        f" g_considered={rollout_info['gumbel_considered_actions']:.2f}"
                        f" g_selected_visits={rollout_info['gumbel_selected_action_visit_count']:.2f}"
                        f" g_q=({rollout_info['gumbel_min_root_q']:.4f},"
                        f" {rollout_info['gumbel_max_root_q']:.4f})"
                    )
                tqdm.write(line)

            # Save periodic checkpoints.
            if iteration % checkpoint_interval == 0 or iteration == num_iterations:
                ckpt_path = os.path.join(
                    results_dir, f"checkpoint_{iteration:05d}.pkl"
                )
                self.save_checkpoint(ckpt_path)
                tqdm.write(f"  Saved checkpoint: {ckpt_path}")

        return history
