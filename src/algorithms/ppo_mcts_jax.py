"""PPO + parallel MCTS trainer using JAX and mctx.

This implements the same Expert Iteration loop as ppo_mcts.py but with
batched MCTS via Google DeepMind's mctx library. All B environments are
searched in parallel on GPU using mctx.muzero_policy, giving a massive
speedup over the sequential PyTorch implementation.

Architecture:
  - Environment states are pure JAX arrays (jax_env.EnvState).
  - Policy-value network is Flax (jax_net.PolicyValueNet).
  - MCTS uses mctx.muzero_policy with a learned dynamics model replaced by
    the true environment dynamics (perfect simulator).
  - PPO update uses optax for gradient clipping and Adam.

Flow per iteration:
  1. Reset B parallel environments with sampled targets.
  2. For each step (up to max_steps):
     a. Batch-evaluate policy+value for all B states.
     b. Run mctx.muzero_policy for all B states in parallel.
     c. Execute chosen actions in all B environments.
     d. Store transitions.
  3. Compute GAE advantages.
  4. Run PPO mini-batch updates.
  5. Adjust curriculum.
"""

import functools
import math
import os
import pickle
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import jax
import jax.numpy as jnp
import mctx
import numpy as np
import optax
import flax.linen as nn
from flax.training import train_state
from tqdm import tqdm

from ..config import Config
from ..environment.fast_polynomial import FastPoly
from ..environment.factor_library import FactorLibrary
from ..game_board.generator import generate_random_circuit
from ..game_board.on_path import OnPathCache
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
    action: int
    reward: float
    network_log_prob: float
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
        self.fixed_complexities = fixed_complexities
        if fixed_complexities:
            config.curriculum_enabled = False
        self.env_config = make_env_config(config)
        self.batch_size = batch_size
        self.library_cache_size = 512
        if config.reward_mode not in ("legacy", "clean_sparse", "clean_onpath"):
            raise ValueError(f"Unknown reward_mode: {config.reward_mode}")
        if config.on_path_phi_mode not in ("count", "max_step", "depth_weighted"):
            raise ValueError(f"Unknown on_path_phi_mode: {config.on_path_phi_mode}")
        if config.on_path_depth_weight_power < 0:
            raise ValueError("on_path_depth_weight_power must be non-negative")
        if config.on_path_route_consistency_mode not in (
            "best_route_phi",
            "lock_on_first_hit",
            "off",
        ):
            raise ValueError(
                "on_path_route_consistency_mode must be one of "
                "'best_route_phi', 'lock_on_first_hit', or 'off'"
            )
        if config.on_path_num_routes < 1 or config.on_path_num_routes > 32:
            raise ValueError("on_path_num_routes must be between 1 and 32")

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
        self.dwell_iterations_at_level = 0
        self.window_success_rate = 0.0
        self.backoff_patience_counter = 0
        self.consecutive_nonfinite_minibatches = 0
        self.skipped_minibatch_updates_total = 0
        self.skipped_outer_iterations = 0

        factor_library: Optional[FactorLibrary] = None
        if config.reward_mode == "legacy" and config.factor_library_enabled:
            factor_library = FactorLibrary(
                mod=config.mod,
                n_vars=config.n_variables,
                max_degree=config.effective_max_degree,
            )
        self.factor_library = factor_library

        self._rng = np.random.default_rng(config.seed)
        self.on_path_cache: Optional[OnPathCache] = None
        if config.reward_mode == "clean_onpath":
            if not config.graph_onpath_cache_dir:
                raise ValueError(
                    "reward_mode='clean_onpath' requires graph_onpath_cache_dir"
                )
            self.on_path_cache = OnPathCache.load(
                config.graph_onpath_cache_dir,
                config,
                self._required_on_path_complexities(),
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
            _factor_hits, _library_hits, _additive_complete, _mult_complete,
            _on_path_hit, _on_path_phi,
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
        """Run batched MCTS using mctx.muzero_policy.

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

        policy_output = mctx.muzero_policy(
            params=params,
            rng_key=rng_key,
            root=root,
            recurrent_fn=recurrent_fn,
            num_simulations=self.config.mcts_simulations,
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

    def _required_on_path_complexities(self) -> List[int]:
        if self.fixed_complexities:
            return list(self.fixed_complexities)
        if self.config.curriculum_enabled:
            return list(range(self.config.starting_complexity, self.config.max_complexity + 1))
        return [self.config.max_complexity]

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

    def _sample_targets_multi(self, n: int) -> Tuple[List[FastPoly], List[int], list]:
        """Sample n target polynomials, distributing across complexities.

        When ``fixed_complexities`` is set, the batch is split evenly across
        all specified complexity levels. Otherwise falls back to the single
        current_complexity used by curriculum.

        Args:
            n: Total number of targets to sample.

        Returns:
            (targets, complexity_labels, on_path_contexts) — lists of length n.
        """
        if self.fixed_complexities:
            targets = []
            labels = []
            contexts = []
            per_c = n // len(self.fixed_complexities)
            remainder = n % len(self.fixed_complexities)
            for idx, c in enumerate(self.fixed_complexities):
                count = per_c + (1 if idx < remainder else 0)
                for _ in range(count):
                    if self.config.reward_mode == "clean_onpath":
                        assert self.on_path_cache is not None
                        ctx = self.on_path_cache.sample_train_context(c, self._rng)
                        targets.append(ctx.target_poly)
                        contexts.append(ctx)
                    else:
                        targets.append(self._sample_one_target(c))
                        contexts.append(None)
                    labels.append(c)
            return targets, labels, contexts
        else:
            targets = []
            contexts = []
            for _ in range(n):
                if self.config.reward_mode == "clean_onpath":
                    assert self.on_path_cache is not None
                    ctx = self.on_path_cache.sample_train_context(
                        self.current_complexity, self._rng
                    )
                    targets.append(ctx.target_poly)
                    contexts.append(ctx)
                else:
                    targets.append(self._sample_one_target(self.current_complexity))
                    contexts.append(None)
            return targets, [self.current_complexity] * n, contexts

    def _fastpoly_to_jax(self, poly: FastPoly) -> jnp.ndarray:
        """Convert FastPoly to flat JAX int32 coefficient array."""
        return jnp.array(poly.coeffs.flatten(), dtype=jnp.int32)

    def _prepare_on_path_contexts(self, contexts: list):
        max_size = self.env_config.on_path_max_size
        target_size = self.env_config.target_size
        if self.config.reward_mode != "clean_onpath":
            batch = len(contexts)
            return (
                jnp.zeros((batch, max_size, target_size), dtype=jnp.int32),
                jnp.zeros((batch, max_size), dtype=jnp.uint32),
                jnp.zeros((batch, max_size), dtype=jnp.int32),
                jnp.zeros((batch, max_size), dtype=jnp.uint32),
                jnp.zeros((batch, max_size), dtype=jnp.bool_),
                jnp.zeros((batch,), dtype=jnp.int32),
            )
        assert self.on_path_cache is not None
        (
            coeffs,
            hashes,
            steps,
            route_masks,
            active,
            target_steps,
        ) = self.on_path_cache.pack_jax_contexts(
            contexts,
            max_size=max_size,
            target_size=target_size,
        )
        return (
            jnp.array(coeffs, dtype=jnp.int32),
            jnp.array(hashes, dtype=jnp.uint32),
            jnp.array(steps, dtype=jnp.int32),
            jnp.array(route_masks, dtype=jnp.uint32),
            jnp.array(active),
            jnp.array(target_steps, dtype=jnp.int32),
        )

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
        targets, complexity_labels, on_path_contexts = self._sample_targets_multi(B)
        complexity_labels_np = np.array(complexity_labels)
        target_arrays = jnp.stack(
            [self._fastpoly_to_jax(t) for t in targets], axis=0
        )  # (B, target_size)
        subgoal_coeffs, subgoal_active, subgoal_library_known = (
            self._prepare_initial_subgoals(targets)
        )
        (
            on_path_coeffs,
            on_path_hashes,
            on_path_steps,
            on_path_route_masks,
            on_path_active,
            target_board_steps,
        ) = self._prepare_on_path_contexts(on_path_contexts)
        library_coeffs, library_mask = self._export_library_cache()

        # Batch reset.
        states = jax.vmap(
            lambda tc, sgc, sga, sgl, opc, oph, ops, opr, opa, tbs: env_reset(
                self.env_config, tc, sgc, sga, sgl, opc, oph, ops, opr, opa, tbs
            )
        )(
            target_arrays,
            subgoal_coeffs,
            subgoal_active,
            subgoal_library_known,
            on_path_coeffs,
            on_path_hashes,
            on_path_steps,
            on_path_route_masks,
            on_path_active,
            target_board_steps,
        )

        transitions = []
        episode_rewards = np.zeros(B)
        episode_successes = np.zeros(B, dtype=bool)
        episode_on_path_hit = np.zeros(B, dtype=bool)
        active = np.ones(B, dtype=bool)  # Track which envs are still running.
        factor_hits = 0
        library_hits = 0
        on_path_hits = 0
        episode_on_path_phi = np.zeros(B, dtype=np.float32)
        successful_final_states = []

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

            # Sample actions from MCTS visit counts (with temperature).
            action_weights = policy_output.action_weights  # (B, max_actions)
            rng, sample_rng = jax.random.split(rng)
            actions = jax.random.categorical(
                sample_rng, jnp.log(action_weights + 1e-8)
            )  # (B,)

            # Get network log probs for PPO ratio.
            logits, values = jax.vmap(
                lambda obs: self.network.apply(params, obs)
            )(obs_batch)
            log_probs = jax.nn.log_softmax(logits)
            network_log_probs = log_probs[
                jnp.arange(B), actions
            ]  # (B,)

            # Step all environments.
            (
                next_states, rewards, dones, successes,
                factor_hits_step, library_hits_step,
                _additive_complete, _mult_complete,
                on_path_hits_step, on_path_phi_step,
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
            on_path_hits_np = np.array(on_path_hits_step)
            on_path_phi_np = np.array(on_path_phi_step)
            values_np = np.array(values)
            network_lp_np = np.array(network_log_probs)

            # Store transitions for active environments.
            for i in range(B):
                if active[i]:
                    # Extract single-env observation (numpy-ify for storage).
                    obs_i = jax.tree.map(lambda x: np.array(x[i]), obs_batch)
                    transitions.append(Transition(
                        obs=obs_i,
                        action=int(actions_np[i]),
                        reward=float(rewards_np[i]),
                        network_log_prob=float(network_lp_np[i]),
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
                    if on_path_hits_np[i]:
                        on_path_hits += 1
                        episode_on_path_hit[i] = True
                    episode_on_path_phi[i] = on_path_phi_np[i]
                    if dones_np[i]:
                        if successes_np[i]:
                            successful_final_states.append(
                                jax.tree.map(lambda x: np.array(x[i]), next_states)
                            )
                        active[i] = False

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
        episode_lengths = np.array(next_states.steps_taken)
        outcome_info = self._outcome_bucket_metrics(
            episode_rewards=episode_rewards,
            episode_successes=episode_successes,
            episode_on_path_hit=episode_on_path_hit,
            episode_phi=episode_on_path_phi,
            episode_lengths=episode_lengths,
        )

        rollout_info = {
            "episodes": B,
            "success_rate": float(success_rate),
            "avg_reward": float(avg_reward),
            "complexity": self.current_complexity,
            "factor_hits": factor_hits,
            "library_hits": library_hits,
            "library_size": len(self.factor_library) if self.factor_library else 0,
            "on_path_hits": on_path_hits,
            "on_path_phi": float(episode_on_path_phi.mean()),
            "target_board_step": float(np.array(target_board_steps).mean()),
            "episode_length": float(episode_lengths.mean()),
            **outcome_info,
        }

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

        last_gae = 0.0
        for t in reversed(range(n)):
            if transitions[t].done:
                next_value = 0.0
                last_gae = 0.0
            elif t + 1 < n:
                next_value = transitions[t + 1].value
            else:
                next_value = 0.0

            delta = (transitions[t].reward
                     + self.config.gamma * next_value
                     - transitions[t].value)
            last_gae = delta + self.config.gamma * self.config.gae_lambda * last_gae
            advantages[t] = last_gae
            returns[t] = advantages[t] + transitions[t].value

        return advantages, returns

    def _rollout_data_is_finite(
        self,
        transitions: List[Transition],
        advantages: np.ndarray,
        returns: np.ndarray,
    ) -> bool:
        """Validate rollout scalars before applying any PPO update."""
        if not transitions:
            return False

        rewards = np.array([t.reward for t in transitions], dtype=np.float32)
        values = np.array([t.value for t in transitions], dtype=np.float32)
        log_probs = np.array(
            [t.network_log_prob for t in transitions],
            dtype=np.float32,
        )
        arrays = (rewards, values, log_probs, advantages, returns)
        return all(np.all(np.isfinite(arr)) for arr in arrays)

    @staticmethod
    def _masked_mean(values: np.ndarray, mask: np.ndarray) -> float:
        if not np.any(mask):
            return 0.0
        return float(np.asarray(values)[mask].mean())

    @staticmethod
    def _trailing_mean(values: List[float], window: int) -> Optional[float]:
        if len(values) < window:
            return None
        return float(np.mean(values[-window:]))

    @classmethod
    def _outcome_bucket_metrics(
        cls,
        episode_rewards: np.ndarray,
        episode_successes: np.ndarray,
        episode_on_path_hit: np.ndarray,
        episode_phi: np.ndarray,
        episode_lengths: np.ndarray,
    ) -> dict:
        success_mask = np.asarray(episode_successes, dtype=bool)
        hit_mask = np.asarray(episode_on_path_hit, dtype=bool)
        failure_with_hit_mask = hit_mask & ~success_mask
        failure_no_hit_mask = ~hit_mask & ~success_mask

        success_count = int(success_mask.sum())
        failure_with_hit_count = int(failure_with_hit_mask.sum())
        failure_no_hit_count = int(failure_no_hit_mask.sum())
        hit_denominator = success_count + failure_with_hit_count
        episode_total = int(success_mask.size)

        return {
            "success_count": success_count,
            "failure_with_hit_count": failure_with_hit_count,
            "failure_no_hit_count": failure_no_hit_count,
            "p_hit": (
                float(hit_denominator / episode_total)
                if episode_total > 0 else 0.0
            ),
            "p_solve_given_hit": (
                float(success_count / hit_denominator)
                if hit_denominator > 0 else 0.0
            ),
            "avg_return_success": cls._masked_mean(
                episode_rewards, success_mask
            ),
            "avg_return_failure_with_hit": cls._masked_mean(
                episode_rewards, failure_with_hit_mask
            ),
            "avg_return_failure_no_hit": cls._masked_mean(
                episode_rewards, failure_no_hit_mask
            ),
            "avg_phi_success": cls._masked_mean(
                episode_phi, success_mask
            ),
            "avg_phi_failure_with_hit": cls._masked_mean(
                episode_phi, failure_with_hit_mask
            ),
            "avg_episode_length_success": cls._masked_mean(
                episode_lengths, success_mask
            ),
            "avg_episode_length_failure_with_hit": cls._masked_mean(
                episode_lengths, failure_with_hit_mask
            ),
            "avg_episode_length_failure_no_hit": cls._masked_mean(
                episode_lengths, failure_no_hit_mask
            ),
        }

    @staticmethod
    def _empty_loss_info(skipped_outer: bool = False) -> dict:
        return {
            "pg_loss": 0.0,
            "vf_loss": 0.0,
            "entropy": 0.0,
            "weighted_vf_loss": 0.0,
            "pg_to_weighted_vf_ratio": 0.0,
            "max_abs_logit": 0.0,
            "max_log_ratio": 0.0,
            "max_return": 0.0,
            "max_abs_advantage": 0.0,
            "grad_global_norm": 0.0,
            "approx_kl": 0.0,
            "kl_early_stop_count": 0,
            "kl_rejected_updates": 0,
            "kl_rejection_rate": 0.0,
            "skipped_minibatch_updates": 0,
            "applied_minibatch_updates": 0,
            "skipped_outer_iteration": int(skipped_outer),
        }

    def update(self, transitions: List[Transition],
               advantages: np.ndarray, returns: np.ndarray) -> dict:
        """Run PPO clipped surrogate update.

        Args:
            transitions: Collected transitions.
            advantages: GAE advantages, shape (N,).
            returns: Target returns, shape (N,).

        Returns:
            Dict with mean pg_loss, vf_loss, entropy.
        """
        n = len(transitions)
        if n == 0:
            return self._empty_loss_info(skipped_outer=True)

        if not self._rollout_data_is_finite(transitions, advantages, returns):
            return self._empty_loss_info(skipped_outer=True)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        if not np.all(np.isfinite(advantages)):
            return self._empty_loss_info(skipped_outer=True)

        adv_jax = jnp.array(advantages)
        ret_jax = jnp.array(returns)
        old_log_probs = jnp.array([t.network_log_prob for t in transitions])
        actions = jnp.array([t.action for t in transitions], dtype=jnp.int32)

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
        max_abs_logit = 0.0
        max_log_ratio = 0.0
        max_return = 0.0
        max_abs_advantage = 0.0
        max_grad_global_norm = 0.0
        max_approx_kl = 0.0
        kl_early_stop_count = 0
        kl_rejected_updates = 0
        num_updates = 0
        skipped_updates = 0

        bs = self.config.batch_size
        target_kl = float(self.config.target_kl)

        for epoch in range(self.config.ppo_epochs):
            perm = np.random.permutation(n)
            stop_for_kl = False

            for start in range(0, n, bs):
                end = min(start + bs, n)
                idx = perm[start:end]
                idx_jax = jnp.array(idx)

                batch_obs = jax.tree.map(lambda x: x[idx_jax], obs_stacked)
                batch_actions = actions[idx_jax]
                batch_adv = adv_jax[idx_jax]
                batch_ret = ret_jax[idx_jax]
                batch_old_lp = old_log_probs[idx_jax]

                self.train_state, loss_info = self._ppo_step(
                    self.train_state, batch_obs, batch_actions,
                    batch_adv, batch_ret, batch_old_lp,
                )
                applied = bool(np.array(loss_info["applied"]))
                kl_rejected = bool(np.array(loss_info["kl_rejected"]))

                max_abs_logit = max(max_abs_logit, float(loss_info["max_abs_logit"]))
                max_log_ratio = max(max_log_ratio, float(loss_info["max_log_ratio"]))
                max_return = max(max_return, float(loss_info["max_return"]))
                max_abs_advantage = max(
                    max_abs_advantage,
                    float(loss_info["max_abs_advantage"]),
                )
                max_grad_global_norm = max(
                    max_grad_global_norm,
                    float(loss_info["grad_global_norm"]),
                )
                max_approx_kl = max(
                    max_approx_kl,
                    float(loss_info["approx_kl"]),
                )

                if applied:
                    self.consecutive_nonfinite_minibatches = 0
                    total_pg_loss += float(loss_info["pg_loss"])
                    total_vf_loss += float(loss_info["vf_loss"])
                    total_entropy += float(loss_info["entropy"])
                    num_updates += 1
                elif kl_rejected:
                    self.consecutive_nonfinite_minibatches = 0
                    kl_rejected_updates += 1
                    attempted_kl_updates = num_updates + kl_rejected_updates
                    if (
                        target_kl > 0.0
                        and kl_rejected_updates / max(attempted_kl_updates, 1) > 0.5
                    ):
                        kl_early_stop_count += 1
                        stop_for_kl = True
                        break
                else:
                    skipped_updates += 1
                    self.skipped_minibatch_updates_total += 1
                    self.consecutive_nonfinite_minibatches += 1
                    if (
                        self.consecutive_nonfinite_minibatches
                        >= self.config.nonfinite_update_limit
                    ):
                        raise RuntimeError(
                            "Aborting after "
                            f"{self.consecutive_nonfinite_minibatches} consecutive "
                            "non-finite PPO minibatch updates"
                        )
            if stop_for_kl:
                break

        mean_pg_loss = total_pg_loss / max(num_updates, 1)
        mean_vf_loss = total_vf_loss / max(num_updates, 1)
        mean_entropy = total_entropy / max(num_updates, 1)
        weighted_vf_loss = float(self.config.vf_coef) * mean_vf_loss
        pg_to_weighted_vf_ratio = (
            abs(mean_pg_loss) / max(abs(weighted_vf_loss), 1e-8)
        )
        attempted_kl_updates = num_updates + kl_rejected_updates
        kl_rejection_rate = (
            kl_rejected_updates / max(attempted_kl_updates, 1)
        )

        return {
            "pg_loss": mean_pg_loss,
            "vf_loss": mean_vf_loss,
            "entropy": mean_entropy,
            "weighted_vf_loss": weighted_vf_loss,
            "pg_to_weighted_vf_ratio": pg_to_weighted_vf_ratio,
            "max_abs_logit": max_abs_logit,
            "max_log_ratio": max_log_ratio,
            "max_return": max_return,
            "max_abs_advantage": max_abs_advantage,
            "grad_global_norm": max_grad_global_norm,
            "approx_kl": max_approx_kl,
            "kl_early_stop_count": kl_early_stop_count,
            "kl_rejected_updates": kl_rejected_updates,
            "kl_rejection_rate": kl_rejection_rate,
            "skipped_minibatch_updates": skipped_updates,
            "applied_minibatch_updates": num_updates,
            "skipped_outer_iteration": 0,
        }

    @functools.partial(jax.jit, static_argnums=(0,))
    def _ppo_step(self, state, obs, actions, advantages, returns, old_log_probs):
        """Single JIT-compiled PPO gradient step."""

        def loss_fn(params):
            # vmap network over batch.
            logits, values = jax.vmap(
                lambda o: self.network.apply(params, o)
            )(obs)

            # New log probs.
            log_probs = jax.nn.log_softmax(logits)
            new_lp = log_probs[jnp.arange(actions.shape[0]), actions]

            # Entropy.
            probs = jax.nn.softmax(logits)
            entropy = -(probs * log_probs).sum(axis=-1).mean()

            # PPO clipped ratio.
            raw_log_ratio = new_lp - old_log_probs
            log_ratio = jnp.clip(
                raw_log_ratio,
                -self.config.ppo_log_ratio_clip,
                self.config.ppo_log_ratio_clip,
            )
            ratio = jnp.exp(log_ratio)
            approx_kl = jnp.mean((ratio - 1.0) - log_ratio)
            surr1 = ratio * advantages
            surr2 = jnp.clip(
                ratio, 1 - self.config.ppo_clip, 1 + self.config.ppo_clip
            ) * advantages
            pg_loss = -jnp.minimum(surr1, surr2).mean()

            # Value loss.
            vf_loss = jnp.mean((values - returns) ** 2)

            total = (pg_loss
                     + self.config.vf_coef * vf_loss
                     - self.config.ent_coef * entropy)

            aux = {
                "pg_loss": pg_loss,
                "vf_loss": vf_loss,
                "entropy": entropy,
                "max_abs_logit": jnp.max(jnp.abs(logits)),
                "max_log_ratio": jnp.max(jnp.abs(log_ratio)),
                "max_return": jnp.max(jnp.abs(returns)),
                "max_abs_advantage": jnp.max(jnp.abs(advantages)),
                "approx_kl": approx_kl,
            }
            return total, aux

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (total_loss, loss_info), grads = grad_fn(state.params)
        grad_global_norm = optax.global_norm(grads)
        finite_grads = jnp.all(
            jnp.array([
                jnp.all(jnp.isfinite(x))
                for x in jax.tree_util.tree_leaves(grads)
            ])
        )
        finite_loss = jnp.all(
            jnp.array([jnp.all(jnp.isfinite(x)) for x in loss_info.values()])
        )
        should_apply = (
            finite_grads
            & finite_loss
            & jnp.isfinite(total_loss)
            & jnp.isfinite(grad_global_norm)
        )
        target_kl = jnp.asarray(self.config.target_kl, dtype=loss_info["approx_kl"].dtype)
        kl_ok = (target_kl <= 0.0) | (loss_info["approx_kl"] <= target_kl)
        kl_rejected = should_apply & ~kl_ok
        should_apply = should_apply & kl_ok

        state = jax.lax.cond(
            should_apply,
            lambda s: s.apply_gradients(grads=grads),
            lambda s: s,
            state,
        )
        loss_info = {
            **loss_info,
            "grad_global_norm": grad_global_norm,
            "applied": should_apply,
            "kl_rejected": kl_rejected,
        }
        return state, loss_info

    # ------------------------------------------------------------------
    # Curriculum
    # ------------------------------------------------------------------

    def _maybe_advance_curriculum(self):
        if not self.config.curriculum_enabled:
            return
        window = max(1, int(self.config.curriculum_window))
        recent = self.success_history[-window:]
        self.window_success_rate = sum(recent) / len(recent) if recent else 0.0

        min_dwell = max(0, int(self.config.curriculum_min_dwell_iterations))
        if self.dwell_iterations_at_level < min_dwell:
            return

        if len(self.success_history) < window:
            return

        rate = self.window_success_rate
        if (rate >= self.config.advance_threshold
                and self.current_complexity < self.config.max_complexity):
            self.current_complexity += 1
            self.success_history.clear()
            self.dwell_iterations_at_level = 0
            self.window_success_rate = 0.0
            self.backoff_patience_counter = 0
            print(f"[Curriculum] Advanced to complexity {self.current_complexity}")
            return

        if self.config.backoff_threshold < 0:
            self.backoff_patience_counter = 0
            return

        if (rate <= self.config.backoff_threshold
                and self.current_complexity > self.config.starting_complexity):
            patience = max(0, int(self.config.curriculum_backoff_patience_iterations))
            self.backoff_patience_counter += 1
            if self.backoff_patience_counter < patience:
                return
            self.current_complexity -= 1
            self.success_history.clear()
            self.dwell_iterations_at_level = 0
            self.window_success_rate = 0.0
            self.backoff_patience_counter = 0
            print(f"[Curriculum] Backed off to complexity {self.current_complexity}")
        else:
            self.backoff_patience_counter = 0

    # ------------------------------------------------------------------
    # Checkpoint save / load
    # ------------------------------------------------------------------

    def save_checkpoint(
        self,
        path: str,
        checkpoint_metadata: Optional[dict] = None,
    ) -> None:
        """Save model params, optimizer state, and config to a pickle file.

        Args:
            path: File path (e.g. 'results/.../checkpoint_00050.pkl').
            checkpoint_metadata: Optional run metadata to store with the checkpoint.
        """
        state_dict = {
            "params": self.train_state.params,
            "opt_state": self.train_state.opt_state,
            "step": int(self.train_state.step),
            "config": self.config,
            "fixed_complexities": self.fixed_complexities,
            "current_complexity": self.current_complexity,
            "dwell_iterations_at_level": self.dwell_iterations_at_level,
            "window_success_rate": self.window_success_rate,
            "backoff_patience_counter": self.backoff_patience_counter,
            "skipped_minibatch_updates_total": self.skipped_minibatch_updates_total,
            "skipped_outer_iterations": self.skipped_outer_iterations,
            "checkpoint_metadata": dict(checkpoint_metadata or {}),
            "algorithm": "ppo-mcts-jax",
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(state_dict, f)

    def load_checkpoint(self, path: str) -> None:
        """Load model params and optimizer state from a pickle checkpoint.

        Args:
            path: Path to a checkpoint saved by save_checkpoint().
        """
        with open(path, "rb") as f:
            state_dict = pickle.load(f)
        self.train_state = self.train_state.replace(
            params=state_dict["params"],
            opt_state=state_dict["opt_state"],
            step=state_dict["step"],
        )
        self.current_complexity = state_dict.get(
            "current_complexity", self.current_complexity
        )
        self.dwell_iterations_at_level = state_dict.get(
            "dwell_iterations_at_level", self.dwell_iterations_at_level
        )
        self.window_success_rate = state_dict.get(
            "window_success_rate", self.window_success_rate
        )
        self.backoff_patience_counter = state_dict.get(
            "backoff_patience_counter", self.backoff_patience_counter
        )
        self.skipped_minibatch_updates_total = state_dict.get(
            "skipped_minibatch_updates_total",
            self.skipped_minibatch_updates_total,
        )
        self.skipped_outer_iterations = state_dict.get(
            "skipped_outer_iterations", self.skipped_outer_iterations
        )

    # ------------------------------------------------------------------
    # Main train loop
    # ------------------------------------------------------------------

    def train(self, num_iterations: int,
              results_dir: str = "results") -> dict:
        """Run the full PPO+MCTS (JAX) training loop.

        Each iteration:
          1. Collect batch_size episodes using batched MCTS.
          2. Compute GAE.
          3. PPO update.
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
            "weighted_vf_loss": [], "pg_to_weighted_vf_ratio": [],
            "success_rate": [], "avg_reward": [], "complexity": [],
            "on_path_phi": [], "on_path_hits": [], "episode_length": [],
            "dwell_iterations_at_level": [], "window_success_rate": [],
            "max_abs_logit": [], "max_log_ratio": [], "max_return": [],
            "max_abs_advantage": [], "grad_global_norm": [],
            "approx_kl": [], "kl_early_stop_count": [],
            "kl_rejected_updates": [], "kl_rejection_rate": [],
            "skipped_minibatch_updates": [], "skipped_outer_iterations": [],
            "success_count": [], "failure_with_hit_count": [],
            "failure_no_hit_count": [], "p_hit": [], "p_solve_given_hit": [],
            "avg_return_success": [], "avg_return_failure_with_hit": [],
            "avg_return_failure_no_hit": [], "avg_phi_success": [],
            "avg_phi_failure_with_hit": [],
            "avg_episode_length_success": [],
            "avg_episode_length_failure_with_hit": [],
            "avg_episode_length_failure_no_hit": [],
        }
        # Per-complexity history when using fixed_complexities.
        if self.fixed_complexities:
            for c in self.fixed_complexities:
                history[f"success_rate_C{c}"] = []
                history[f"avg_reward_C{c}"] = []

        print("Iteration 1: JIT-compiling MCTS + env (this may take a few minutes)...",
              flush=True)

        def log_line(msg: str) -> None:
            if self.config.disable_progress_bar:
                print(msg, flush=True)
            else:
                tqdm.write(msg)

        pbar = tqdm(
            range(1, num_iterations + 1),
            desc="Training",
            unit="iter",
            disable=self.config.disable_progress_bar,
        )
        best_trailing_success_mean = -math.inf
        best_metric_key = "success_rate"
        if self.fixed_complexities and len(self.fixed_complexities) == 1:
            best_metric_key = f"success_rate_C{self.fixed_complexities[0]}"

        for iteration in pbar:
            iter_start = time.time()

            success_len_before = len(self.success_history)
            transitions, rollout_info = self.collect_rollouts()

            if iteration == 1:
                print(f"  Rollout done ({time.time() - iter_start:.1f}s). "
                      "Running PPO update...", flush=True)

            advantages, returns = self.compute_gae(transitions)
            if self._rollout_data_is_finite(transitions, advantages, returns):
                loss_info = self.update(transitions, advantages, returns)
                if loss_info["skipped_outer_iteration"]:
                    self.skipped_outer_iterations += 1
                    del self.success_history[success_len_before:]
            else:
                self.skipped_outer_iterations += 1
                del self.success_history[success_len_before:]
                loss_info = self._empty_loss_info(skipped_outer=True)

            if self.config.curriculum_enabled and not loss_info["skipped_outer_iteration"]:
                self.dwell_iterations_at_level += 1
                self._maybe_advance_curriculum()

            iter_time = time.time() - iter_start
            if iteration == 1:
                print(f"  Iteration 1 complete ({iter_time:.1f}s). "
                      "Subsequent iterations should be much faster.", flush=True)

            # Update tqdm postfix with key metrics.
            if not self.config.disable_progress_bar:
                pbar.set_postfix({
                    "success": f"{rollout_info['success_rate']:.1%}",
                    "reward": f"{rollout_info['avg_reward']:.2f}",
                    "entropy": f"{loss_info['entropy']:.2f}",
                    "s/iter": f"{iter_time:.1f}",
                })

            history["pg_loss"].append(loss_info["pg_loss"])
            history["vf_loss"].append(loss_info["vf_loss"])
            history["entropy"].append(loss_info["entropy"])
            history["weighted_vf_loss"].append(loss_info["weighted_vf_loss"])
            history["pg_to_weighted_vf_ratio"].append(
                loss_info["pg_to_weighted_vf_ratio"]
            )
            history["success_rate"].append(rollout_info["success_rate"])
            history["avg_reward"].append(rollout_info["avg_reward"])
            history["complexity"].append(rollout_info["complexity"])
            history["on_path_phi"].append(rollout_info["on_path_phi"])
            history["on_path_hits"].append(rollout_info["on_path_hits"])
            history["episode_length"].append(rollout_info["episode_length"])
            history["dwell_iterations_at_level"].append(
                self.dwell_iterations_at_level
            )
            history["window_success_rate"].append(self.window_success_rate)
            history["max_abs_logit"].append(loss_info["max_abs_logit"])
            history["max_log_ratio"].append(loss_info["max_log_ratio"])
            history["max_return"].append(loss_info["max_return"])
            history["max_abs_advantage"].append(loss_info["max_abs_advantage"])
            history["grad_global_norm"].append(loss_info["grad_global_norm"])
            history["approx_kl"].append(loss_info["approx_kl"])
            history["kl_early_stop_count"].append(
                loss_info["kl_early_stop_count"]
            )
            history["kl_rejected_updates"].append(
                loss_info["kl_rejected_updates"]
            )
            history["kl_rejection_rate"].append(loss_info["kl_rejection_rate"])
            history["skipped_minibatch_updates"].append(
                loss_info["skipped_minibatch_updates"]
            )
            history["skipped_outer_iterations"].append(self.skipped_outer_iterations)
            for key in (
                "success_count",
                "failure_with_hit_count",
                "failure_no_hit_count",
                "p_hit",
                "p_solve_given_hit",
                "avg_return_success",
                "avg_return_failure_with_hit",
                "avg_return_failure_no_hit",
                "avg_phi_success",
                "avg_phi_failure_with_hit",
                "avg_episode_length_success",
                "avg_episode_length_failure_with_hit",
                "avg_episode_length_failure_no_hit",
            ):
                history[key].append(rollout_info[key])

            if self.fixed_complexities:
                for c in self.fixed_complexities:
                    history[f"success_rate_C{c}"].append(
                        rollout_info.get(f"success_rate_C{c}", 0.0)
                    )
                    history[f"avg_reward_C{c}"].append(
                        rollout_info.get(f"avg_reward_C{c}", 0.0)
                    )

            if (
                iteration > 50
                and rollout_info["success_count"] > 0
                and rollout_info["avg_phi_success"] < 0.95
            ):
                log_line(
                    "  WARNING: avg_phi_success below 0.95 "
                    f"({rollout_info['avg_phi_success']:.3f}); "
                    "check OnPath hit accounting"
                )

            if len(history[best_metric_key]) >= 10:
                trailing_success_mean = self._trailing_mean(
                    history[best_metric_key], 10
                )
                if (
                    trailing_success_mean is not None
                    and trailing_success_mean > best_trailing_success_mean
                ):
                    best_trailing_success_mean = trailing_success_mean
                    best_ckpt_path = os.path.join(results_dir, "checkpoint_best.pkl")
                    self.save_checkpoint(
                        best_ckpt_path,
                        checkpoint_metadata={
                            "kind": "best_trailing_success",
                            "metric_key": best_metric_key,
                            "trailing_window": 10,
                            "best_score": best_trailing_success_mean,
                            "iteration": iteration,
                        },
                    )

            if self.config.wandb_enabled:
                import wandb
                log_dict = {
                    "iteration": iteration,
                    "pg_loss": loss_info["pg_loss"],
                    "vf_loss": loss_info["vf_loss"],
                    "entropy": loss_info["entropy"],
                    "weighted_vf_loss": loss_info["weighted_vf_loss"],
                    "pg_to_weighted_vf_ratio": (
                        loss_info["pg_to_weighted_vf_ratio"]
                    ),
                    "success_rate": rollout_info["success_rate"],
                    "avg_reward": rollout_info["avg_reward"],
                    "episodes": rollout_info["episodes"],
                    "factor_hits": rollout_info["factor_hits"],
                    "library_hits": rollout_info["library_hits"],
                    "library_size": rollout_info["library_size"],
                    "on_path_hits": rollout_info["on_path_hits"],
                    "on_path_phi": rollout_info["on_path_phi"],
                    "target_board_step": rollout_info["target_board_step"],
                    "episode_length": rollout_info["episode_length"],
                    "rollout_complexity": rollout_info["complexity"],
                    "current_complexity_after_curriculum": self.current_complexity,
                    "dwell_iterations_at_level": self.dwell_iterations_at_level,
                    "window_success_rate": self.window_success_rate,
                    "max_abs_logit": loss_info["max_abs_logit"],
                    "max_log_ratio": loss_info["max_log_ratio"],
                    "max_return": loss_info["max_return"],
                    "max_abs_advantage": loss_info["max_abs_advantage"],
                    "grad_global_norm": loss_info["grad_global_norm"],
                    "approx_kl": loss_info["approx_kl"],
                    "kl_early_stop_count": loss_info["kl_early_stop_count"],
                    "kl_rejected_updates": loss_info["kl_rejected_updates"],
                    "kl_rejection_rate": loss_info["kl_rejection_rate"],
                    "skipped_minibatch_updates": (
                        loss_info["skipped_minibatch_updates"]
                    ),
                    "skipped_outer_iterations": self.skipped_outer_iterations,
                    "train/by_outcome/success_count": (
                        rollout_info["success_count"]
                    ),
                    "train/by_outcome/failure_with_hit_count": (
                        rollout_info["failure_with_hit_count"]
                    ),
                    "train/by_outcome/failure_no_hit_count": (
                        rollout_info["failure_no_hit_count"]
                    ),
                    "train/by_outcome/P_hit": rollout_info["p_hit"],
                    "train/by_outcome/P_solve_given_hit": (
                        rollout_info["p_solve_given_hit"]
                    ),
                    "train/by_outcome/avg_return_success": (
                        rollout_info["avg_return_success"]
                    ),
                    "train/by_outcome/avg_return_failure_with_hit": (
                        rollout_info["avg_return_failure_with_hit"]
                    ),
                    "train/by_outcome/avg_return_failure_no_hit": (
                        rollout_info["avg_return_failure_no_hit"]
                    ),
                    "train/by_outcome/avg_phi_success": (
                        rollout_info["avg_phi_success"]
                    ),
                    "train/by_outcome/avg_phi_failure_with_hit": (
                        rollout_info["avg_phi_failure_with_hit"]
                    ),
                }
                if self.fixed_complexities:
                    for c in self.fixed_complexities:
                        log_dict[f"success_rate_C{c}"] = rollout_info.get(
                            f"success_rate_C{c}", 0.0
                        )
                        log_dict[f"avg_reward_C{c}"] = rollout_info.get(
                            f"avg_reward_C{c}", 0.0
                        )
                wandb.log(log_dict, step=iteration)

            if iteration % self.config.log_interval == 0:
                line = (
                    f"[PPO+MCTS-JAX iter {iteration}] "
                    f"reward_mode={self.config.reward_mode} "
                    f"rollout_complexity={rollout_info['complexity']} "
                    f"current_complexity_after_curriculum={self.current_complexity} "
                    f"dwell_iterations_at_level={self.dwell_iterations_at_level} "
                    f"window_success_rate={self.window_success_rate:.2%} "
                    f"episodes={rollout_info['episodes']} "
                    f"lib={rollout_info['library_size']} "
                    f"fhits={rollout_info['factor_hits']} "
                    f"lhits={rollout_info['library_hits']} "
                    f"onpath_hits={rollout_info['on_path_hits']} "
                    f"phi={rollout_info['on_path_phi']:.3f} "
                    f"success={rollout_info['success_rate']:.2%} "
                    f"reward={rollout_info['avg_reward']:.3f} "
                    f"pg_loss={loss_info['pg_loss']:.4f} "
                    f"vf_loss={loss_info['vf_loss']:.4f} "
                    f"wvf={loss_info['weighted_vf_loss']:.4f} "
                    f"pg/wvf={loss_info['pg_to_weighted_vf_ratio']:.4f} "
                    f"entropy={loss_info['entropy']:.4f} "
                    f"max_log_ratio={loss_info['max_log_ratio']:.2f} "
                    f"approx_kl={loss_info['approx_kl']:.4f} "
                    f"grad_norm={loss_info['grad_global_norm']:.3f} "
                    f"kl_stop={loss_info['kl_early_stop_count']} "
                    f"kl_rej={loss_info['kl_rejected_updates']} "
                    f"kl_rej_rate={loss_info['kl_rejection_rate']:.1%} "
                    f"skip_mb={loss_info['skipped_minibatch_updates']} "
                    f"skip_outer={self.skipped_outer_iterations} "
                    f"P(hit)={rollout_info['p_hit']:.1%} "
                    f"P(solve|hit)={rollout_info['p_solve_given_hit']:.1%} "
                    f"R_succ={rollout_info['avg_return_success']:+.1f} "
                    f"R_fail_hit={rollout_info['avg_return_failure_with_hit']:+.1f} "
                    f"R_fail={rollout_info['avg_return_failure_no_hit']:+.1f} "
                    f"({iter_time:.1f}s/iter)"
                )
                if self.fixed_complexities:
                    parts = []
                    for c in self.fixed_complexities:
                        sr = rollout_info.get(f"success_rate_C{c}", 0.0)
                        parts.append(f"C{c}={sr:.1%}")
                    line += " | " + " ".join(parts)
                log_line(line)

            # Save periodic checkpoints.
            if iteration % checkpoint_interval == 0 or iteration == num_iterations:
                ckpt_path = os.path.join(
                    results_dir, f"checkpoint_{iteration:05d}.pkl"
                )
                self.save_checkpoint(ckpt_path)
                log_line(f"  Saved checkpoint: {ckpt_path}")

        return history
