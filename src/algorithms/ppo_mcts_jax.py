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
from dataclasses import dataclass
from typing import List, Optional, Tuple

import jax
import jax.numpy as jnp
import mctx
import numpy as np
import optax
import flax.linen as nn
from flax.training import train_state

from ..config import Config
from ..environment.fast_polynomial import FastPoly
from ..game_board.generator import sample_target, build_game_board
from .jax_env import (
    EnvConfig, EnvState, make_env_config,
    reset as env_reset, step as env_step,
    get_observation, get_valid_actions_mask,
    decode_action, poly_add, poly_mul,
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

    Attributes:
        config: Shared hyperparameter configuration.
        env_config: JAX-compatible environment config.
        network: Flax policy-value network.
        tx: optax optimizer.
        train_state: Flax TrainState with params and optimizer state.
        batch_size: Number of parallel environments for data collection.
    """

    def __init__(self, config: Config, device: str = "cpu",
                 batch_size: int = 64) -> None:
        """Initialize the JAX PPO+MCTS trainer.

        Args:
            config: Configuration dataclass with all hyperparameters.
            device: Ignored (JAX auto-detects GPU/TPU).
            batch_size: Number of parallel environments during rollouts.
        """
        self.config = config
        self.env_config = make_env_config(config)
        self.batch_size = batch_size

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

        # Curriculum state.
        self.current_complexity = (
            config.starting_complexity if config.curriculum_enabled
            else config.max_complexity
        )
        self.success_history: List[bool] = []

        # Lazily-built game boards.
        self._boards = {}

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

    def _get_board(self, complexity: int) -> dict:
        if complexity not in self._boards:
            self._boards[complexity] = build_game_board(self.config, complexity)
        return self._boards[complexity]

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

    def _recurrent_fn(self, params, rng_key, action, embedding):
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
        next_states, rewards, dones, successes = jax.vmap(
            lambda s, a: env_step(self.env_config, s, a)
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

    def _batched_mcts_search(self, params, rng_key, obs_batch, env_states):
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

        policy_output = mctx.muzero_policy(
            params=params,
            rng_key=rng_key,
            root=root,
            recurrent_fn=self._recurrent_fn,
            num_simulations=self.config.mcts_simulations,
            max_depth=self.config.max_steps,
            invalid_actions=~obs_batch['mask'],  # True = invalid
        )
        return policy_output

    # ------------------------------------------------------------------
    # Data collection
    # ------------------------------------------------------------------

    def _sample_targets(self, n: int) -> List[FastPoly]:
        """Sample n target polynomials from the game board."""
        board = self._get_board(self.current_complexity)
        targets = []
        for _ in range(n):
            poly, _ = sample_target(self.config, self.current_complexity, board)
            targets.append(poly)
        return targets

    def _fastpoly_to_jax(self, poly: FastPoly) -> jnp.ndarray:
        """Convert FastPoly to flat JAX int32 coefficient array."""
        return jnp.array(poly.coeffs.flatten(), dtype=jnp.int32)

    def collect_rollouts(self) -> Tuple[List[Transition], dict]:
        """Collect rollout data using batched MCTS for action selection.

        Runs self.batch_size parallel episodes. At each step, all B
        environments are searched simultaneously via mctx.

        Returns:
            (transitions, rollout_info) where transitions is a flat list
            and rollout_info has episode statistics.
        """
        B = self.batch_size
        params = self.train_state.params
        rng = jax.random.PRNGKey(
            np.random.randint(0, 2**31)
        )

        # Sample targets and reset environments.
        targets = self._sample_targets(B)
        target_arrays = jnp.stack(
            [self._fastpoly_to_jax(t) for t in targets], axis=0
        )  # (B, target_size)

        # Batch reset.
        states = jax.vmap(
            lambda tc: env_reset(self.env_config, tc)
        )(target_arrays)

        transitions = []
        episode_rewards = np.zeros(B)
        episode_successes = np.zeros(B, dtype=bool)
        active = np.ones(B, dtype=bool)  # Track which envs are still running.

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
            next_states, rewards, dones, successes = jax.vmap(
                lambda s, a: env_step(self.env_config, s, a)
            )(states, actions)

            # Convert to numpy for storage.
            actions_np = np.array(actions)
            rewards_np = np.array(rewards)
            dones_np = np.array(dones)
            successes_np = np.array(successes)
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
                    if dones_np[i]:
                        active[i] = False

            states = next_states

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
        }
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
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

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
        num_updates = 0

        bs = self.config.batch_size

        for epoch in range(self.config.ppo_epochs):
            perm = np.random.permutation(n)

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

                total_pg_loss += float(loss_info['pg_loss'])
                total_vf_loss += float(loss_info['vf_loss'])
                total_entropy += float(loss_info['entropy'])
                num_updates += 1

        return {
            "pg_loss": total_pg_loss / max(num_updates, 1),
            "vf_loss": total_vf_loss / max(num_updates, 1),
            "entropy": total_entropy / max(num_updates, 1),
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
            ratio = jnp.exp(new_lp - old_log_probs)
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

            return total, {'pg_loss': pg_loss, 'vf_loss': vf_loss, 'entropy': entropy}

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (_, loss_info), grads = grad_fn(state.params)
        state = state.apply_gradients(grads=grads)
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
    # Main train loop
    # ------------------------------------------------------------------

    def train(self, num_iterations: int) -> dict:
        """Run the full PPO+MCTS (JAX) training loop.

        Each iteration:
          1. Collect batch_size episodes using batched MCTS.
          2. Compute GAE.
          3. PPO update.
          4. Adjust curriculum.

        Args:
            num_iterations: Number of collect + update cycles.

        Returns:
            History dict with metric lists.
        """
        history = {
            "pg_loss": [], "vf_loss": [], "entropy": [],
            "success_rate": [], "avg_reward": [], "complexity": [],
        }

        for iteration in range(1, num_iterations + 1):
            transitions, rollout_info = self.collect_rollouts()
            advantages, returns = self.compute_gae(transitions)
            loss_info = self.update(transitions, advantages, returns)
            self._maybe_advance_curriculum()

            history["pg_loss"].append(loss_info["pg_loss"])
            history["vf_loss"].append(loss_info["vf_loss"])
            history["entropy"].append(loss_info["entropy"])
            history["success_rate"].append(rollout_info["success_rate"])
            history["avg_reward"].append(rollout_info["avg_reward"])
            history["complexity"].append(rollout_info["complexity"])

            if self.config.wandb_enabled:
                import wandb
                wandb.log({
                    "iteration": iteration,
                    "pg_loss": loss_info["pg_loss"],
                    "vf_loss": loss_info["vf_loss"],
                    "entropy": loss_info["entropy"],
                    "success_rate": rollout_info["success_rate"],
                    "avg_reward": rollout_info["avg_reward"],
                    "complexity": rollout_info["complexity"],
                    "episodes": rollout_info["episodes"],
                }, step=iteration)

            if iteration % self.config.log_interval == 0:
                print(
                    f"[PPO+MCTS-JAX iter {iteration}] "
                    f"complexity={rollout_info['complexity']} "
                    f"episodes={rollout_info['episodes']} "
                    f"success={rollout_info['success_rate']:.2%} "
                    f"reward={rollout_info['avg_reward']:.3f} "
                    f"pg_loss={loss_info['pg_loss']:.4f} "
                    f"vf_loss={loss_info['vf_loss']:.4f} "
                    f"entropy={loss_info['entropy']:.4f}"
                )

        return history
