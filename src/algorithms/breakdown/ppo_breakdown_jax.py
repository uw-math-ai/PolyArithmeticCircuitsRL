"""JAX / Flax / optax PPO trainer for the top-down decomposition task.

This is the **JAX counterpart** to :class:`PPOMCTSBreakdownTrainer` (PyTorch)
in ``ppo_mcts_breakdown``. It mirrors the structure of the existing JAX
trainer for the *forward* circuit-construction task in
``src/algorithms/ppo_mcts_jax.py`` — same Flax + optax + ``train_state``
patterns, same ``Config`` knobs — but adapted to the breakdown environment.

Key design constraints
----------------------

* The breakdown env's candidate generator calls
  ``FactorLibrary.factorize_poly`` and ``FactorLibrary.exact_quotient`` at
  every step. Both wrap SymPy and therefore cannot be JIT'd. The forward
  ``jax_env.py`` works around this by *precomputing* factor subgoals once
  per episode (at reset). The breakdown env, by contrast, needs fresh
  factorizations of *new* residuals/cofactors discovered during
  decomposition, so the env step itself stays on host.

* As a result this trainer **does not** use ``mctx`` for batched MCTS the
  way ``ppo_mcts_jax.py`` does. Tree search would either need
  ``jax.pure_callback`` for every leaf expansion (which destroys most of
  the JIT speedup) or a JAX-friendly factorization shim (significant
  separate engineering effort). For MCTS-guided training of the breakdown
  task, use the PyTorch :class:`PPOMCTSBreakdownTrainer` instead.

What we *do* accelerate with JAX
--------------------------------

* **Batched policy/value inference** — at each rollout step we stack the
  observations from B parallel ``BreakdownGame`` instances and run a
  single JIT'd forward pass. This is where the bulk of the inference cost
  sits, so this alone gives a meaningful speedup.
* **JIT'd PPO loss + optax update** — the gradient step on the rollout
  buffer compiles once and then runs at near-GPU throughput.
* **Optional GPU placement** — when JAX detects a GPU it will run the
  network and update on it automatically.

The combined effect is a JAX trainer that scales to larger ``batch_size``
and larger networks than the PyTorch breakdown trainers without changing
the env or the search algorithm.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

# JAX deps — imported at module scope to mirror the existing
# ``ppo_mcts_jax.py`` trainer. If these aren't installed, importing this
# module will fail but the rest of the package keeps working because
# the ``__init__.py`` does not import it eagerly.
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.training import train_state

from ...config import Config
from ...environment.factor_library import FactorLibrary
from ...environment.fast_polynomial import FastPoly
from ...game_board.generator import (
    build_game_board,
    generate_random_circuit,
    sample_target,
)

from .breakdown_env import (
    CANDIDATE_FEATURE_DIM,
    CONTEXT_FEATURE_DIM,
    BreakdownGame,
    BreakdownObservation,
)


# ----------------------------------------------------------------------
# Flax network — JAX counterpart of BreakdownPolicyValueNet (PyTorch).
# ----------------------------------------------------------------------


class BreakdownPolicyValueNetJax(nn.Module):
    """Flax policy-value network for the :class:`BreakdownGame`.

    Same architecture as the PyTorch ``BreakdownPolicyValueNet`` so the two
    trainers stay comparable: three-stream MLP encoder (focus, target,
    context) → fused embedding, broadcast across the K candidate slots and
    concatenated with per-candidate features for a per-slot scoring MLP.

    Inputs (all expected to carry a leading batch axis):

    * ``focus``       — ``[B, target_size]``
    * ``target``      — ``[B, target_size]``
    * ``context``     — ``[B, context_dim]``
    * ``cand_feats``  — ``[B, K, candidate_feature_dim]``
    * ``mask``        — ``[B, K]`` (bool)

    Outputs:

    * ``logits``  — ``[B, K]`` with masked positions set to ``-1e9``.
    * ``value``   — ``[B]``.
    """

    target_size: int
    context_dim: int = CONTEXT_FEATURE_DIM
    candidate_feature_dim: int = CANDIDATE_FEATURE_DIM
    max_options: int = 32
    hidden_dim: int = 256
    embedding_dim: int = 256

    @nn.compact
    def __call__(self, obs: dict) -> Tuple[jnp.ndarray, jnp.ndarray]:
        focus = obs["focus"]
        target = obs["target"]
        context = obs["context"]
        cand_feats = obs["cand_feats"]
        mask = obs["mask"]

        # --- Three-stream encoder ---
        f = nn.Dense(self.hidden_dim, name="focus_0")(focus)
        f = nn.relu(f)
        f = nn.Dense(self.embedding_dim, name="focus_1")(f)

        t = nn.Dense(self.hidden_dim, name="target_0")(target)
        t = nn.relu(t)
        t = nn.Dense(self.embedding_dim, name="target_1")(t)

        c = nn.Dense(self.hidden_dim, name="context_0")(context)
        c = nn.relu(c)
        c = nn.Dense(self.embedding_dim, name="context_1")(c)

        fused = jnp.concatenate([f, t, c], axis=-1)
        fused = nn.Dense(self.hidden_dim, name="fusion_0")(fused)
        fused = nn.relu(fused)
        fused = nn.Dense(self.embedding_dim, name="fusion_1")(fused)
        fused = nn.relu(fused)

        # --- Per-candidate scorer ---
        # fused: [B, emb] → expand to [B, K, emb], concat with cand_feats,
        # then run a small MLP per slot.
        b = fused.shape[0]
        k = cand_feats.shape[1]
        fused_rep = jnp.broadcast_to(fused[:, None, :], (b, k, self.embedding_dim))
        merged = jnp.concatenate([fused_rep, cand_feats], axis=-1)

        s = nn.Dense(self.hidden_dim, name="score_0")(merged)
        s = nn.relu(s)
        s = nn.Dense(1, name="score_1")(s).squeeze(-1)  # [B, K]

        logits = jnp.where(mask, s, -1e9)

        # --- Value head (state-only) ---
        v = nn.Dense(self.hidden_dim, name="value_0")(fused)
        v = nn.relu(v)
        v = nn.Dense(1, name="value_1")(v).squeeze(-1)  # [B]

        return logits, v


# ----------------------------------------------------------------------
# Helpers: numpy <-> jax obs batching
# ----------------------------------------------------------------------


def _stack_obs_to_arrays(obs_list: List[BreakdownObservation]) -> Dict[str, np.ndarray]:
    """Stack a list of breakdown observations into a numpy obs dict.

    The result has the same keys as the JAX network expects, with shapes
    ``[B, ...]``. Conversion to ``jnp.ndarray`` is left to the caller so
    we can keep all numpy-side bookkeeping cheap.
    """
    return {
        "focus": np.stack([o.focus_vec for o in obs_list], axis=0).astype(np.float32),
        "target": np.stack([o.target_vec for o in obs_list], axis=0).astype(np.float32),
        "context": np.stack([o.context_vec for o in obs_list], axis=0).astype(np.float32),
        "cand_feats": np.stack(
            [o.candidate_features for o in obs_list], axis=0
        ).astype(np.float32),
        "mask": np.stack([o.mask for o in obs_list], axis=0).astype(np.bool_),
    }


def _arrays_to_jax(obs_arrays: Dict[str, np.ndarray]) -> Dict[str, jnp.ndarray]:
    """Move a numpy obs dict onto the default JAX device."""
    return {k: jnp.asarray(v) for k, v in obs_arrays.items()}


# ----------------------------------------------------------------------
# JIT'd inference and PPO update
# ----------------------------------------------------------------------


@functools.partial(jax.jit, static_argnums=(0,))
def _apply_network(apply_fn, params, obs_jax):
    """One JIT'd network forward pass over a batched obs dict."""
    return apply_fn({"params": params["params"]}, obs_jax)


def _ppo_loss(
    params,
    apply_fn,
    obs_batch: Dict[str, jnp.ndarray],
    actions: jnp.ndarray,
    old_log_probs: jnp.ndarray,
    advantages: jnp.ndarray,
    returns: jnp.ndarray,
    search_targets: jnp.ndarray,
    ppo_clip: float,
    vf_coef: float,
    ent_coef: float,
    distill_coef: float,
):
    """Standard clipped-surrogate PPO loss with optional distillation.

    Mirrors :meth:`PPOMCTSBreakdownTrainer.update`'s loss in JAX form so we
    can JIT the whole step. ``search_targets`` is the optional MCTS-style
    target distribution; for this trainer it is set to a one-hot of the
    chosen action (no MCTS), which makes the distill term a small auxiliary
    cross-entropy on the actually-taken action and is harmless.
    """
    logits, values = apply_fn({"params": params["params"]}, obs_batch)
    log_probs_all = jax.nn.log_softmax(logits, axis=-1)
    new_log_probs = jnp.take_along_axis(
        log_probs_all, actions[:, None], axis=-1
    ).squeeze(-1)

    ratio = jnp.exp(new_log_probs - old_log_probs)
    surr1 = ratio * advantages
    surr2 = jnp.clip(ratio, 1.0 - ppo_clip, 1.0 + ppo_clip) * advantages
    pg_loss = -jnp.mean(jnp.minimum(surr1, surr2))

    vf_loss = jnp.mean((values - returns) ** 2)

    probs = jnp.exp(log_probs_all)
    entropy = -jnp.sum(probs * log_probs_all, axis=-1)
    entropy = jnp.mean(entropy)

    distill = -jnp.sum(
        jnp.where(search_targets > 0, search_targets * log_probs_all, 0.0),
        axis=-1,
    )
    distill_loss = jnp.mean(distill)

    loss = (
        pg_loss
        + vf_coef * vf_loss
        - ent_coef * entropy
        + distill_coef * distill_loss
    )
    return loss, (pg_loss, vf_loss, entropy, distill_loss)


@functools.partial(
    jax.jit,
    static_argnames=("apply_fn", "ppo_clip", "vf_coef", "ent_coef", "distill_coef"),
)
def _ppo_update_step(
    state: train_state.TrainState,
    apply_fn,
    obs_batch: Dict[str, jnp.ndarray],
    actions: jnp.ndarray,
    old_log_probs: jnp.ndarray,
    advantages: jnp.ndarray,
    returns: jnp.ndarray,
    search_targets: jnp.ndarray,
    ppo_clip: float,
    vf_coef: float,
    ent_coef: float,
    distill_coef: float,
):
    """Compute loss, grads, and apply optax update — all JIT'd."""
    grad_fn = jax.value_and_grad(_ppo_loss, has_aux=True)
    (loss, (pg, vf, ent, distill)), grads = grad_fn(
        state.params,
        apply_fn,
        obs_batch,
        actions,
        old_log_probs,
        advantages,
        returns,
        search_targets,
        ppo_clip,
        vf_coef,
        ent_coef,
        distill_coef,
    )
    state = state.apply_gradients(grads=grads)
    return state, {
        "loss": loss,
        "pg_loss": pg,
        "vf_loss": vf,
        "entropy": ent,
        "distill_loss": distill,
    }


# ----------------------------------------------------------------------
# Rollout buffer
# ----------------------------------------------------------------------


@dataclass
class _JaxRolloutStep:
    """One transition collected from a parallel ``BreakdownGame`` rollout."""

    obs: BreakdownObservation
    action: int
    reward: float
    log_prob: float
    value: float
    done: bool


# ----------------------------------------------------------------------
# Trainer
# ----------------------------------------------------------------------


class PPOBreakdownJAXTrainer:
    """JAX/Flax/optax PPO trainer for the breakdown environment.

    Each iteration:

    1. **Collect** ``steps_per_update`` transitions across ``batch_size``
       parallel ``BreakdownGame`` instances (host-side env stepping;
       JIT-batched policy/value inference).
    2. **Compute GAE** advantages and returns on host.
    3. **PPO update** for ``ppo_epochs`` epochs with mini-batch JIT'd
       gradient steps via optax.
    4. **Adjust curriculum** identically to the PyTorch trainer.

    The trainer maintains its own :class:`FactorLibrary` (unique to this
    trainer instance — no sharing with the existing forward
    ``ppo_mcts_jax`` library so experiments stay isolated).

    Attributes:
        config (Config): Shared hyperparameter configuration.
        batch_size (int): Number of parallel ``BreakdownGame`` rollouts.
        network (BreakdownPolicyValueNetJax): Flax policy-value network.
        train_state (flax.training.train_state.TrainState): Optax state.
        envs (List[BreakdownGame]): Pool of host-side decomposition envs.
        factor_library (FactorLibrary): Shared across all ``envs``.
    """

    MAX_BOARD_COMPLEXITY = 4

    def __init__(
        self,
        config: Config,
        batch_size: int = 64,
        max_options: int = 32,
        max_breakdown_steps: int = 16,
        size_penalty_per_node: float = 0.05,
        log_path: Optional[str] = None,
    ) -> None:
        """Initialise the JAX PPO trainer.

        Args:
            config: Shared :class:`Config`. JAX trainer-specific knobs
                (``ppo_lr``, ``ppo_clip``, ``ppo_epochs``, ``vf_coef``,
                ``ent_coef``, ``gumbel_distill_coef``, ``gamma``,
                ``gae_lambda``, ``max_grad_norm``, ``steps_per_update``,
                ``batch_size``, curriculum thresholds...) are read from it.
            batch_size: Number of parallel ``BreakdownGame`` instances run
                during rollout collection. Matches the role of
                ``batch_size`` in :class:`PPOMCTSJAXTrainer`.
            max_options: Width of the candidate action space.
            max_breakdown_steps: Hard step limit per decomposition episode.
            size_penalty_per_node: Soft penalty for trees larger than
                ``config.max_complexity``.
            log_path: Optional file for textual logs.
        """
        self.config = config
        self.batch_size = int(batch_size)
        self.max_options = int(max_options)
        self.max_breakdown_steps = int(max_breakdown_steps)
        self.log_path = log_path

        # Network + initial parameters.
        self.network = BreakdownPolicyValueNetJax(
            target_size=config.target_size,
            context_dim=CONTEXT_FEATURE_DIM,
            candidate_feature_dim=CANDIDATE_FEATURE_DIM,
            max_options=self.max_options,
            hidden_dim=config.hidden_dim,
            embedding_dim=config.embedding_dim,
        )
        rng = jax.random.PRNGKey(config.seed)
        dummy_obs = {
            "focus": jnp.zeros((1, config.target_size), dtype=jnp.float32),
            "target": jnp.zeros((1, config.target_size), dtype=jnp.float32),
            "context": jnp.zeros((1, CONTEXT_FEATURE_DIM), dtype=jnp.float32),
            "cand_feats": jnp.zeros(
                (1, self.max_options, CANDIDATE_FEATURE_DIM), dtype=jnp.float32
            ),
            "mask": jnp.ones((1, self.max_options), dtype=jnp.bool_),
        }
        params = self.network.init(rng, dummy_obs)

        # Optimiser: clip-by-global-norm + Adam, matching ``ppo_mcts_jax``.
        self.tx = optax.chain(
            optax.clip_by_global_norm(config.max_grad_norm),
            optax.adam(config.ppo_lr),
        )
        self.train_state = train_state.TrainState.create(
            apply_fn=self.network.apply,
            params=params,
            tx=self.tx,
        )

        # Shared factor library and a pool of parallel envs.
        self.factor_library = FactorLibrary(
            mod=config.mod,
            n_vars=config.n_variables,
            max_degree=config.effective_max_degree,
        )
        self.envs: List[BreakdownGame] = [
            BreakdownGame(
                config=config,
                factor_library=self.factor_library,
                max_options=self.max_options,
                max_steps=self.max_breakdown_steps,
                size_penalty_per_node=size_penalty_per_node,
            )
            for _ in range(self.batch_size)
        ]

        # Curriculum state.
        self.current_complexity = (
            config.starting_complexity
            if config.curriculum_enabled
            else config.max_complexity
        )
        self.success_history: List[bool] = []

        self._boards: Dict[int, dict] = {}
        self._rng = np.random.default_rng(config.seed)

    # ------------------------------------------------------------------
    # Logging / target sampling
    # ------------------------------------------------------------------

    def _log(self, msg: str) -> None:
        print(msg)
        if self.log_path:
            with open(self.log_path, "a") as fh:
                fh.write(msg + "\n")

    def _get_board(self, complexity: int) -> dict:
        if complexity not in self._boards:
            self._boards[complexity] = build_game_board(self.config, complexity)
        return self._boards[complexity]

    def _sample_target(self, complexity: int) -> FastPoly:
        if complexity <= self.MAX_BOARD_COMPLEXITY:
            board = self._get_board(complexity)
            poly, _ = sample_target(self.config, complexity, board)
            return poly
        poly, _ = generate_random_circuit(self.config, complexity)
        return poly

    # ------------------------------------------------------------------
    # Rollout collection
    # ------------------------------------------------------------------

    def _reset_env(self, env: BreakdownGame) -> Optional[BreakdownObservation]:
        """Sample a fresh target and reset ``env``. Returns ``None`` if the
        target was trivial (already a base node), in which case the env
        must be reset again externally before stepping.
        """
        target = self._sample_target(self.current_complexity)
        obs = env.reset(target)
        if env.done:
            # Trivial target — count immediate success and re-roll.
            self.success_history.append(True)
            return None
        return obs

    def collect_rollouts(self) -> Tuple[List[_JaxRolloutStep], dict]:
        """Run ``batch_size`` envs in lockstep, collecting transitions.

        Each step:

        * Stack observations from all live envs.
        * Run a single JIT'd batched forward pass for logits/values.
        * Sample one action per env from the masked categorical.
        * Step each env on host (this is the SymPy-bound bit).
        * Record the transition; reset envs that finished, and continue
          until the buffer holds at least ``config.steps_per_update``.

        Returns:
            ``(buffer, info)`` analogous to the PyTorch trainer.
        """
        buffer: List[_JaxRolloutStep] = []

        # State for the parallel envs.
        obs_per_env: List[Optional[BreakdownObservation]] = [None] * self.batch_size
        ep_reward: List[float] = [0.0] * self.batch_size

        episodes = 0
        successes = 0
        total_reward = 0.0
        factor_hits = 0
        library_hits = 0
        skipped_trivial = 0

        # Initial reset: keep rolling until every env has a non-trivial state.
        for i in range(self.batch_size):
            while obs_per_env[i] is None:
                target = self._sample_target(self.current_complexity)
                obs = self.envs[i].reset(target)
                if self.envs[i].done:
                    episodes += 1
                    successes += 1
                    skipped_trivial += 1
                    continue
                obs_per_env[i] = obs

        while len(buffer) < self.config.steps_per_update:
            # Stack live observations and run batched inference.
            obs_arrays = _stack_obs_to_arrays(obs_per_env)  # type: ignore[arg-type]
            obs_jax = _arrays_to_jax(obs_arrays)
            logits, values = _apply_network(
                self.network.apply, self.train_state.params, obs_jax
            )
            logits_np = np.asarray(logits)
            values_np = np.asarray(values)

            # Sample one action per env from the masked softmax.
            log_probs_np = logits_np - np.logaddexp.reduce(logits_np, axis=-1, keepdims=True)
            probs_np = np.exp(log_probs_np)
            probs_np = np.where(np.isfinite(probs_np), probs_np, 0.0)
            total = probs_np.sum(axis=-1, keepdims=True)
            probs_np = np.where(total > 0, probs_np / np.maximum(total, 1e-8), probs_np)

            actions = np.empty(self.batch_size, dtype=np.int64)
            chosen_log_probs = np.empty(self.batch_size, dtype=np.float32)
            for i in range(self.batch_size):
                p = probs_np[i]
                if p.sum() <= 0.0:
                    # Degenerate fallback — pick first valid candidate.
                    valid = np.where(obs_arrays["mask"][i])[0]
                    a = int(valid[0]) if len(valid) > 0 else 0
                else:
                    a = int(self._rng.choice(self.max_options, p=p))
                actions[i] = a
                chosen_log_probs[i] = float(log_probs_np[i, a])

            # Step every env on host; record transitions.
            for i in range(self.batch_size):
                env = self.envs[i]
                a = int(actions[i])
                pre_obs = obs_per_env[i]
                next_obs, reward, done, info = env.step(a)
                ep_reward[i] += reward

                buffer.append(
                    _JaxRolloutStep(
                        obs=pre_obs,
                        action=a,
                        reward=float(reward),
                        log_prob=float(chosen_log_probs[i]),
                        value=float(values_np[i]),
                        done=bool(done),
                    )
                )
                if info.get("factor_hit", False):
                    factor_hits += 1
                if info.get("library_hit", False):
                    library_hits += 1

                if done:
                    episode_success = bool(info.get("is_success", False))
                    episodes += 1
                    total_reward += ep_reward[i]
                    successes += int(episode_success)
                    self.success_history.append(episode_success)
                    if episode_success:
                        env.register_decomposition_in_library()
                    ep_reward[i] = 0.0

                    # Reset until we get a non-trivial new target.
                    new_obs: Optional[BreakdownObservation] = None
                    while new_obs is None:
                        target = self._sample_target(self.current_complexity)
                        new_obs = env.reset(target)
                        if env.done:
                            episodes += 1
                            successes += 1
                            skipped_trivial += 1
                            self.success_history.append(True)
                            new_obs = None
                            # Avoid infinite loop on degenerate cases.
                            if episodes > 10 * self.batch_size:
                                break
                    obs_per_env[i] = new_obs
                else:
                    obs_per_env[i] = next_obs

        info_dict = {
            "episodes": episodes,
            "success_rate": successes / max(episodes, 1),
            "avg_reward": total_reward / max(episodes, 1),
            "complexity": self.current_complexity,
            "factor_hits": factor_hits,
            "library_hits": library_hits,
            "library_size": len(self.factor_library),
            "trivial_skipped": skipped_trivial,
        }
        return buffer, info_dict

    # ------------------------------------------------------------------
    # GAE
    # ------------------------------------------------------------------

    def compute_gae(
        self, buffer: List[_JaxRolloutStep]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Standard GAE-λ computation, identical to the PyTorch version.

        We respect episode boundaries (``done=True``) by zeroing the
        bootstrap value and advantage carry across them. Because the buffer
        is filled across multiple parallel envs in arbitrary order, the
        episode boundary signal in ``done`` is what we rely on to keep
        advantages well-defined (the surrogate for tracking each env's
        trajectory separately).
        """
        n = len(buffer)
        advantages = np.zeros(n, dtype=np.float32)
        returns = np.zeros(n, dtype=np.float32)
        last_gae = 0.0
        last_value = 0.0
        for t in reversed(range(n)):
            if buffer[t].done:
                next_value = 0.0
                last_gae = 0.0
            elif t + 1 < n:
                next_value = buffer[t + 1].value
            else:
                next_value = last_value
            delta = (
                buffer[t].reward + self.config.gamma * next_value - buffer[t].value
            )
            last_gae = (
                delta + self.config.gamma * self.config.gae_lambda * last_gae
            )
            advantages[t] = last_gae
            returns[t] = advantages[t] + buffer[t].value
        return advantages, returns

    # ------------------------------------------------------------------
    # PPO update
    # ------------------------------------------------------------------

    def update(
        self,
        buffer: List[_JaxRolloutStep],
        advantages: np.ndarray,
        returns: np.ndarray,
    ) -> dict:
        """JIT'd PPO mini-batch updates over the rollout buffer."""
        n = len(buffer)
        if n == 0:
            return {
                "pg_loss": 0.0,
                "vf_loss": 0.0,
                "entropy": 0.0,
                "distill_loss": 0.0,
            }

        adv = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        actions_np = np.asarray([s.action for s in buffer], dtype=np.int32)
        old_lp_np = np.asarray([s.log_prob for s in buffer], dtype=np.float32)
        adv_np = adv.astype(np.float32)
        ret_np = returns.astype(np.float32)

        # No MCTS: distillation target is one-hot of the chosen action.
        # This keeps the auxiliary loss harmless (collapses to standard
        # cross-entropy on the action that was actually taken).
        search_np = np.zeros((n, self.max_options), dtype=np.float32)
        search_np[np.arange(n), actions_np] = 1.0

        obs_arrays = _stack_obs_to_arrays([s.obs for s in buffer])

        total_pg = 0.0
        total_vf = 0.0
        total_ent = 0.0
        total_distill = 0.0
        n_updates = 0

        for _ in range(self.config.ppo_epochs):
            idx_perm = self._rng.permutation(n)
            for start in range(0, n, self.config.batch_size):
                end = min(start + self.config.batch_size, n)
                bidx = idx_perm[start:end]

                obs_batch = {
                    k: jnp.asarray(v[bidx]) for k, v in obs_arrays.items()
                }
                self.train_state, info = _ppo_update_step(
                    self.train_state,
                    self.network.apply,
                    obs_batch,
                    jnp.asarray(actions_np[bidx]),
                    jnp.asarray(old_lp_np[bidx]),
                    jnp.asarray(adv_np[bidx]),
                    jnp.asarray(ret_np[bidx]),
                    jnp.asarray(search_np[bidx]),
                    float(self.config.ppo_clip),
                    float(self.config.vf_coef),
                    float(self.config.ent_coef),
                    float(self.config.gumbel_distill_coef),
                )
                total_pg += float(info["pg_loss"])
                total_vf += float(info["vf_loss"])
                total_ent += float(info["entropy"])
                total_distill += float(info["distill_loss"])
                n_updates += 1

        return {
            "pg_loss": total_pg / max(n_updates, 1),
            "vf_loss": total_vf / max(n_updates, 1),
            "entropy": total_ent / max(n_updates, 1),
            "distill_loss": total_distill / max(n_updates, 1),
        }

    # ------------------------------------------------------------------
    # Curriculum
    # ------------------------------------------------------------------

    def _maybe_advance_curriculum(self) -> None:
        if not self.config.curriculum_enabled:
            return
        window = 50
        if len(self.success_history) < window:
            return
        rate = sum(self.success_history[-window:]) / window
        if (
            rate >= self.config.advance_threshold
            and self.current_complexity < self.config.max_complexity
        ):
            self.current_complexity += 1
            self.success_history.clear()
            self._log(
                f"[Curriculum] Advanced to complexity {self.current_complexity}"
            )
        elif (
            rate <= self.config.backoff_threshold
            and self.current_complexity > self.config.starting_complexity
        ):
            self.current_complexity -= 1
            self.success_history.clear()
            self._log(
                f"[Curriculum] Backed off to complexity {self.current_complexity}"
            )

    # ------------------------------------------------------------------
    # Public train / evaluate
    # ------------------------------------------------------------------

    def train(self, num_iterations: int) -> dict:
        """Run ``num_iterations`` collect → GAE → PPO update cycles.

        Returns:
            History dict matching the keys of the PyTorch trainers
            (``pg_loss``, ``vf_loss``, ``entropy``, ``success_rate``,
            ``avg_reward``, ``complexity``).
        """
        history = {
            "pg_loss": [],
            "vf_loss": [],
            "entropy": [],
            "success_rate": [],
            "avg_reward": [],
            "complexity": [],
        }

        for it in range(1, num_iterations + 1):
            buffer, info = self.collect_rollouts()
            adv, ret = self.compute_gae(buffer)
            losses = self.update(buffer, adv, ret)
            self._maybe_advance_curriculum()

            history["pg_loss"].append(losses["pg_loss"])
            history["vf_loss"].append(losses["vf_loss"])
            history["entropy"].append(losses["entropy"])
            history["success_rate"].append(info["success_rate"])
            history["avg_reward"].append(info["avg_reward"])
            history["complexity"].append(info["complexity"])

            if it % self.config.log_interval == 0:
                self._log(
                    f"[PPO Breakdown JAX iter {it}] "
                    f"complexity={info['complexity']} "
                    f"episodes={info['episodes']} "
                    f"success={info['success_rate']:.2%} "
                    f"reward={info['avg_reward']:.3f} "
                    f"lib={info['library_size']} "
                    f"fhits={info['factor_hits']} "
                    f"lhits={info['library_hits']} "
                    f"trivial={info['trivial_skipped']} "
                    f"pg_loss={losses['pg_loss']:.4f} "
                    f"vf_loss={losses['vf_loss']:.4f} "
                    f"entropy={losses['entropy']:.4f} "
                    f"distill={losses['distill_loss']:.4f}"
                )

        return history

    def evaluate(
        self,
        complexities: Optional[List[int]] = None,
        num_trials: int = 50,
    ) -> Dict[int, float]:
        """Greedy roll-out evaluation. Mirrors the PyTorch trainer's API."""
        if complexities is None:
            complexities = list(
                range(
                    self.config.starting_complexity,
                    self.config.max_complexity + 1,
                )
            )
        results: Dict[int, float] = {}
        env = self.envs[0]
        for c in complexities:
            successes = 0
            for _ in range(num_trials):
                target = self._sample_target(c)
                obs = env.reset(target)
                if env.done:
                    successes += 1
                    continue
                while not env.done:
                    obs_arrays = _stack_obs_to_arrays([obs])
                    obs_jax = _arrays_to_jax(obs_arrays)
                    logits, _ = _apply_network(
                        self.network.apply, self.train_state.params, obs_jax
                    )
                    a = int(jnp.argmax(logits[0]).item())
                    obs, _, _, info = env.step(a)
                if info.get("is_success", False):
                    successes += 1
            results[c] = successes / max(num_trials, 1)
        return results
