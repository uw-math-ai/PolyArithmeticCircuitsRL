"""Oracle action labeling for the supervised diagnostic.

Given an EnvState in clean_onpath mode, computes which valid actions
produce a polynomial that is on a cached optimal route. The label
generator forward-simulates each candidate action via env_step and
inspects the on_path_hit_flag returned. The mask of "is this an oracle
action?" is then turned into a normalized distribution for cross-entropy
training.

These helpers are also useful for Phase-1 oracle-prior MCTS (mixing the
oracle action distribution into the MCTS root prior).
"""

from typing import Tuple

import jax
import jax.numpy as jnp

from .jax_env import (
    EnvConfig, EnvState, step as env_step, get_observation,
)


def _empty_library(env_config: EnvConfig):
    """Dummy library_coeffs / library_mask for clean_onpath (factor lib disabled)."""
    library_size = 1
    coeffs = jnp.zeros((library_size, env_config.target_size), dtype=jnp.int32)
    mask = jnp.zeros((library_size,), dtype=jnp.bool_)
    return coeffs, mask


def oracle_action_scores(
    env_config: EnvConfig, state: EnvState
) -> jnp.ndarray:
    """Return a boolean (max_actions,) mask of oracle-optimal actions.

    For each action a in [0, max_actions):
      - It is an oracle action iff
        (a) the action is valid given the current state's node count, AND
        (b) executing it produces a polynomial that matches an active
            on-path node (per env_step's on_path_hit_flag).

    The function reuses env_step, so route-consistency / lock-on-first-hit
    rules are honored exactly as during training rollouts.
    """
    library_coeffs, library_mask = _empty_library(env_config)

    def score_action(a):
        out = env_step(env_config, state, a, library_coeffs, library_mask)
        # env_step return tuple index 8 is on_path_hit_flag.
        return out[8]

    actions = jnp.arange(env_config.max_actions, dtype=jnp.int32)
    is_oracle = jax.vmap(score_action)(actions)  # (max_actions,)
    # Reuse the validity mask the policy network sees.
    valid_mask = get_observation(env_config, state)['mask']
    return is_oracle & valid_mask


def oracle_action_distribution(
    env_config: EnvConfig, state: EnvState
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Return (distribution, has_label).

    distribution: (max_actions,) float32; uniform over oracle actions,
        zero on non-oracle actions, sums to 1 if any oracle action exists,
        else all-zero.
    has_label: bool scalar — True iff at least one oracle action exists.
    """
    is_oracle = oracle_action_scores(env_config, state)
    counts = is_oracle.astype(jnp.float32)
    total = counts.sum()
    has_label = total > 0.0
    dist = jnp.where(has_label, counts / jnp.maximum(total, 1.0), counts)
    return dist, has_label


