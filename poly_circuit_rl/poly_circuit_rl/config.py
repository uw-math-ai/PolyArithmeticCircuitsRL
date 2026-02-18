"""Central configuration dataclass for the poly-circuit RL project.

All hyperparameters live here as a single frozen dataclass so that
experiments are fully reproducible and configurations can be passed
as a single object.  Derived quantities (obs_dim, action_dim, etc.)
are computed as properties to avoid redundancy.

See configs/default.yaml for documentation of each group of parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

# Sentinel: tanh scale of 0.0 disables normalization (raw float passthrough).
_DEFAULT_EVAL_NORM_SCALE: float = 100.0

from .core.action_codec import action_space_size


@dataclass(frozen=True)
class Config:
    """Frozen hyperparameter config for training and environment setup.

    Groups:
        Environment:  n_vars, max_ops, L, max_nodes, m, eval_low/high, step_cost
        Transformer:  d_pos, d_model, n_heads, n_layers, dropout
        DQN:          lr, gamma, batch_size, buffer_size, eps_*, target_update_tau,
                      train_freq, learning_starts
        HER:          her_k
        Curriculum:   curriculum_levels, curriculum_window, curriculum_threshold
        Training:     total_steps, eval_every, eval_episodes, seed, log_dir
        Sampling:     interesting_ratio
    """
    # --- Environment ---
    n_vars: int = 2
    max_ops: int = 4
    L: int = 16               # max visible nodes in observation
    max_nodes: int = 20       # hard cap on circuit nodes
    m: int = 16               # number of evaluation points
    eval_low: int = -3
    eval_high: int = 3
    step_cost: float = 0.05   # per-op penalty for ADD/MUL
    shaping_coeff: float = 0.3  # eval-distance reward shaping bonus
    eval_norm_scale: float = _DEFAULT_EVAL_NORM_SCALE  # tanh(v/scale) applied to eval vectors in obs
    max_episode_steps: Optional[int] = None  # hard cap; None → max_ops + max_nodes + 5

    # --- Node encoding ---
    # d_node_raw: per-node feature dim in flat obs (before embedding)
    #   type_onehot(3) + op_onehot(2) + parent1_idx(1) + parent2_idx(1)
    #   + pos_idx(1) + leaf_id(n_vars+1) + eval_vector(m)

    # --- Transformer ---
    d_pos: int = 8            # embedding dim for parent/position indices
    d_model: int = 64         # transformer hidden dim
    n_heads: int = 4
    n_layers: int = 3
    dropout: float = 0.1

    # --- DQN ---
    lr: float = 1e-3
    gamma: float = 0.99
    batch_size: int = 256
    buffer_size: int = 100_000
    eps_start: float = 1.0
    eps_end: float = 0.1
    eps_decay_steps: int = 50_000
    target_update_tau: float = 0.005
    train_freq: int = 4
    learning_starts: int = 1000

    # --- HER ---
    her_k: int = 4

    # --- Curriculum ---
    curriculum_levels: Tuple[int, ...] = (1, 2, 3, 4, 5, 6)
    curriculum_window: int = 200
    curriculum_threshold: float = 0.80

    # --- Training ---
    total_steps: int = 500_000
    eval_every: int = 5000
    eval_episodes: int = 50
    seed: int = 42
    log_dir: str = "runs/"

    # --- Mixed sampling ---
    interesting_ratio: float = 0.7  # fraction of interesting polys at high curriculum

    # --- Auto-generation of interesting polynomials ---
    auto_interesting: bool = True               # auto-generate when no JSONL provided
    gen_max_graph_nodes: Optional[int] = 100_000  # safety cap on graph size (None = unlimited)
    gen_max_successors: Optional[int] = 50        # per-node expansion cap (None = unlimited)

    @property
    def n_leaf_types(self) -> int:
        """Number of leaf types: one per variable + const_1."""
        return self.n_vars + 1

    @property
    def d_node_raw(self) -> int:
        """Raw per-node feature dim in flat observation (before embedding)."""
        return 3 + 2 + 2 + 1 + self.n_leaf_types + self.m

    @property
    def obs_dim(self) -> int:
        """Total flat observation size: L*d_node_raw + m (target) + 1 (steps_left)."""
        return self.L * self.d_node_raw + self.m + 1

    @property
    def action_dim(self) -> int:
        return action_space_size(self.L)

    @property
    def d_node_continuous(self) -> int:
        """Continuous features per node (everything except indices to embed).
        type_onehot(3) + op_onehot(2) + leaf_id(n_leaf_types) + eval_vector(m)."""
        return 3 + 2 + self.n_leaf_types + self.m
