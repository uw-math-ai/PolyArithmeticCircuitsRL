"""Central configuration dataclass for the poly-circuit RL project.

All hyperparameters live here as a single frozen dataclass so that
experiments are fully reproducible and configurations can be passed
as a single object.  Derived quantities (obs_dim, action_dim, etc.)
are computed as properties to avoid redundancy.

See configs/default.yaml for documentation of each group of parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

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
        Curriculum:   curriculum_levels, curriculum_window,
                      curriculum_train_threshold, curriculum_eval_threshold
        Training:     total_steps, eval_every, eval_episodes, seed, log_dir
        Sampling:     interesting_ratio
    """
    # --- Environment ---
    n_vars: int = 2
    max_ops: int = 6
    L: int = 16               # max visible nodes in observation
    max_nodes: int = 20       # hard cap on circuit nodes
    m: int = 16               # number of evaluation points
    eval_low: int = -3
    eval_high: int = 3
    step_cost: float = 0.05   # per-op penalty for ADD/MUL
    shaping_coeff: float = 0.0  # eval-distance reward shaping bonus (disabled; can mislead toward naive paths)
    factor_shaping_coeff: float = 0.0  # penalty for ADD producing factorizable result
    reward_mode: Literal["sparse", "shaped", "full"] = "full"

    # --- Factor Library ---
    factor_library_enabled: bool = True       # enable cross-episode factor subgoal rewards
    factor_subgoal_reward: float = 0.3        # reward per distinct subgoal match
    factor_library_bonus: float = 0.15        # extra when subgoal was library-known
    completion_bonus: float = 0.5             # one op away from target (additive or multiplicative)
    factor_library_max_size: int = 10_000     # LRU cap for cross-episode known factors

    # --- Expert Demos ---
    expert_demo_count: int = 300              # expert demonstrations to pre-fill buffer
    demos_per_advance: int = 150              # fresh demos seeded at each curriculum advance
    allow_partial_demos: bool = True          # if False, fail when demo prefill is too small
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
    eps_end: float = 0.02
    eps_decay_steps: int = 25_000
    eps_advance_floor: float = 0.5            # on curriculum advance, eps bumped to max(current, this)
    target_update_tau: float = 0.005
    train_freq: int = 4
    learning_starts: int = 200

    # --- Buffer hygiene on advance ---
    buffer_keep_recent: int = 5_000           # on-policy transitions retained at advance (demos always kept)

    # --- HER ---
    her_k: int = 4

    # --- Curriculum ---
    curriculum_levels: Tuple[int, ...] = (1, 2, 3, 4, 5, 6)
    curriculum_window: int = 150
    curriculum_train_threshold: float = 0.25
    curriculum_eval_threshold: float = 0.70

    # --- Training ---
    total_steps: int = 500_000
    eval_every: int = 1000
    eval_episodes: int = 100
    seed: int = 42
    log_dir: str = "runs/"
    wandb_artifact_min_interval_steps: int = 50_000

    # --- Mixed sampling ---
    interesting_ratio: float = 0.7  # fraction of interesting polys at high curriculum

    # --- Auto-generation of interesting polynomials ---
    auto_interesting: bool = True               # auto-generate when no JSONL provided
    gen_max_graph_nodes: Optional[int] = 100_000  # safety cap on graph size (None = unlimited)
    gen_max_successors: Optional[int] = 50        # per-node expansion cap (None = unlimited)
    gen_max_seconds: float = 60.0                # wall-clock cap for graph generation

    # --- Diagnostic ---
    oracle_mask: bool = False  # restrict actions to optimal DAG paths (diagnostic only)

    # --- MCTS ---
    use_mcts: bool = True               # use MCTS for action selection
    mcts_simulations: int = 50          # number of MCTS simulations per action
    mcts_c_puct: float = 1.5            # PUCT exploration constant
    mcts_temperature: float = 1.0       # temperature for visit-count action selection
    mcts_warmup_episodes: int = 500     # episodes at a new level before MCTS activates

    def __post_init__(self) -> None:
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        assert self.L >= 1 and self.max_nodes >= 1, "L and max_nodes must be >= 1"
        assert all(
            self.curriculum_levels[i] < self.curriculum_levels[i + 1]
            for i in range(len(self.curriculum_levels) - 1)
        ), "curriculum_levels must be strictly increasing"
        assert self.m >= 1 and self.eval_norm_scale > 0, "m >= 1 and eval_norm_scale > 0 are required"
        assert 0 < self.gamma < 1, "gamma must satisfy 0 < gamma < 1"
        assert 0 <= self.eps_end <= self.eps_start <= 1, "epsilon bounds must satisfy 0 <= eps_end <= eps_start <= 1"
        assert 0.0 <= self.eps_advance_floor <= 1.0, "eps_advance_floor must be in [0, 1]"
        assert self.buffer_keep_recent >= 0, "buffer_keep_recent must be >= 0"
        assert self.demos_per_advance >= 0, "demos_per_advance must be >= 0"
        assert self.mcts_warmup_episodes >= 0, "mcts_warmup_episodes must be >= 0"
        assert self.reward_mode in {"sparse", "shaped", "full"}, "reward_mode must be one of: sparse, shaped, full"
        assert self.factor_library_max_size >= 1, "factor_library_max_size must be >= 1"
        assert self.wandb_artifact_min_interval_steps >= 0, "wandb_artifact_min_interval_steps must be >= 0"
        assert self.gen_max_seconds > 0, "gen_max_seconds must be > 0"
        if self.factor_shaping_coeff > 0:
            try:
                import sympy  # noqa: F401
            except ImportError as exc:
                raise RuntimeError("factor_shaping_coeff>0 requires sympy") from exc

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
