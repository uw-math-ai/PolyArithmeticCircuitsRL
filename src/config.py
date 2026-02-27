from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class Config:
    """Centralised configuration dataclass for all hyperparameters.

    Groups settings by concern: environment geometry, reward shaping, factor
    library, neural network architecture, and per-algorithm training knobs.

    Derived properties (max_nodes, max_actions, effective_max_degree,
    target_size) are computed from the base fields and should not be set
    directly.
    """

    # -------------------------------------------------------------------------
    # Environment
    # -------------------------------------------------------------------------
    n_variables: int = 2        # Number of polynomial input variables (e.g., 2 → F_p[x0,x1])
    mod: int = 5                # Prime modulus p for F_p arithmetic
    max_complexity: int = 6     # Maximum number of operations allowed per episode
    max_steps: int = 10         # Hard step limit per episode (terminates if reached)
    max_degree: int = -1        # Max degree per variable in dense representation
                                # (-1 = auto: set to max_complexity)

    # -------------------------------------------------------------------------
    # Rewards
    # -------------------------------------------------------------------------
    success_reward: float = 9.0         # Reward when the target is matched (kept your value)
    step_penalty: float = -0.2          # Per-step penalty (kept your value)
    use_reward_shaping: bool = True     # Enable potential-based shaping

    # -------------------------------------------------------------------------
    # Factor library and subgoal rewards (from upstream)
    # -------------------------------------------------------------------------
    factor_library_enabled: bool = True
    factor_subgoal_reward: float = 1.0
    factor_library_bonus: float = 0.5
    completion_bonus: float = 3.0

    # Model
    hidden_dim: int = 128
    embedding_dim: int = 128
    num_gnn_layers: int = 3
    node_feature_dim: int = 4  # [is_input, is_constant, is_op, op_type_value]

    # PPO
    ppo_lr: float = 3e-4
    ppo_clip: float = 0.2
    ppo_epochs: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    batch_size: int = 64
    steps_per_update: int = 2048
    max_grad_norm: float = 0.5

    # AlphaZero / MCTS
    mcts_simulations: int = 100
    mcts_c_puct: float = 1.4
    az_lr: float = 1e-3
    az_games_per_iter: int = 100
    az_training_epochs: int = 10
    az_batch_size: int = 64
    az_buffer_size: int = 50000
    temperature_init: float = 1.0
    temperature_final: float = 0.1
    temperature_decay_steps: int = 30

    # SAC (discrete, masked actions)
    sac_actor_lr: float = 3e-4
    sac_critic_lr: float = 3e-4
    sac_alpha_lr: float = 3e-4
    sac_gamma: float = 0.99
    sac_tau: float = 0.005
    sac_init_alpha: float = 0.2
    sac_auto_entropy_tuning: bool = True
    # Target entropy is ratio * log(valid_action_count)
    sac_target_entropy_ratio: float = 0.7
    sac_batch_size: int = 128
    sac_replay_size: int = 200000
    sac_min_replay_size: int = 5000
    sac_steps_per_update: int = 2048
    sac_updates_per_iter: int = 128
    # Replay sampling: probability of sampling from success bucket when available
    sac_success_sample_ratio: float = 0.55
    # Constructive warm-start
    sac_warmstart_episodes: int = 300
    sac_bc_coef: float = 0.2
    # Optional MCTS policy distillation
    sac_use_mcts_distillation: bool = True
    sac_mcts_distill_prob: float = 0.2
    sac_distill_coef: float = 0.3
    # Periodic checkpointing
    sac_checkpoint_interval: int = 50
    sac_checkpoint_dir: str = "models"
    # Stuck detection and adaptive assistance
    sac_stuck_detection_enabled: bool = True
    sac_stuck_window: int = 12
    sac_stuck_min_iters: int = 12
    sac_stuck_slope_threshold: float = 0.003
    sac_stuck_sr_ceiling: float = 0.55
    sac_stuck_recovery_margin: float = 0.07
    sac_assist_cooldown_iters: int = 12
    sac_assist_distill_prob: float = 0.65
    sac_assist_distill_coef: float = 1.0

    # Curriculum
    curriculum_enabled: bool = True
    starting_complexity: int = 3
    advance_threshold: float = 0.9
    backoff_threshold: float = 0.05

    # Training
    device: str = "cpu"
    seed: int = 42
    log_interval: int = 10

    @property
    def effective_max_degree(self) -> int:
        """Resolved max degree per variable for polynomial representation."""
        return self.max_degree if self.max_degree > 0 else self.max_complexity

    @property
    def max_nodes(self) -> int:
        """Maximum number of nodes: n_variables + 1 (constant) + max_complexity (ops)."""
        return self.n_variables + 1 + self.max_complexity

    @property
    def max_actions(self) -> int:
        """Total action space size: max_nodes * (max_nodes + 1) for (add/mul) x upper-tri pairs."""
        return self.max_nodes * (self.max_nodes + 1)

    @property
    def target_size(self) -> int:
        """Size of the target polynomial coefficient vector.

        Dense rectangular representation: (max_degree+1)^n_variables.
        """
        return (self.effective_max_degree + 1) ** self.n_variables
