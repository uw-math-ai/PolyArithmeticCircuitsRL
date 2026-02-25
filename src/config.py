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
    success_reward: float = 10.0        # Reward given when the target is matched exactly
    step_penalty: float = -0.1          # Per-step penalty to encourage shorter circuits
    use_reward_shaping: bool = True     # Enable potential-based shaping (Ng et al., 1999)

    # -------------------------------------------------------------------------
    # Factor library and subgoal rewards
    # -------------------------------------------------------------------------
    # When enabled, the target polynomial is factorized at each episode reset
    # (using SymPy over Z). Non-trivial factors become subgoals: the agent is
    # rewarded for building them as intermediate circuit nodes.
    factor_library_enabled: bool = True
    # Bonus reward for constructing any non-trivial factor of the current target.
    # Applied once per factor per episode (cannot be collected twice for the same
    # factor in one episode).
    factor_subgoal_reward: float = 1.0
    # Additional bonus when the constructed factor is already in the library
    # (i.e., was built in a previous successful episode). Stacks on top of
    # factor_subgoal_reward to further incentivise reuse of known sub-circuits.
    factor_library_bonus: float = 0.5

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

    # SAC (discrete, masked)
    sac_actor_lr: float = 1e-4
    sac_critic_lr: float = 3e-4
    sac_alpha_lr: float = 1e-4
    sac_tau: float = 0.01
    sac_batch_size: int = 256
    sac_steps_per_iter: int = 2048
    sac_update_to_data_ratio: float = 1.0
    sac_replay_size: int = 100000
    sac_min_replay_size: int = 10000
    sac_initial_random_steps: int = 2000
    sac_n_step: int = 3

    # State-dependent entropy target: -scale * log(|A_valid(s)|)
    sac_target_entropy_scale: float = 0.98
    sac_alpha_init: float = 0.2
    sac_alpha_min: float = 1e-4
    sac_alpha_max: float = 10.0

    # Replay sampling mix
    sac_current_complexity_fraction: float = 0.5
    sac_success_fraction: float = 0.2
    sac_recent_fraction: float = 0.2
    sac_recent_window: int = 20000

    # Optional stabilizers
    sac_use_cql: bool = False
    sac_cql_alpha: float = 0.0

    # Optional BC warm start from board-derived demonstrations
    sac_bc_warmstart_enabled: bool = False
    sac_bc_samples: int = 5000
    sac_bc_steps: int = 1000
    sac_bc_batch_size: int = 128

    # Optional fixed-complexity warm-up before adaptive curriculum
    sac_fixed_complexities: List[int] = field(default_factory=lambda: [3, 4])
    sac_fixed_complexity_iters: int = 20
    sac_curriculum_window: int = 50

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

    # Curriculum
    curriculum_enabled: bool = True
    starting_complexity: int = 2
    advance_threshold: float = 0.7
    backoff_threshold: float = 0.4

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
