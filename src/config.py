from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    """Centralised configuration dataclass for all hyperparameters.

    Groups settings by concern: environment geometry, rewards, factor library,
    neural network architecture, PPO updates, MCTS, and logging.

    Derived properties (max_nodes, max_actions, effective_max_degree,
    target_size) are computed from the base fields and should not be set
    directly.
    """

    # -------------------------------------------------------------------------
    # Environment
    # -------------------------------------------------------------------------
    n_variables: int = 2        # Number of polynomial input variables (e.g., 2 → F_p[x0,x1])
    mod: int = 5                # Prime modulus p for F_p arithmetic
    max_complexity: int = 6     # Maximum target/cache complexity used for training
    max_build_complexity: int = -1  # Operation-node budget (-1 = max_complexity)
    build_complexity_slack: int = -1  # Per-target extra op budget (-1 = disabled)
    max_steps: int = 10         # Hard step limit per episode (terminates if reached)
    max_degree: int = -1        # Max degree per variable in dense representation
                                # (-1 = auto: set to max_complexity)

    # -------------------------------------------------------------------------
    # Rewards
    # -------------------------------------------------------------------------
    # Reward modes:
    #   legacy: current term/factor/completion shaping baseline.
    #   clean_sparse: terminal_success_reward + step_penalty only.
    #   clean_onpath: clean_sparse + cached sequential-route on-path potential shaping.
    reward_mode: str = "clean_onpath"
    success_reward: float = 10.0                # Reward given when the target is matched exactly
    terminal_success_reward: float = 10.0       # Clean-mode terminal reward.
    step_penalty: float = -0.1                  # Per-step penalty to encourage shorter circuits
    use_reward_shaping: bool = True             # Enable potential-based shaping (Ng et al., 1999)
    graph_onpath_shaping_coeff: float = 3.0
    graph_onpath_cache_dir: Optional[str] = None
    on_path_terminal_zero: bool = True
    on_path_phi_mode: str = "depth_weighted"    # "count", "max_step", or "depth_weighted"
    on_path_depth_weight_power: float = 1.0     #1,2,3,4(1) ->1,4,9,16(2)
    on_path_max_size: int = 4096
    on_path_split_seed: int = 42
    on_path_route_consistency: bool = True # prevents reward from mixing unrelated pieces from incompatible optimal routes.
    on_path_route_consistency_mode: str = "best_route_phi"
    # best_route_phi: compute progress for each coherent route and use the best one. This is the current preferred mode.
    # lock_on_first_hit: once the agent hits a route, only compatible route nodes count afterward. Stricter and more brittle.
    # off: union all on-path nodes, ignoring route compatibility.
    on_path_num_routes: int = 32
    # Non-PBRS additive bonus paid each step phi increases (coherent-route progress).
    # Bounded: phi <= 1 caps total per-episode bonus at this coefficient.
    # 0.0 disables. Recommended: 0.5 for fixed-C2 diagnostic.
    on_route_bonus_coeff: float = 3.0

    # -------------------------------------------------------------------------
    # Factor library and subgoal rewards
    # -------------------------------------------------------------------------
    # When enabled, the target polynomial is factorized at each episode reset
    # (using SymPy over Z). Non-trivial factors become subgoals: the agent is
    # rewarded for building them as intermediate circuit nodes.
    factor_library_enabled: bool = False
    # Bonus reward for constructing any non-trivial factor of the current target.
    # Applied once per factor per episode (cannot be collected twice for the same
    # factor in one episode).
    factor_subgoal_reward: float = 1.0
    # Additional bonus when the constructed factor is already in the library
    # (i.e., was built in a previous successful episode). Stacks on top of
    # factor_subgoal_reward to further incentivise reuse of known sub-circuits.
    factor_library_bonus: float = 0.5
    # Completion bonus fired when the agent creates a node v such that the
    # circuit now contains both v and its "complement" for one final operation:
    #   Additive:       T - v  is already in the circuit  (one add away from T)
    #   Multiplicative: T / v  is already in the circuit (one mul away from T)
    # Fires at most once per direction (additive / multiplicative) per episode,
    # so the agent cannot farm it by repeatedly building the same node.
    completion_bonus: float = 3.0

    # Model
    hidden_dim: int = 256
    embedding_dim: int = 256
    num_gnn_layers: int = 4
    node_feature_dim: int = 4  # [is_input, is_constant, is_op, op_type_value]

    # PPO
    ppo_lr: float = 3e-4
    ppo_clip: float = 0.2
    ppo_log_ratio_clip: float = 10.0
    ppo_epochs: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ent_coef: float = 0.01
    ent_coef_final: float = 0.01
    ent_coef_anneal_fraction: float = 0.5
    vf_coef: float = 0.1
    value_clip_enabled: bool = True
    value_clip_range: float = 0.2
    adv_normalize_per_minibatch: bool = True
    batch_size: int = 256
    steps_per_update: int = 4096
    max_grad_norm: float = 0.5
    nonfinite_update_limit: int = 3
    target_kl: float = 0.0

    # MCTS
    mcts_simulations: int = 100
    mcts_c_puct: float = 1.4
    temperature_init: float = 1.0
    temperature_final: float = 0.1
    temperature_decay_steps: int = 30

    # Curriculum
    curriculum_enabled: bool = True
    starting_complexity: int = 2
    advance_threshold: float = 0.7
    backoff_threshold: float = 0.4
    curriculum_window: int = 50
    curriculum_min_dwell_iterations: int = 1
    curriculum_backoff_patience_iterations: int = 0

    # Training
    device: str = "cpu"
    seed: int = 42
    log_interval: int = 10
    disable_progress_bar: bool = False

    # Weights & Biases logging
    wandb_enabled: bool = False
    wandb_project: str = "PolyArithmeticCircuitsRL"
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None

    @property
    def effective_max_degree(self) -> int:
        """Resolved max degree per variable for polynomial representation."""
        return self.max_degree if self.max_degree > 0 else self.max_complexity

    @property
    def effective_max_build_complexity(self) -> int:
        """Resolved operation-node budget for episode state/action arrays."""
        return (
            self.max_build_complexity
            if self.max_build_complexity > 0
            else self.max_complexity
        )

    @property
    def max_nodes(self) -> int:
        """Maximum nodes: inputs + constant + operation-node budget."""
        return self.n_variables + 1 + self.effective_max_build_complexity

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
