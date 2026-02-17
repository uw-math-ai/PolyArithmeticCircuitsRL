from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class Config:
    # Environment
    n_variables: int = 2
    mod: int = 5
    max_complexity: int = 6  # max operations allowed
    max_steps: int = 10  # max episode steps
    max_degree: int = -1  # max degree per variable (-1 = auto: max_complexity)

    # Rewards
    success_reward: float = 10.0
    step_penalty: float = -0.1
    use_reward_shaping: bool = True

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
