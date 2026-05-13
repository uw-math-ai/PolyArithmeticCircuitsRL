"""Optional RL fine-tuning entry point.

Wires the split-based decomposition environment to the PPO trainer in
``train_ppo``. PPO fine-tuning is positioned as an optional later refinement
on top of the search-distillation stack (see ``AlphaZero RL Circuit Discovery
Plan.md`` § "Training strategy: self-improving search").
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .decomp_env import DecompEnv
from .polynomial import SparsePolynomial


@dataclass(frozen=True)
class OptionalRLConfig:
    algorithm: str = "ppo"
    iterations: int = 100
    rollouts_per_update: int = 8
    candidates_per_step: int = 16
    learning_rate: float = 3e-4
    library_reward_weight: float = 1.0
    seed: int | None = None
    use_mcts: bool = False
    mcts_simulations: int = 32
    mcts_max_depth: int = 4
    mcts_temperature: float = 1.0
    mcts_distill_coef: float = 1.0


def run_optional_rl(
    targets: Sequence[SparsePolynomial],
    env: DecompEnv,
    model,
    config: OptionalRLConfig | None = None,
):
    cfg = config or OptionalRLConfig()
    if cfg.algorithm != "ppo":
        raise NotImplementedError(
            f"Only PPO is wired here; got algorithm={cfg.algorithm!r}"
        )
    from .train_ppo import PPOConfig, train_ppo

    ppo_cfg = PPOConfig(
        rollouts_per_update=cfg.rollouts_per_update,
        candidates_per_step=cfg.candidates_per_step,
        learning_rate=cfg.learning_rate,
        library_reward_weight=cfg.library_reward_weight,
        seed=cfg.seed,
        use_mcts=cfg.use_mcts,
        mcts_simulations=cfg.mcts_simulations,
        mcts_max_depth=cfg.mcts_max_depth,
        mcts_temperature=cfg.mcts_temperature,
        mcts_distill_coef=cfg.mcts_distill_coef,
    )
    return train_ppo(
        targets=targets,
        model=model,
        env=env,
        config=ppo_cfg,
        iterations=cfg.iterations,
    )
