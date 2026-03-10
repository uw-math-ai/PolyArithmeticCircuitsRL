"""Entry point for training and evaluation."""

import argparse
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .config import Config
from .models.policy_value_net import PolicyValueNet
from .algorithms.ppo import PPOTrainer
from .algorithms.ppo_mcts import PPOMCTSTrainer
from .algorithms.alphazero import AlphaZeroTrainer
from .algorithms.sac import SACTrainer
from .evaluation.evaluate import evaluate_model


def save_training_plots(history: dict, results_dir: str) -> None:
    """Save loss, success rate, and reward plots to results_dir.

    Args:
        history: Dict of metric lists from trainer.train().
        results_dir: Directory path where plots are saved.
    """
    iters = range(1, len(history["pg_loss"]) + 1)

    # --- Loss curves ---
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(iters, history["pg_loss"], label="Policy loss")
    ax.plot(iters, history["vf_loss"], label="Value loss")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "loss_curve.png"), dpi=150)
    plt.close(fig)

    # --- Success rate ---
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(iters, history["success_rate"], color="green")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Success Rate")
    ax.set_title("Success Rate per Iteration")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "success_rate.png"), dpi=150)
    plt.close(fig)

    # --- Average reward ---
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(iters, history["avg_reward"], color="orange")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Avg Reward")
    ax.set_title("Average Reward per Iteration")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "reward_per_episode.png"), dpi=150)
    plt.close(fig)

    print(f"Plots saved to {results_dir}")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description="Polynomial Arithmetic Circuits RL")
    parser.add_argument("--algorithm", choices=["ppo", "ppo-mcts", "alphazero", "sac"],
                        default="ppo", help="Training algorithm")
    parser.add_argument("--iterations", type=int, default=100,
                        help="Number of training iterations")
    parser.add_argument("--eval-only", action="store_true",
                        help="Run evaluation only (requires --checkpoint)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint")
    parser.add_argument("--save-path", type=str, default="checkpoint.pt",
                        help="Path to save model checkpoint")
    parser.add_argument("--results-dir", type=str, default=None,
                        help="Directory for results (plots + checkpoint). "
                             "Defaults to results/{algo}_C{complexity}")

    # Override config values
    parser.add_argument("--n-variables", type=int, default=None)
    parser.add_argument("--mod", type=int, default=None)
    parser.add_argument("--max-complexity", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-curriculum", action="store_true")
    parser.add_argument("--mcts-simulations", type=int, default=None,
                        help="Number of MCTS simulations per action (default: 100)")
    parser.add_argument("--steps-per-update", type=int, default=None,
                        help="Environment steps collected per PPO update (default: 2048)")
    parser.add_argument("--sac-use-cql", action="store_true")
    parser.add_argument("--sac-cql-alpha", type=float, default=None)
    parser.add_argument("--sac-bc-warmstart", action="store_true")
    parser.add_argument("--sac-bc-samples", type=int, default=None)
    parser.add_argument("--sac-bc-steps", type=int, default=None)
    parser.add_argument("--sac-fixed-complexity-iters", type=int, default=None)
    parser.add_argument("--sac-target-entropy-scale", type=float, default=None)
    parser.add_argument("--no-factor-library", action="store_true",
                        help="Disable the factor library and subgoal rewards")
    parser.add_argument("--factor-subgoal-reward", type=float, default=None,
                        help="Bonus reward for building a factor of the target polynomial")
    parser.add_argument("--factor-library-bonus", type=float, default=None,
                        help="Additional bonus for building a factor that is already in the library")
    parser.add_argument("--completion-bonus", type=float, default=None,
                        help="Bonus for having both pieces for a single final add/mul to reach T")

    args = parser.parse_args()

    # Build config
    config = Config()
    if args.n_variables is not None:
        config.n_variables = args.n_variables
    if args.mod is not None:
        config.mod = args.mod
    if args.max_complexity is not None:
        config.max_complexity = args.max_complexity
    if args.max_steps is not None:
        config.max_steps = args.max_steps
    if args.hidden_dim is not None:
        config.hidden_dim = args.hidden_dim
        config.embedding_dim = args.hidden_dim
    if args.device is not None:
        config.device = args.device
    if args.seed is not None:
        config.seed = args.seed
    if args.no_curriculum:
        config.curriculum_enabled = False
    if args.sac_use_cql:
        config.sac_use_cql = True
    if args.sac_cql_alpha is not None:
        config.sac_cql_alpha = args.sac_cql_alpha
    if args.sac_bc_warmstart:
        config.sac_bc_warmstart_enabled = True
    if args.sac_bc_samples is not None:
        config.sac_bc_samples = args.sac_bc_samples
    if args.sac_bc_steps is not None:
        config.sac_bc_steps = args.sac_bc_steps
    if args.sac_fixed_complexity_iters is not None:
        config.sac_fixed_complexity_iters = args.sac_fixed_complexity_iters
    if args.sac_target_entropy_scale is not None:
        config.sac_target_entropy_scale = args.sac_target_entropy_scale
    if args.mcts_simulations is not None:
        config.mcts_simulations = args.mcts_simulations
    if args.steps_per_update is not None:
        config.steps_per_update = args.steps_per_update
    if args.no_factor_library:
        config.factor_library_enabled = False
    if args.factor_subgoal_reward is not None:
        config.factor_subgoal_reward = args.factor_subgoal_reward
    if args.factor_library_bonus is not None:
        config.factor_library_bonus = args.factor_library_bonus
    if args.completion_bonus is not None:
        config.completion_bonus = args.completion_bonus

    # Auto-detect device
    if config.device == "cpu" and torch.cuda.is_available():
        config.device = "cuda"
    if config.device == "cpu" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        config.device = "mps"

    set_seed(config.seed)

    print(f"Config: n_vars={config.n_variables}, mod={config.mod}, "
          f"max_complexity={config.max_complexity}, device={config.device}")
    print(f"Max nodes={config.max_nodes}, max_actions={config.max_actions}, "
          f"target_size={config.target_size}")

    if args.algorithm == "sac":
        trainer = SACTrainer(config, device=config.device)
        actor_params = sum(p.numel() for p in trainer.actor.parameters())
        critic_params = sum(p.numel() for p in trainer.critic.parameters())
        print(f"SAC actor parameters: {actor_params:,}")
        print(f"SAC critic parameters: {critic_params:,}")

        if args.checkpoint:
            trainer.load_checkpoint(args.checkpoint)
            print(f"Loaded SAC checkpoint from {args.checkpoint}")

        if args.eval_only:
            print("\n=== Evaluation ===")
            trainer.evaluate(verbose=True, num_trials=100)
            return

        print(f"\n=== Training with {args.algorithm.upper()} ===")
        trainer.train(args.iterations)
        trainer.save_checkpoint(args.save_path)
        print(f"\nSaved checkpoint to {args.save_path}")

        print("\n=== Final Evaluation ===")
        trainer.evaluate(verbose=True, num_trials=100)
    else:
        # Build model
        model = PolicyValueNet(config)
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Load checkpoint if specified
        if args.checkpoint:
            state = torch.load(args.checkpoint, map_location=config.device, weights_only=True)
            model.load_state_dict(state["model"])
            print(f"Loaded checkpoint from {args.checkpoint}")

        if args.eval_only:
            print("\n=== Evaluation ===")
            evaluate_model(model, config, algorithm=args.algorithm, device=config.device)
            return

        # Determine results directory.
        results_dir = args.results_dir or os.path.join(
            "results", f"{args.algorithm}_C{config.max_complexity}"
        )
        os.makedirs(results_dir, exist_ok=True)
        save_path = os.path.join(results_dir, "checkpoint.pt")

        # Train
        algo_label = "PPO+MCTS" if args.algorithm == "ppo-mcts" else args.algorithm.upper()
        print(f"\n=== Training with {algo_label} ===")
        print(f"Results will be saved to {results_dir}")

        if args.algorithm == "ppo":
            trainer = PPOTrainer(config, model, device=config.device)
        elif args.algorithm == "ppo-mcts":
            trainer = PPOMCTSTrainer(config, model, device=config.device)
        else:
            trainer = AlphaZeroTrainer(config, model, device=config.device)

        history = trainer.train(args.iterations)

        # Save plots if history is available.
        if history:
            save_training_plots(history, results_dir)

        # Save checkpoint
        torch.save({
            "model": model.state_dict(),
            "config": config,
            "algorithm": args.algorithm,
        }, save_path)
        print(f"\nSaved checkpoint to {save_path}")

        # Final evaluation
        print("\n=== Final Evaluation ===")
        evaluate_model(model, config, algorithm=args.algorithm, device=config.device)


if __name__ == "__main__":
    main()
