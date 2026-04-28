"""Entry point for training and evaluation."""

import argparse
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import Config


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
    ax.plot(iters, history["success_rate"], color="green", label="Overall")
    # Per-complexity curves if present.
    per_c_keys = sorted(k for k in history if k.startswith("success_rate_C"))
    for key in per_c_keys:
        label = key.replace("success_rate_", "")
        ax.plot(iters, history[key], label=label, alpha=0.7)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Success Rate")
    ax.set_title("Success Rate per Iteration")
    ax.set_ylim(-0.05, 1.05)
    if per_c_keys:
        ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "success_rate.png"), dpi=150)
    plt.close(fig)

    # --- Average reward ---
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(iters, history["avg_reward"], color="orange", label="Overall")
    per_c_reward_keys = sorted(k for k in history if k.startswith("avg_reward_C"))
    for key in per_c_reward_keys:
        label = key.replace("avg_reward_", "")
        ax.plot(iters, history[key], label=label, alpha=0.7)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Avg Reward")
    ax.set_title("Average Reward per Iteration")
    if per_c_reward_keys:
        ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "reward_per_episode.png"), dpi=150)
    plt.close(fig)

    print(f"Plots saved to {results_dir}")


def set_seed(seed: int, use_torch: bool = True):
    """Set random seeds for reproducibility.

    Args:
        seed: Random seed.
        use_torch: Whether to also seed PyTorch (skipped for JAX-only runs).
    """
    random.seed(seed)
    np.random.seed(seed)
    if use_torch:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description="Polynomial Arithmetic Circuits RL")
    parser.add_argument("--algorithm", choices=["ppo", "ppo-mcts", "ppo-mcts-jax", "alphazero", "sac"],
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
    parser.add_argument("--max-degree", type=int, default=None,
                        help="Max degree per variable (default: auto = max_complexity)")
    parser.add_argument("--ent-coef", type=float, default=None,
                        help="Entropy coefficient for PPO (default: 0.01)")
    parser.add_argument("--ppo-lr", type=float, default=None,
                        help="Learning rate for the PPO/MCTS JAX optimizer")
    parser.add_argument("--ppo-epochs", type=int, default=None,
                        help="Number of policy/value epochs per rollout")
    parser.add_argument("--mcts-simulations", type=int, default=None,
                        help="Number of MCTS simulations per action (default: 100)")
    parser.add_argument("--search", choices=["puct", "gumbel"], default=None,
                        help="Search backend for ppo-mcts / alphazero (default: puct)")
    parser.add_argument("--gumbel-num-simulations", type=int, default=None,
                        help="Number of Gumbel root search simulations (default: 32)")
    parser.add_argument("--gumbel-max-num-considered-actions", type=int, default=None,
                        help="Maximum number of root actions retained after Gumbel-Top-k (default: 16)")
    parser.add_argument("--gumbel-scale", type=float, default=None,
                        help="Scale applied to sampled Gumbel root noise (default: 1.0)")
    parser.add_argument("--gumbel-c-visit", type=float, default=None,
                        help="Completed-Q visit scaling constant for Gumbel search (default: 50.0)")
    parser.add_argument("--gumbel-c-scale", type=float, default=None,
                        help="Completed-Q scale multiplier for Gumbel search (default: 0.1)")
    parser.add_argument("--gumbel-distill-coef", type=float, default=None,
                        help="Auxiliary distillation loss weight for PPO+MCTS Gumbel training (default: 0.5)")
    parser.add_argument("--no-gumbel-q-normalize", action="store_true",
                        help="Disable min-max normalization before the Gumbel completed-Q transform")
    parser.add_argument("--gumbel-root-only", action="store_true",
                        help="Use the root-only hand-written Gumbel search implementation")
    parser.add_argument("--steps-per-update", type=int, default=None,
                        help="Environment steps collected per PPO update (default: 2048)")
    parser.add_argument("--mcts-batch-size", type=int, default=256,
                        help="Number of parallel environments for ppo-mcts-jax (default: 256)")
    parser.add_argument("--fixed-complexities", type=int, nargs="+", default=None,
                        help="Train on these complexity levels in parallel (e.g. 5 6 7 8). "
                             "Disables curriculum. max-complexity is auto-set to the maximum.")
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

    # Weights & Biases
    parser.add_argument("--wandb", action="store_true",
                        help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default=None,
                        help="W&B project name (default: PolyArithmeticCircuitsRL)")
    parser.add_argument("--wandb-entity", type=str, default=None,
                        help="W&B entity (team or user)")
    parser.add_argument("--wandb-run-name", type=str, default=None,
                        help="W&B run name (default: auto-generated)")

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
    if args.max_degree is not None:
        config.max_degree = args.max_degree
    if args.ent_coef is not None:
        config.ent_coef = args.ent_coef
    if args.ppo_lr is not None:
        config.ppo_lr = args.ppo_lr
    if args.ppo_epochs is not None:
        config.ppo_epochs = args.ppo_epochs
    if args.mcts_simulations is not None:
        config.mcts_simulations = args.mcts_simulations
    if args.search is not None:
        config.search = args.search
    if args.gumbel_num_simulations is not None:
        config.gumbel_num_simulations = args.gumbel_num_simulations
    if args.gumbel_max_num_considered_actions is not None:
        config.gumbel_max_num_considered_actions = args.gumbel_max_num_considered_actions
    if args.gumbel_scale is not None:
        config.gumbel_scale = args.gumbel_scale
    if args.gumbel_c_visit is not None:
        config.gumbel_c_visit = args.gumbel_c_visit
    if args.gumbel_c_scale is not None:
        config.gumbel_c_scale = args.gumbel_c_scale
    if args.gumbel_distill_coef is not None:
        config.gumbel_distill_coef = args.gumbel_distill_coef
    if args.no_gumbel_q_normalize:
        config.gumbel_q_normalize = False
    if args.gumbel_root_only:
        config.gumbel_root_only = True
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
    if args.wandb:
        config.wandb_enabled = True
    if args.wandb_project is not None:
        config.wandb_project = args.wandb_project
    if args.wandb_entity is not None:
        config.wandb_entity = args.wandb_entity
    if args.wandb_run_name is not None:
        config.wandb_run_name = args.wandb_run_name

    # Auto-detect device.
    is_jax = args.algorithm == "ppo-mcts-jax"
    if is_jax:
        import jax
        jax_devices = jax.devices()
        jax_backend = jax_devices[0].platform if jax_devices else "cpu"
        config.device = f"jax:{jax_backend}"
        print(f"JAX backend: {jax_backend} | devices: {jax_devices}")
        if jax_backend == "cpu":
            print("WARNING: JAX is running on CPU. For GPU, ensure jax[cuda12] "
                  "is installed and --nv is passed to apptainer.")
    else:
        import torch
        if config.device == "cpu" and torch.cuda.is_available():
            config.device = "cuda"
        if config.device == "cpu" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            config.device = "mps"

    set_seed(config.seed, use_torch=not is_jax)

    # Initialise Weights & Biases if enabled.
    if config.wandb_enabled:
        import wandb
        tags = [args.algorithm]
        if args.fixed_complexities:
            tags.append(f"C{'_'.join(str(c) for c in args.fixed_complexities)}")
        wandb_config = {
            k: v for k, v in vars(config).items()
            if not k.startswith("_")
        }
        if args.fixed_complexities:
            wandb_config["fixed_complexities"] = args.fixed_complexities
        wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity,
            name=config.wandb_run_name,
            config=wandb_config,
            tags=tags,
        )

    print(f"Config: n_vars={config.n_variables}, mod={config.mod}, "
          f"max_complexity={config.max_complexity}, device={config.device}")
    print(f"Max nodes={config.max_nodes}, max_actions={config.max_actions}, "
          f"target_size={config.target_size}")

    if args.algorithm == "sac":
        from .algorithms.sac import SACTrainer
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

        results_dir = args.results_dir or os.path.join(
            "results", f"sac_C{config.max_complexity}"
        )
        os.makedirs(results_dir, exist_ok=True)
        save_path = os.path.join(results_dir, "checkpoint.pt")

        print(f"\n=== Training with {args.algorithm.upper()} ===")
        print(f"Results will be saved to {results_dir}")
        history = trainer.train(args.iterations)
        trainer.save_checkpoint(save_path)
        print(f"\nSaved checkpoint to {save_path}")

        if history:
            save_training_plots(history, results_dir)

        print("\n=== Final Evaluation ===")
        trainer.evaluate(verbose=True, num_trials=100)

        if config.wandb_enabled:
            import wandb
            wandb.finish()
    elif args.algorithm == "ppo-mcts-jax":
        from .algorithms.ppo_mcts_jax import PPOMCTSJAXTrainer

        # When fixed-complexities is set, ensure max_complexity accommodates all.
        fixed_c = args.fixed_complexities
        if fixed_c:
            needed = max(fixed_c)
            if config.max_complexity < needed:
                config.max_complexity = needed
            # Also scale max_steps proportionally if needed.
            if config.max_steps < needed + 4:
                config.max_steps = needed + 4

        trainer = PPOMCTSJAXTrainer(
            config,
            batch_size=args.mcts_batch_size,
            fixed_complexities=fixed_c,
        )
        fc_str = f" fixed_complexities={fixed_c}" if fixed_c else ""
        print(f"JAX PPO+MCTS with batch_size={args.mcts_batch_size}{fc_str}")

        if args.checkpoint:
            trainer.load_checkpoint(args.checkpoint)
            print(f"Loaded PPO+MCTS JAX checkpoint from {args.checkpoint}")

        if args.eval_only:
            print("Eval-only mode not supported for ppo-mcts-jax yet.")
            return

        if fixed_c:
            label = f"C{'_'.join(str(c) for c in fixed_c)}"
        else:
            label = f"C{config.max_complexity}"
        results_dir = args.results_dir or os.path.join(
            "results", f"ppo-mcts-jax_{label}"
        )
        os.makedirs(results_dir, exist_ok=True)

        print(f"\n=== Training with PPO+MCTS (JAX) ===")
        print(f"Results will be saved to {results_dir}")
        history = trainer.train(args.iterations, results_dir=results_dir)

        if history:
            save_training_plots(history, results_dir)

        if config.wandb_enabled:
            import wandb
            wandb.finish()
    else:
        from .models.policy_value_net import PolicyValueNet
        from .algorithms.ppo import PPOTrainer
        from .algorithms.ppo_mcts import PPOMCTSTrainer
        from .algorithms.alphazero import AlphaZeroTrainer
        from .evaluation.evaluate import evaluate_model

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
            log_path = os.path.join(results_dir, "log.txt")
            trainer = PPOMCTSTrainer(
                config, model, device=config.device, log_path=log_path,
            )
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

        if config.wandb_enabled:
            import wandb
            wandb.finish()


if __name__ == "__main__":
    main()
