"""Entry point for PPO+MCTS training and evaluation."""

import argparse
import os
import random

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import Config


def save_training_plots(history: dict, results_dir: str) -> None:
    """Save loss, success rate, and reward plots to results_dir."""
    iters = range(1, len(history["pg_loss"]) + 1)

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

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(iters, history["success_rate"], color="green", label="Overall")
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


def set_seed(seed: int, use_torch: bool = True) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    if use_torch:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Polynomial Arithmetic Circuits PPO+MCTS")
    parser.add_argument(
        "--algorithm",
        choices=["ppo-mcts", "ppo-mcts-jax"],
        default="ppo-mcts",
        help="Training algorithm",
    )
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Run PyTorch PPO+MCTS evaluation only (requires --checkpoint)",
    )
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Directory for plots and checkpoints. Defaults to results/{algo}_C{complexity}",
    )

    parser.add_argument("--n-variables", type=int, default=None)
    parser.add_argument("--mod", type=int, default=None)
    parser.add_argument("--max-complexity", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-curriculum", action="store_true")
    parser.add_argument("--starting-complexity", type=int, default=None)
    parser.add_argument("--advance-threshold", type=float, default=None)
    parser.add_argument("--backoff-threshold", type=float, default=None)
    parser.add_argument("--curriculum-window", type=int, default=None)
    parser.add_argument(
        "--curriculum-min-dwell-iterations",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--max-degree",
        type=int,
        default=None,
        help="Max degree per variable (default: auto = max_complexity)",
    )
    parser.add_argument("--ent-coef", type=float, default=None)
    parser.add_argument(
        "--ppo-epochs",
        type=int,
        default=None,
        help="PPO optimization epochs per rollout batch",
    )
    parser.add_argument("--mcts-simulations", type=int, default=None)
    parser.add_argument("--steps-per-update", type=int, default=None)
    parser.add_argument(
        "--reward-mode",
        choices=["legacy", "clean_sparse", "clean_onpath"],
        default=None,
        help="Reward mode for ablations.",
    )
    parser.add_argument("--terminal-success-reward", type=float, default=None)
    parser.add_argument("--graph-onpath-cache-dir", type=str, default=None)
    parser.add_argument("--graph-onpath-shaping-coeff", type=float, default=None)
    parser.add_argument(
        "--on-path-phi-mode",
        choices=["count", "max_step"],
        default=None,
    )
    parser.add_argument("--on-path-max-size", type=int, default=None)
    parser.add_argument("--on-path-split-seed", type=int, default=None)
    parser.add_argument("--on-path-num-routes", type=int, default=None)
    parser.add_argument(
        "--no-on-path-route-consistency",
        action="store_true",
        help="Disable coherent-route masking for clean_onpath rewards.",
    )
    parser.add_argument(
        "--mcts-batch-size",
        type=int,
        default=256,
        help="Number of parallel environments for ppo-mcts-jax",
    )
    parser.add_argument(
        "--fixed-complexities",
        type=int,
        nargs="+",
        default=None,
        help="Train JAX PPO+MCTS on these complexity levels in parallel.",
    )
    parser.add_argument("--no-factor-library", action="store_true")
    parser.add_argument("--factor-subgoal-reward", type=float, default=None)
    parser.add_argument("--factor-library-bonus", type=float, default=None)
    parser.add_argument("--completion-bonus", type=float, default=None)

    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)

    args = parser.parse_args()

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
    if args.starting_complexity is not None:
        config.starting_complexity = args.starting_complexity
    if args.advance_threshold is not None:
        config.advance_threshold = args.advance_threshold
    if args.backoff_threshold is not None:
        config.backoff_threshold = args.backoff_threshold
    if args.curriculum_window is not None:
        config.curriculum_window = args.curriculum_window
    if args.curriculum_min_dwell_iterations is not None:
        config.curriculum_min_dwell_iterations = (
            args.curriculum_min_dwell_iterations
        )
    if args.max_degree is not None:
        config.max_degree = args.max_degree
    if args.ent_coef is not None:
        config.ent_coef = args.ent_coef
    if args.ppo_epochs is not None:
        config.ppo_epochs = args.ppo_epochs
    if args.mcts_simulations is not None:
        config.mcts_simulations = args.mcts_simulations
    if args.steps_per_update is not None:
        config.steps_per_update = args.steps_per_update
    if args.reward_mode is not None:
        config.reward_mode = args.reward_mode
    if args.terminal_success_reward is not None:
        config.terminal_success_reward = args.terminal_success_reward
    if args.graph_onpath_cache_dir is not None:
        config.graph_onpath_cache_dir = args.graph_onpath_cache_dir
    if args.graph_onpath_shaping_coeff is not None:
        config.graph_onpath_shaping_coeff = args.graph_onpath_shaping_coeff
    if args.on_path_phi_mode is not None:
        config.on_path_phi_mode = args.on_path_phi_mode
    if args.on_path_max_size is not None:
        config.on_path_max_size = args.on_path_max_size
    if args.on_path_split_seed is not None:
        config.on_path_split_seed = args.on_path_split_seed
    if args.on_path_num_routes is not None:
        config.on_path_num_routes = args.on_path_num_routes
    if args.no_on_path_route_consistency:
        config.on_path_route_consistency = False
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

    is_jax = args.algorithm == "ppo-mcts-jax"
    if is_jax:
        import jax

        jax_devices = jax.devices()
        jax_backend = jax_devices[0].platform if jax_devices else "cpu"
        config.device = f"jax:{jax_backend}"
        print(f"JAX backend: {jax_backend} | devices: {jax_devices}")
        if jax_backend == "cpu":
            print(
                "WARNING: JAX is running on CPU. For GPU, ensure jax[cuda12] "
                "is installed and --nv is passed to apptainer."
            )
    else:
        import torch

        if config.device == "cpu" and torch.cuda.is_available():
            config.device = "cuda"
        if (
            config.device == "cpu"
            and hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            config.device = "mps"

    set_seed(config.seed, use_torch=not is_jax)

    if config.wandb_enabled:
        import wandb

        tags = [args.algorithm]
        if args.fixed_complexities:
            tags.append(f"C{'_'.join(str(c) for c in args.fixed_complexities)}")
        wandb_config = {
            k: v for k, v in vars(config).items() if not k.startswith("_")
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

    print(
        f"Config: n_vars={config.n_variables}, mod={config.mod}, "
        f"max_complexity={config.max_complexity}, device={config.device}, "
        f"reward_mode={config.reward_mode}"
    )
    print(
        f"Max nodes={config.max_nodes}, max_actions={config.max_actions}, "
        f"target_size={config.target_size}"
    )

    if args.algorithm == "ppo-mcts-jax":
        from .algorithms.ppo_mcts_jax import PPOMCTSJAXTrainer

        if args.eval_only:
            print("Use eval_ppo_mcts_jax.py for JAX PPO+MCTS checkpoint evaluation.")
            return

        fixed_c = args.fixed_complexities
        if fixed_c:
            needed = max(fixed_c)
            if config.max_complexity < needed:
                config.max_complexity = needed
            if config.max_steps < needed + 4:
                config.max_steps = needed + 4
            config.curriculum_enabled = False

        trainer = PPOMCTSJAXTrainer(
            config,
            batch_size=args.mcts_batch_size,
            fixed_complexities=fixed_c,
        )
        fc_str = f" fixed_complexities={fixed_c}" if fixed_c else ""
        print(f"JAX PPO+MCTS with batch_size={args.mcts_batch_size}{fc_str}")

        label = f"C{'_'.join(str(c) for c in fixed_c)}" if fixed_c else f"C{config.max_complexity}"
        results_dir = args.results_dir or os.path.join("results", f"ppo-mcts-jax_{label}")
        os.makedirs(results_dir, exist_ok=True)

        print("\n=== Training with PPO+MCTS (JAX) ===")
        print(f"Results will be saved to {results_dir}")
        history = trainer.train(args.iterations, results_dir=results_dir)

        if history:
            save_training_plots(history, results_dir)
    else:
        import torch

        from .algorithms.ppo_mcts import PPOMCTSTrainer
        from .evaluation.evaluate import evaluate_model
        from .models.policy_value_net import PolicyValueNet

        model = PolicyValueNet(config)
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        if args.checkpoint:
            state = torch.load(args.checkpoint, map_location=config.device, weights_only=False)
            model.load_state_dict(state["model"] if isinstance(state, dict) and "model" in state else state)
            print(f"Loaded checkpoint from {args.checkpoint}")

        if args.eval_only:
            print("\n=== Evaluation ===")
            if config.reward_mode == "clean_onpath":
                print("Evaluation is oracle-free: using reward_mode=clean_sparse.")
                config.reward_mode = "clean_sparse"
                config.graph_onpath_cache_dir = None
            evaluate_model(model, config, algorithm=args.algorithm, device=config.device)
            return

        results_dir = args.results_dir or os.path.join(
            "results", f"ppo-mcts_C{config.max_complexity}"
        )
        os.makedirs(results_dir, exist_ok=True)

        print("\n=== Training with PPO+MCTS ===")
        print(f"Results will be saved to {results_dir}")
        trainer = PPOMCTSTrainer(
            config,
            model,
            device=config.device,
            log_path=os.path.join(results_dir, "log.txt"),
        )
        history = trainer.train(args.iterations)

        if history:
            save_training_plots(history, results_dir)

        save_path = os.path.join(results_dir, "checkpoint.pt")
        torch.save(
            {
                "model": model.state_dict(),
                "config": config,
                "algorithm": args.algorithm,
            },
            save_path,
        )
        print(f"\nSaved checkpoint to {save_path}")

    if config.wandb_enabled:
        import wandb

        wandb.finish()


if __name__ == "__main__":
    main()
