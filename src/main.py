"""Entry point for training and evaluation."""

import argparse
import random

import numpy as np
import torch

from .config import Config
from .models.policy_value_net import PolicyValueNet
from .algorithms.ppo import PPOTrainer
from .algorithms.alphazero import AlphaZeroTrainer
from .evaluation.evaluate import evaluate_model


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description="Polynomial Arithmetic Circuits RL")
    parser.add_argument("--algorithm", choices=["ppo", "alphazero"], default="ppo",
                        help="Training algorithm")
    parser.add_argument("--iterations", type=int, default=100,
                        help="Number of training iterations")
    parser.add_argument("--eval-only", action="store_true",
                        help="Run evaluation only (requires --checkpoint)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint")
    parser.add_argument("--save-path", type=str, default="checkpoint.pt",
                        help="Path to save model checkpoint")

    # Override config values
    parser.add_argument("--n-variables", type=int, default=None)
    parser.add_argument("--mod", type=int, default=None)
    parser.add_argument("--max-complexity", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-curriculum", action="store_true")

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

    # Train
    print(f"\n=== Training with {args.algorithm.upper()} ===")

    if args.algorithm == "ppo":
        trainer = PPOTrainer(config, model, device=config.device)
    else:
        trainer = AlphaZeroTrainer(config, model, device=config.device)

    trainer.train(args.iterations)

    # Save checkpoint
    torch.save({
        "model": model.state_dict(),
        "config": config,
        "algorithm": args.algorithm,
    }, args.save_path)
    print(f"\nSaved checkpoint to {args.save_path}")

    # Final evaluation
    print("\n=== Final Evaluation ===")
    evaluate_model(model, config, algorithm=args.algorithm, device=config.device)


if __name__ == "__main__":
    main()
