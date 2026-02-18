#!/usr/bin/env python3
"""Train a DQN+HER agent for polynomial circuit construction.

Usage:
    # Basic training with defaults
    python scripts/train.py

    # With interesting polynomial data
    python scripts/train.py --interesting data/game_board_C4.analysis.jsonl

    # Override hyperparameters
    python scripts/train.py --n_vars 2 --max_ops 4 --total_steps 500000
"""

import argparse
import sys

from poly_circuit_rl.config import Config
from poly_circuit_rl.rl.trainer import train


def main():
    parser = argparse.ArgumentParser(
        description="Train DQN+HER agent for polynomial circuit construction",
    )

    # Environment
    parser.add_argument("--n_vars", type=int, default=2, help="Number of variables")
    parser.add_argument("--max_ops", type=int, default=4, help="Max operations budget")
    parser.add_argument("--L", type=int, default=16, help="Max visible nodes")
    parser.add_argument("--m", type=int, default=16, help="Number of eval points")
    parser.add_argument("--step_cost", type=float, default=0.05, help="Per-op penalty")
    parser.add_argument("--shaping_coeff", type=float, default=0.3, help="Eval-distance shaping bonus")

    # Transformer
    parser.add_argument("--d_model", type=int, default=64, help="Transformer hidden dim")
    parser.add_argument("--n_heads", type=int, default=4, help="Attention heads")
    parser.add_argument("--n_layers", type=int, default=3, help="Transformer layers")

    # DQN
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--buffer_size", type=int, default=100_000, help="Replay buffer size")
    parser.add_argument("--eps_decay_steps", type=int, default=50_000, help="Epsilon decay steps")

    # Training
    parser.add_argument("--total_steps", type=int, default=500_000, help="Total env steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--log_dir", type=str, default="runs/", help="Checkpoint directory")

    # Data
    parser.add_argument("--interesting", type=str, default=None,
                        help="Path to analysis JSONL for interesting polynomials")
    parser.add_argument("--no-auto-interesting", action="store_true",
                        help="Disable auto-generation of interesting polynomials")
    parser.add_argument("--gen-max-graph-nodes", type=int, default=None,
                        help="Safety cap on graph nodes during auto-generation")
    parser.add_argument("--gen-max-successors", type=int, default=None,
                        help="Per-node expansion cap during auto-generation")

    args = parser.parse_args()

    config = Config(
        n_vars=args.n_vars,
        max_ops=args.max_ops,
        L=args.L,
        m=args.m,
        step_cost=args.step_cost,
        shaping_coeff=args.shaping_coeff,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        lr=args.lr,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        eps_decay_steps=args.eps_decay_steps,
        total_steps=args.total_steps,
        seed=args.seed,
        log_dir=args.log_dir,
        auto_interesting=not args.no_auto_interesting,
        gen_max_graph_nodes=args.gen_max_graph_nodes,
        gen_max_successors=args.gen_max_successors,
    )

    print(f"Config: n_vars={config.n_vars}, L={config.L}, m={config.m}")
    print(f"  obs_dim={config.obs_dim}, action_dim={config.action_dim}")
    print(f"  d_model={config.d_model}, n_heads={config.n_heads}, n_layers={config.n_layers}")
    print(f"  total_steps={config.total_steps}, seed={config.seed}")

    train(config=config, interesting_jsonl=args.interesting)


if __name__ == "__main__":
    main()
