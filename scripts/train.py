#!/usr/bin/env python3
"""Train a DQN+HER agent for polynomial circuit construction.

Usage:
    # Basic training with defaults
    python scripts/train.py

    # With interesting polynomial data
    python scripts/train.py --interesting data/game_board_C4.analysis.jsonl

    # Override hyperparameters
    python scripts/train.py --n_vars 2 --max_ops 6 --total_steps 500000
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from poly_circuit_rl.config import Config
from poly_circuit_rl.rl.trainer import train


def main():
    defaults = Config()
    parser = argparse.ArgumentParser(
        description="Train DQN+HER agent for polynomial circuit construction",
    )

    # Environment
    parser.add_argument("--n_vars", type=int, default=defaults.n_vars, help="Number of variables")
    parser.add_argument("--max_ops", type=int, default=defaults.max_ops, help="Max operations budget")
    parser.add_argument("--L", type=int, default=defaults.L, help="Max visible nodes")
    parser.add_argument("--max_nodes", type=int, default=None, help="Max circuit nodes (must be <= L)")
    parser.add_argument("--m", type=int, default=defaults.m, help="Number of eval points")
    parser.add_argument("--step_cost", type=float, default=defaults.step_cost, help="Per-op penalty")
    parser.add_argument(
        "--reward_mode",
        type=str,
        choices=["sparse", "shaped", "full"],
        default=defaults.reward_mode,
        help="Reward composition mode",
    )
    parser.add_argument(
        "--shaping_coeff",
        type=float,
        default=defaults.shaping_coeff,
        help="Eval-distance shaping bonus",
    )

    # Transformer
    parser.add_argument("--d_model", type=int, default=defaults.d_model, help="Transformer hidden dim")
    parser.add_argument("--n_heads", type=int, default=defaults.n_heads, help="Attention heads")
    parser.add_argument("--n_layers", type=int, default=defaults.n_layers, help="Transformer layers")

    # DQN
    parser.add_argument("--lr", type=float, default=defaults.lr, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=defaults.batch_size, help="Batch size")
    parser.add_argument("--buffer_size", type=int, default=defaults.buffer_size, help="Replay buffer size")
    parser.add_argument(
        "--eps_decay_steps",
        type=int,
        default=defaults.eps_decay_steps,
        help="Epsilon decay steps",
    )

    # Runtime knobs (None = use Config default)
    parser.add_argument("--expert-demo-count", type=int, default=None,
                        help="Expert demos for startup prefill (Config default used if omitted)")
    parser.add_argument("--demos-per-advance", type=int, default=None,
                        help="Fresh demos injected at each curriculum advance")
    parser.add_argument("--allow-partial-demos", dest="allow_partial_demos",
                        action=argparse.BooleanOptionalAction, default=None,
                        help="Accept partial demo prefill (default True in Config)")
    parser.add_argument("--curriculum-train-threshold", type=float, default=None,
                        help="Train SR gate for curriculum advance")
    parser.add_argument("--curriculum-eval-threshold", type=float, default=None,
                        help="Eval SR gate for curriculum advance")
    parser.add_argument("--curriculum-window", type=int, default=None,
                        help="Window size for train-SR running average")
    parser.add_argument("--eps-advance-floor", type=float, default=None,
                        help="Eps rewound to this value at each advance")
    parser.add_argument("--mcts-warmup-episodes", type=int, default=None,
                        help="Episodes at a new level before MCTS activates")
    parser.add_argument("--buffer-keep-recent", type=int, default=None,
                        help="Non-demo transitions kept at advance (demos always kept)")
    parser.add_argument("--eval-episodes", type=int, default=None,
                        help="Episodes per eval checkpoint")

    # Training
    parser.add_argument("--total_steps", type=int, default=defaults.total_steps, help="Total env steps")
    parser.add_argument("--seed", type=int, default=defaults.seed, help="Random seed")
    parser.add_argument("--log_dir", type=str, default=defaults.log_dir, help="Checkpoint directory")

    # Data
    parser.add_argument("--interesting", type=str, default=None,
                        help="Path to analysis JSONL for interesting polynomials")
    parser.add_argument("--eval_jsonl", type=str, default=None,
                        help="Path to held-out eval JSONL (from scripts/build_dataset.py)")
    parser.add_argument("--no-auto-interesting", action="store_true",
                        help="Disable auto-generation of interesting polynomials")
    parser.add_argument("--gen-max-graph-nodes", type=int, default=None,
                        help="Safety cap on graph nodes during auto-generation")
    parser.add_argument("--gen-max-successors", type=int, default=None,
                        help="Per-node expansion cap during auto-generation")
    parser.add_argument("--gen-max-seconds", type=float, default=defaults.gen_max_seconds,
                        help="Wall-clock cap for auto-generation graph enumeration")
    parser.add_argument("--wandb", action="store_true",
                        help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="PolyArithmeticCircuitsRL",
                        help="W&B project name (used when --wandb is set)")
    parser.add_argument("--wandb-entity", type=str, default=None,
                        help="W&B entity/team (optional)")
    parser.add_argument("--wandb-run-name", type=str, default=None,
                        help="W&B run name (optional)")

    args = parser.parse_args()

    config_kwargs = dict(
        n_vars=args.n_vars,
        max_ops=args.max_ops,
        L=args.L,
        max_nodes=args.max_nodes if args.max_nodes is not None else args.L,
        m=args.m,
        step_cost=args.step_cost,
        reward_mode=args.reward_mode,
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
        gen_max_seconds=args.gen_max_seconds,
    )

    # Runtime knobs: only override Config defaults when user explicitly passed the flag
    optional_overrides = {
        "expert_demo_count":          args.expert_demo_count,
        "demos_per_advance":          args.demos_per_advance,
        "allow_partial_demos":        args.allow_partial_demos,
        "curriculum_train_threshold": args.curriculum_train_threshold,
        "curriculum_eval_threshold":  args.curriculum_eval_threshold,
        "curriculum_window":          args.curriculum_window,
        "eps_advance_floor":          args.eps_advance_floor,
        "mcts_warmup_episodes":       args.mcts_warmup_episodes,
        "buffer_keep_recent":         args.buffer_keep_recent,
        "eval_episodes":              args.eval_episodes,
    }
    for k, v in optional_overrides.items():
        if v is not None:
            config_kwargs[k] = v

    config = Config(**config_kwargs)

    print(f"Config: n_vars={config.n_vars}, L={config.L}, m={config.m}")
    print(f"  obs_dim={config.obs_dim}, action_dim={config.action_dim}")
    print(f"  d_model={config.d_model}, n_heads={config.n_heads}, n_layers={config.n_layers}")
    print(f"  total_steps={config.total_steps}, seed={config.seed}")
    if args.wandb:
        print(f"  wandb_project={args.wandb_project}, wandb_entity={args.wandb_entity}, "
              f"wandb_run_name={args.wandb_run_name}")

    train(
        config=config,
        interesting_jsonl=args.interesting,
        eval_jsonl=args.eval_jsonl,
        wandb_project=args.wandb_project if args.wandb else None,
        wandb_entity=args.wandb_entity if args.wandb else None,
        wandb_run_name=args.wandb_run_name if args.wandb else None,
    )


if __name__ == "__main__":
    main()
