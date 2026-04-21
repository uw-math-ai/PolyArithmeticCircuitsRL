#!/usr/bin/env python3
"""Evaluate a trained DQN+HER agent.

Usage:
    python scripts/evaluate.py --checkpoint runs/best_lvl2.pt --episodes 200
"""

import argparse
from dataclasses import replace
from typing import Optional
import warnings

from poly_circuit_rl.config import Config
from poly_circuit_rl.env.circuit_env import PolyCircuitEnv
from poly_circuit_rl.rl.agent import (
    DQNAgent,
    config_from_checkpoint_payload,
    load_checkpoint_payload,
)
from poly_circuit_rl.rl.trainer import evaluate


def config_for_evaluation(
    checkpoint_path: str,
    max_ops_override: Optional[int] = None,
) -> Config:
    """Load evaluation config from checkpoint payload.

    Prefers the exact saved training config to guarantee architecture compatibility.
    Falls back to Config() for legacy checkpoints without a saved config payload.
    """
    payload = load_checkpoint_payload(checkpoint_path, map_location="cpu")
    loaded = config_from_checkpoint_payload(payload)
    if loaded is None:
        warnings.warn(
            "Checkpoint has no saved config; falling back to default Config().",
            stacklevel=2,
        )
        loaded = Config()
    if max_ops_override is not None:
        loaded = replace(loaded, max_ops=max_ops_override)
    return loaded


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained DQN+HER agent")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint")
    parser.add_argument(
        "--max_ops",
        type=int,
        default=None,
        help="Optional max-ops override (defaults to checkpoint config)",
    )
    parser.add_argument("--episodes", type=int, default=200, help="Number of eval episodes")
    args = parser.parse_args()

    config = config_for_evaluation(
        args.checkpoint,
        max_ops_override=args.max_ops,
    )
    env = PolyCircuitEnv(config)
    agent = DQNAgent(config)
    agent.load(args.checkpoint)

    results = evaluate(env, agent, config.max_ops, args.episodes)

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Max ops:    {config.max_ops}")
    print(f"Episodes:   {args.episodes}")
    print(f"Success:    {results['success_rate']:.2%}")
    print(f"Avg reward: {results['avg_reward']:.3f}")
    print(f"Avg steps:  {results['avg_steps']:.1f}")


if __name__ == "__main__":
    main()
