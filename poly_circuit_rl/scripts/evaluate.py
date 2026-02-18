#!/usr/bin/env python3
"""Evaluate a trained DQN+HER agent.

Usage:
    python scripts/evaluate.py --checkpoint runs/best_lvl2.pt --episodes 200
"""

import argparse

from poly_circuit_rl.config import Config
from poly_circuit_rl.env.circuit_env import PolyCircuitEnv
from poly_circuit_rl.rl.agent import DQNAgent
from poly_circuit_rl.rl.trainer import evaluate


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained DQN+HER agent")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint")
    parser.add_argument("--max_ops", type=int, default=4, help="Max operations budget")
    parser.add_argument("--episodes", type=int, default=200, help="Number of eval episodes")
    args = parser.parse_args()

    config = Config(max_ops=args.max_ops)
    env = PolyCircuitEnv(config)
    agent = DQNAgent(config)
    agent.load(args.checkpoint)

    results = evaluate(env, agent, args.max_ops, args.episodes)

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Max ops:    {args.max_ops}")
    print(f"Episodes:   {args.episodes}")
    print(f"Success:    {results['success_rate']:.2%}")
    print(f"Avg reward: {results['avg_reward']:.3f}")
    print(f"Avg steps:  {results['avg_steps']:.1f}")


if __name__ == "__main__":
    main()
