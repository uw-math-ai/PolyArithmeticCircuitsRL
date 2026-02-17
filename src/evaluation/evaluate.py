"""Systematic evaluation harness for trained models."""

from typing import Dict, List, Optional

import torch
import numpy as np

from ..config import Config
from ..models.policy_value_net import PolicyValueNet
from ..environment.circuit_game import CircuitGame
from ..game_board.generator import sample_target, build_game_board
from ..algorithms.mcts import MCTS


def evaluate_model(
    model: PolicyValueNet,
    config: Config,
    algorithm: str = "ppo",
    complexities: Optional[List[int]] = None,
    num_trials: int = 100,
    device: str = "cpu",
    verbose: bool = True,
) -> Dict:
    """Evaluate a trained model across complexity levels.

    Args:
        model: trained PolicyValueNet
        config: configuration
        algorithm: "ppo" (greedy policy) or "alphazero" (MCTS)
        complexities: list of complexity levels to test
        num_trials: number of trials per complexity
        device: torch device
        verbose: print results

    Returns:
        Results dict with per-complexity breakdowns
    """
    if complexities is None:
        complexities = list(range(2, config.max_complexity + 1))

    model.eval()
    env = CircuitGame(config)

    mcts = None
    if algorithm == "alphazero":
        mcts = MCTS(model, config, device)

    results = {}

    for complexity in complexities:
        board = build_game_board(config, complexity)
        successes = 0
        total_steps = 0

        for trial in range(num_trials):
            target_poly, min_steps = sample_target(config, complexity, board)
            obs = env.reset(target_poly)
            success = False

            while not env.done:
                if algorithm == "alphazero" and mcts is not None:
                    action, _ = mcts.get_action_probs(env, temperature=0)
                else:
                    # Greedy policy (PPO evaluation)
                    obs_device = {
                        k: v.to(device) if isinstance(v, torch.Tensor) else v
                        for k, v in obs.items()
                    }
                    with torch.no_grad():
                        logits, _ = model(obs_device)
                    action = logits.argmax(dim=-1).item()

                obs, reward, done, info = env.step(action)

                if info.get("is_success", False):
                    success = True

            if success:
                successes += 1
                total_steps += info["steps_taken"]

        success_rate = successes / num_trials
        avg_steps = total_steps / max(successes, 1)

        results[complexity] = {
            "success_rate": success_rate,
            "avg_steps": avg_steps,
            "num_trials": num_trials,
            "successes": successes,
        }

        if verbose:
            print(
                f"  Complexity {complexity}: "
                f"success={success_rate:.1%} "
                f"({successes}/{num_trials}) "
                f"avg_steps={avg_steps:.1f}"
            )

    # Aggregate
    total_successes = sum(r["successes"] for r in results.values())
    total_trials = sum(r["num_trials"] for r in results.values())
    results["overall"] = {
        "success_rate": total_successes / max(total_trials, 1),
        "total_successes": total_successes,
        "total_trials": total_trials,
    }

    if verbose:
        print(f"  Overall: {results['overall']['success_rate']:.1%} "
              f"({total_successes}/{total_trials})")

    return results
