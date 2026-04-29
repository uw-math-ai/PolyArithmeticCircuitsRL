#!/usr/bin/env python3
"""Greedy 1-step lookahead baseline.

Pure deterministic: at each step, enumerate all valid (op, i, j) actions,
simulate the resulting polynomial, and pick the one with highest term
similarity to the target. If any action's result equals the target, take it
immediately.

No randomness, no learning, no tree search.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Config  # noqa: E402
from src.environment.fast_polynomial import FastPoly  # noqa: E402


def _initial_nodes(config: Config) -> List[FastPoly]:
    n_vars = config.n_variables
    max_deg = config.effective_max_degree
    mod = config.mod
    nodes: List[FastPoly] = []
    for i in range(n_vars):
        nodes.append(FastPoly.variable(i, n_vars, max_deg, mod))
    nodes.append(FastPoly.constant(1, n_vars, max_deg, mod))
    return nodes


def _enumerate_valid_actions(num_nodes: int) -> List[Tuple[int, int, int]]:
    """Return list of (op, i, j) for op in {0=add, 1=mul}, 0 <= i <= j < num_nodes."""
    actions: List[Tuple[int, int, int]] = []
    for op in (0, 1):
        for i in range(num_nodes):
            for j in range(i, num_nodes):
                actions.append((op, i, j))
    return actions


def solve_greedy(
    target_coeffs_flat: np.ndarray,
    config: Config,
) -> Dict[str, float]:
    """Run greedy 1-step lookahead on a single target.

    Returns:
        Dict with keys:
          - success (bool)
          - num_steps (int): steps actually taken (success or max_steps)
          - actions (List[(op,i,j)]): the action sequence
          - env_reward (float): step_penalty * num_steps + terminal reward if success
                                + legacy term-similarity shaping when enabled.
    """
    shape = (config.effective_max_degree + 1,) * config.n_variables
    target_poly = FastPoly(
        np.asarray(target_coeffs_flat, dtype=np.int64).reshape(shape),
        config.mod,
    )

    nodes = _initial_nodes(config)
    actions: List[Tuple[int, int, int]] = []
    env_reward = 0.0
    sim_prev = max(node.term_similarity(target_poly) for node in nodes)
    success = False

    for step in range(config.max_steps):
        if len(nodes) >= config.max_nodes:
            break

        # 1) Look for an immediate success.
        chosen: Tuple[int, int, int] = None
        chosen_poly: FastPoly = None
        chosen_sim: float = -1.0

        for op, i, j in _enumerate_valid_actions(len(nodes)):
            new_poly = nodes[i] + nodes[j] if op == 0 else nodes[i] * nodes[j]
            if new_poly == target_poly:
                chosen = (op, i, j)
                chosen_poly = new_poly
                chosen_sim = 1.0
                break
            sim = new_poly.term_similarity(target_poly)
            if sim > chosen_sim:
                chosen = (op, i, j)
                chosen_poly = new_poly
                chosen_sim = sim

        if chosen is None:
            break

        # Apply.
        actions.append(chosen)
        nodes.append(chosen_poly)
        is_success = chosen_poly == target_poly

        # Plain clean modes are sparse; legacy mode keeps PBRS for comparison
        # with older branches.
        sim_now = max(node.term_similarity(target_poly) for node in nodes)
        if (
            config.reward_mode == "legacy"
            and config.use_reward_shaping
            and not is_success
        ):
            shaping = config.gamma * sim_now - sim_prev
            env_reward += config.step_penalty + shaping
        else:
            env_reward += config.step_penalty
            if is_success:
                env_reward += (
                    config.success_reward
                    if config.reward_mode == "legacy"
                    else config.terminal_success_reward
                )
        sim_prev = sim_now

        if is_success:
            success = True
            break

    return {
        "success": bool(success),
        "num_steps": len(actions),
        "actions": actions,
        "env_reward": float(env_reward),
    }


def run_on_cache(cache_path: Path) -> Dict[int, List[Dict]]:
    data = np.load(cache_path, allow_pickle=True)
    config = Config(
        n_variables=int(data["config_n_variables"]),
        mod=int(data["config_mod"]),
        max_complexity=int(np.max(data["generated_complexity"])),
        max_degree=int(data["config_max_degree"]),
        max_steps=int(data["config_max_steps"]),
    )

    target_coeffs = data["target_coeffs"]
    gen_c = data["generated_complexity"]

    by_complexity: Dict[int, List[Dict]] = {}
    for idx in range(target_coeffs.shape[0]):
        c = int(gen_c[idx])
        result = solve_greedy(target_coeffs[idx], config)
        result["target_idx"] = idx
        result["generated_complexity"] = c
        by_complexity.setdefault(c, []).append(result)

    return by_complexity


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="mode")

    p_cache = sub.add_parser(
        "from-cache", help="Run on a target_cache.npz produced by build_baseline_target_cache.py"
    )
    p_cache.add_argument("--target-cache", required=True)

    p_quick = sub.add_parser(
        "quick", help="Quick sanity: sample N random targets at one complexity and run greedy."
    )
    p_quick.add_argument("--num-trials", type=int, default=10)
    p_quick.add_argument("--complexity", type=int, default=2)
    p_quick.add_argument("--n-variables", type=int, default=2)
    p_quick.add_argument("--mod", type=int, default=5)
    p_quick.add_argument("--max-degree", type=int, default=6)
    p_quick.add_argument("--max-steps", type=int, default=14)
    p_quick.add_argument("--seed", type=int, default=0)

    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.mode == "from-cache":
        t0 = time.time()
        by_c = run_on_cache(Path(args.target_cache))
        for c in sorted(by_c):
            results = by_c[c]
            sr = sum(r["success"] for r in results) / len(results)
            avg_steps = np.mean([r["num_steps"] for r in results])
            avg_reward = np.mean([r["env_reward"] for r in results])
            print(
                f"C{c}: SR={sr:.1%} avg_steps={avg_steps:.2f} "
                f"avg_reward={avg_reward:+.3f} N={len(results)}"
            )
        print(f"wallclock={time.time() - t0:.1f}s")
    elif args.mode == "quick":
        import random as _random
        _random.seed(args.seed)
        np.random.seed(args.seed)
        from src.game_board.generator import generate_random_circuit
        config = Config(
            n_variables=args.n_variables,
            mod=args.mod,
            max_complexity=max(args.complexity, 2),
            max_degree=args.max_degree,
            max_steps=args.max_steps,
        )
        succ = 0
        for _ in range(args.num_trials):
            poly, _ = generate_random_circuit(config, args.complexity)
            r = solve_greedy(poly.coeffs.flatten(), config)
            succ += int(r["success"])
        print(f"Greedy on N={args.num_trials} random_C{args.complexity}: "
              f"SR={succ / args.num_trials:.1%}")
    else:
        print("Specify a mode: from-cache or quick")
        sys.exit(2)


if __name__ == "__main__":
    main()
