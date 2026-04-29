#!/usr/bin/env python3
"""Deterministic beam-search baseline using the rich baseline reward."""

from __future__ import annotations

import argparse
import random as _random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Config  # noqa: E402
from src.environment.fast_polynomial import FastPoly  # noqa: E402

from scripts.baseline_reward import (  # noqa: E402
    Action,
    BaselineRewardEvaluator,
    RewardState,
    apply_action,
    enumerate_actions,
    initial_nodes,
)


@dataclass
class _BeamState:
    nodes: List[FastPoly]
    reward_state: RewardState
    actions: List[Action]
    env_reward: float


def _rank_score(state: _BeamState, evaluator: BaselineRewardEvaluator) -> float:
    return state.env_reward + evaluator.best_similarity(state.nodes)


def solve_beam_search(
    target_coeffs_flat: np.ndarray,
    config: Config,
    beam_width: int = 64,
) -> Dict[str, float]:
    """Run deterministic top-k search on a single target."""
    shape = (config.effective_max_degree + 1,) * config.n_variables
    target_poly = FastPoly(
        np.asarray(target_coeffs_flat, dtype=np.int64).reshape(shape),
        config.mod,
    )
    evaluator = BaselineRewardEvaluator(config, target_poly)

    start = _BeamState(
        nodes=initial_nodes(config),
        reward_state=evaluator.initial_state(),
        actions=[],
        env_reward=0.0,
    )
    beam = [start]
    best_seen = start

    for _ in range(config.max_steps):
        candidates: List[_BeamState] = []
        successes: List[_BeamState] = []

        for state in beam:
            if len(state.nodes) >= config.max_nodes:
                continue
            for action in enumerate_actions(len(state.nodes)):
                new_poly = apply_action(state.nodes, action)
                reward_result = evaluator.step_reward(
                    state.nodes, new_poly, state.reward_state
                )
                next_state = _BeamState(
                    nodes=state.nodes + [new_poly],
                    reward_state=reward_result.next_state,
                    actions=state.actions + [action],
                    env_reward=state.env_reward + reward_result.reward,
                )
                if new_poly == target_poly:
                    successes.append(next_state)
                else:
                    candidates.append(next_state)

        if successes:
            best_success = max(
                successes,
                key=lambda s: (s.env_reward, -len(s.actions)),
            )
            return {
                "success": True,
                "num_steps": len(best_success.actions),
                "actions": best_success.actions,
                "env_reward": float(best_success.env_reward),
                "success_only_return": float(config.success_reward),
            }

        if not candidates:
            break

        candidates.sort(
            key=lambda s: (
                _rank_score(s, evaluator),
                s.env_reward,
                -len(s.actions),
            ),
            reverse=True,
        )
        beam = candidates[:max(1, beam_width)]
        if _rank_score(beam[0], evaluator) > _rank_score(best_seen, evaluator):
            best_seen = beam[0]

    return {
        "success": False,
        "num_steps": len(best_seen.actions),
        "actions": best_seen.actions,
        "env_reward": float(best_seen.env_reward),
        "success_only_return": 0.0,
    }


def run_on_cache(
    cache_path: Path,
    beam_width: int = 64,
) -> Dict[int, List[Dict]]:
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
        result = solve_beam_search(
            target_coeffs[idx],
            config,
            beam_width=beam_width,
        )
        result["target_idx"] = idx
        result["generated_complexity"] = c
        by_complexity.setdefault(c, []).append(result)

    return by_complexity


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="mode")

    p_cache = sub.add_parser("from-cache")
    p_cache.add_argument("--target-cache", required=True)
    p_cache.add_argument("--beam-width", type=int, default=64)

    p_quick = sub.add_parser("quick")
    p_quick.add_argument("--num-trials", type=int, default=5)
    p_quick.add_argument("--complexity", type=int, default=2)
    p_quick.add_argument("--beam-width", type=int, default=64)
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
        by_c = run_on_cache(Path(args.target_cache), args.beam_width)
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
        from src.game_board.generator import generate_random_circuit

        _random.seed(args.seed)
        np.random.seed(args.seed)
        config = Config(
            n_variables=args.n_variables,
            mod=args.mod,
            max_complexity=max(args.complexity, 2),
            max_degree=args.max_degree,
            max_steps=args.max_steps,
        )
        succ = 0
        t0 = time.time()
        for _ in range(args.num_trials):
            poly, _ = generate_random_circuit(config, args.complexity)
            r = solve_beam_search(
                poly.coeffs.flatten(), config, beam_width=args.beam_width
            )
            succ += int(r["success"])
        print(f"Beam search (width={args.beam_width}) on N={args.num_trials} "
              f"random_C{args.complexity}: SR={succ / args.num_trials:.1%} "
              f"wallclock={time.time() - t0:.1f}s")
    else:
        print("Specify a mode: from-cache or quick")
        sys.exit(2)


if __name__ == "__main__":
    main()
