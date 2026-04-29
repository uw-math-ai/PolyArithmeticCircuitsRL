#!/usr/bin/env python3
"""Factor-aware uniform-prior MCTS baseline.

This is the same pure-Python UCB1 search shape as baseline_uniform_mcts.py:
uniform action expansion, random rollouts, no learned policy/value, and a fresh
tree at every real environment step. The only difference is the reward used
inside tree search and final accounting: it includes per-target factor subgoal
and completion bonuses from scripts.baseline_reward.
"""

from __future__ import annotations

import argparse
import math
import random as _random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
class _Node:
    """MCTS tree node. State is circuit nodes plus rich reward bookkeeping."""

    nodes: List[FastPoly]
    steps_taken: int
    reward_state: RewardState
    parent: Optional["_Node"] = None
    incoming_action: Optional[Action] = None
    incoming_reward: float = 0.0
    is_terminal: bool = False
    is_success: bool = False

    children: Dict[Action, "_Node"] = field(default_factory=dict)
    untried_actions: List[Action] = field(default_factory=list)

    visit_count: int = 0
    value_sum: float = 0.0


def _expand_child(
    parent: _Node,
    action: Action,
    evaluator: BaselineRewardEvaluator,
    config: Config,
) -> _Node:
    new_poly = apply_action(parent.nodes, action)
    reward_result = evaluator.step_reward(
        parent.nodes, new_poly, parent.reward_state
    )
    new_nodes = parent.nodes + [new_poly]
    steps_taken = parent.steps_taken + 1
    is_success = new_poly == evaluator.target
    is_terminal = (
        is_success
        or steps_taken >= config.max_steps
        or len(new_nodes) >= config.max_nodes
    )
    child = _Node(
        nodes=new_nodes,
        steps_taken=steps_taken,
        reward_state=reward_result.next_state,
        parent=parent,
        incoming_action=action,
        incoming_reward=reward_result.reward,
        is_terminal=is_terminal,
        is_success=is_success,
        untried_actions=[] if is_terminal else enumerate_actions(len(new_nodes)),
    )
    parent.children[action] = child
    return child


def _ucb_select(node: _Node, c_puct: float) -> _Node:
    parent_visits = max(1, node.visit_count)
    log_pv = math.log(parent_visits)
    best_score = -math.inf
    best_child = None
    for child in node.children.values():
        if child.visit_count == 0:
            return child
        q = child.incoming_reward + (child.value_sum / child.visit_count)
        explore = c_puct * math.sqrt(log_pv / child.visit_count)
        score = q + explore
        if score > best_score:
            best_score = score
            best_child = child
    assert best_child is not None
    return best_child


def _rollout(
    node: _Node,
    evaluator: BaselineRewardEvaluator,
    config: Config,
    rng: _random.Random,
) -> Tuple[float, bool]:
    if node.is_terminal:
        return 0.0, node.is_success

    nodes = list(node.nodes)
    reward_state = node.reward_state
    steps_taken = node.steps_taken
    cum_reward = 0.0
    success = False

    while True:
        if len(nodes) >= config.max_nodes or steps_taken >= config.max_steps:
            break
        actions = enumerate_actions(len(nodes))
        if not actions:
            break
        action = actions[rng.randrange(len(actions))]
        new_poly = apply_action(nodes, action)
        reward_result = evaluator.step_reward(nodes, new_poly, reward_state)
        cum_reward += reward_result.reward
        reward_state = reward_result.next_state
        nodes = nodes + [new_poly]
        steps_taken += 1
        if new_poly == evaluator.target:
            success = True
            break

    return cum_reward, success


def _backup(leaf: _Node, rollout_return: float) -> None:
    r_at_node = rollout_return
    node = leaf
    while node is not None:
        node.visit_count += 1
        node.value_sum += r_at_node
        if node.parent is not None:
            r_at_node = node.incoming_reward + r_at_node
        node = node.parent


def _run_one_simulation(
    root: _Node,
    evaluator: BaselineRewardEvaluator,
    config: Config,
    c_puct: float,
    rng: _random.Random,
) -> None:
    cur = root
    while not cur.is_terminal and not cur.untried_actions:
        cur = _ucb_select(cur, c_puct)

    if not cur.is_terminal and cur.untried_actions:
        action = cur.untried_actions.pop(
            rng.randrange(len(cur.untried_actions))
        )
        cur = _expand_child(cur, action, evaluator, config)

    rollout_return, _ = _rollout(cur, evaluator, config, rng)
    _backup(cur, rollout_return)


def solve_factor_mcts(
    target_coeffs_flat: np.ndarray,
    config: Config,
    mcts_simulations: int = 32,
    c_puct: float = 1.4,
    seed: int = 0,
) -> Dict[str, float]:
    """Run factor-aware uniform-prior UCB1 MCTS on one target."""
    rng = _random.Random(seed)
    shape = (config.effective_max_degree + 1,) * config.n_variables
    target_poly = FastPoly(
        np.asarray(target_coeffs_flat, dtype=np.int64).reshape(shape),
        config.mod,
    )
    evaluator = BaselineRewardEvaluator(config, target_poly)

    nodes = initial_nodes(config)
    reward_state = evaluator.initial_state()
    actions_taken: List[Action] = []
    env_reward = 0.0
    success = False

    for _ in range(config.max_steps):
        if len(nodes) >= config.max_nodes:
            break

        root = _Node(
            nodes=list(nodes),
            steps_taken=len(actions_taken),
            reward_state=reward_state,
            untried_actions=enumerate_actions(len(nodes)),
        )
        if not root.untried_actions:
            break

        for _ in range(mcts_simulations):
            _run_one_simulation(root, evaluator, config, c_puct, rng)

        if not root.children:
            break

        def _select_score(child: _Node) -> Tuple[float, int]:
            q = child.incoming_reward + (
                child.value_sum / child.visit_count if child.visit_count else 0.0
            )
            return (q, child.visit_count)

        best_action, best_child = max(
            root.children.items(),
            key=lambda kv: _select_score(kv[1]),
        )

        env_reward += best_child.incoming_reward
        reward_state = best_child.reward_state
        nodes = best_child.nodes
        actions_taken.append(best_action)
        if best_child.is_success:
            success = True
            break

    return {
        "success": bool(success),
        "num_steps": len(actions_taken),
        "actions": actions_taken,
        "env_reward": float(env_reward),
        "success_only_return": float(config.success_reward) if success else 0.0,
    }


def run_on_cache(
    cache_path: Path,
    mcts_simulations: int = 32,
    seed: int = 0,
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
        result = solve_factor_mcts(
            target_coeffs[idx],
            config,
            mcts_simulations=mcts_simulations,
            seed=seed + idx,
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
    p_cache.add_argument("--mcts-simulations", type=int, default=32)
    p_cache.add_argument("--seed", type=int, default=0)

    p_quick = sub.add_parser("quick")
    p_quick.add_argument("--num-trials", type=int, default=5)
    p_quick.add_argument("--complexity", type=int, default=2)
    p_quick.add_argument("--mcts-simulations", type=int, default=16)
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
        by_c = run_on_cache(Path(args.target_cache), args.mcts_simulations, args.seed)
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
        for k in range(args.num_trials):
            poly, _ = generate_random_circuit(config, args.complexity)
            r = solve_factor_mcts(
                poly.coeffs.flatten(), config,
                mcts_simulations=args.mcts_simulations,
                seed=args.seed + k,
            )
            succ += int(r["success"])
        print(f"Factor MCTS (sims={args.mcts_simulations}) on N={args.num_trials} "
              f"random_C{args.complexity}: SR={succ / args.num_trials:.1%} "
              f"wallclock={time.time() - t0:.1f}s")
    else:
        print("Specify a mode: from-cache or quick")
        sys.exit(2)


if __name__ == "__main__":
    main()
