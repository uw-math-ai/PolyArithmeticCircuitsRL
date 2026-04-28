#!/usr/bin/env python3
"""Uniform-prior MCTS baseline (no learned policy or value).

Pure-Python UCB1 MCTS, no JAX/mctx coupling. Uniform prior over valid actions
(each action equally likely a priori). Rollouts use random valid actions to
terminal. Backup uses standard UCB1 average reward.

Reward inside MCTS rollouts uses the env's reward signal (step_penalty +
success_reward + PBRS term-similarity shaping). Factor library bonuses are
intentionally excluded — replicating SymPy factorization at every rollout step
is too slow for a search-only baseline. We report two return values per target:

  - success_only_return: 1.0 if the target was constructed at any point during
    the chosen action sequence, else 0.0.
  - env_reward: cumulative shaped reward from the chosen action sequence
    (matches what the trained PPO agent saw, minus factor library bonuses).
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


def _initial_nodes(config: Config) -> List[FastPoly]:
    n_vars = config.n_variables
    max_deg = config.effective_max_degree
    mod = config.mod
    nodes = [FastPoly.variable(i, n_vars, max_deg, mod) for i in range(n_vars)]
    nodes.append(FastPoly.constant(1, n_vars, max_deg, mod))
    return nodes


def _enumerate_actions(num_nodes: int) -> List[Tuple[int, int, int]]:
    return [
        (op, i, j)
        for op in (0, 1)
        for i in range(num_nodes)
        for j in range(i, num_nodes)
    ]


def _apply(nodes: List[FastPoly], action: Tuple[int, int, int]) -> FastPoly:
    op, i, j = action
    return nodes[i] + nodes[j] if op == 0 else nodes[i] * nodes[j]


def _step_reward(
    nodes_before: List[FastPoly],
    nodes_after: List[FastPoly],
    target: FastPoly,
    is_success: bool,
    config: Config,
) -> float:
    """Step reward = step_penalty + (success_reward | PBRS shaping)."""
    if is_success:
        return config.step_penalty + config.success_reward
    if not config.use_reward_shaping:
        return config.step_penalty
    sim_before = max(n.term_similarity(target) for n in nodes_before)
    sim_after = max(n.term_similarity(target) for n in nodes_after)
    return config.step_penalty + (config.gamma * sim_after - sim_before)


@dataclass
class _Node:
    """MCTS tree node. State is the list of circuit polys + steps_taken."""
    nodes: List[FastPoly]
    steps_taken: int
    parent: Optional["_Node"] = None
    incoming_action: Optional[Tuple[int, int, int]] = None
    incoming_reward: float = 0.0  # reward earned arriving at this node
    is_terminal: bool = False
    is_success: bool = False

    children: Dict[Tuple[int, int, int], "_Node"] = field(default_factory=dict)
    untried_actions: List[Tuple[int, int, int]] = field(default_factory=list)

    visit_count: int = 0
    value_sum: float = 0.0  # sum of returns through this node


def _make_root(config: Config) -> _Node:
    nodes = _initial_nodes(config)
    untried = _enumerate_actions(len(nodes))
    return _Node(
        nodes=nodes,
        steps_taken=0,
        untried_actions=untried,
    )


def _expand_child(
    parent: _Node,
    action: Tuple[int, int, int],
    target: FastPoly,
    config: Config,
) -> _Node:
    new_poly = _apply(parent.nodes, action)
    is_success = (new_poly == target)
    reward = _step_reward(parent.nodes, parent.nodes + [new_poly], target,
                          is_success, config)
    new_nodes = parent.nodes + [new_poly]
    steps_taken = parent.steps_taken + 1
    is_terminal = (
        is_success
        or steps_taken >= config.max_steps
        or len(new_nodes) >= config.max_nodes
    )
    untried = (
        []
        if is_terminal
        else _enumerate_actions(len(new_nodes))
    )
    child = _Node(
        nodes=new_nodes,
        steps_taken=steps_taken,
        parent=parent,
        incoming_action=action,
        incoming_reward=reward,
        is_terminal=is_terminal,
        is_success=is_success,
        untried_actions=untried,
    )
    parent.children[action] = child
    return child


def _ucb_select(node: _Node, c_puct: float) -> _Node:
    """Pick the child with highest UCB1.

    Q(parent, action) = E[incoming_reward + V(child)], so we add the (deterministic)
    incoming_reward to the child's mean value before applying UCB1 exploration.
    Without this, terminal-success children look like V=0 and UCB1 can't tell them
    apart from terminal-failure children.
    """
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
    target: FastPoly,
    config: Config,
    rng: _random.Random,
) -> Tuple[float, bool]:
    """Random rollout from `node` to terminal. Returns (cumulative_reward, success)."""
    if node.is_terminal:
        return 0.0, node.is_success

    nodes = list(node.nodes)
    steps_taken = node.steps_taken
    cum_reward = 0.0
    success = False

    while True:
        if len(nodes) >= config.max_nodes or steps_taken >= config.max_steps:
            break
        actions = _enumerate_actions(len(nodes))
        if not actions:
            break
        op, i, j = actions[rng.randrange(len(actions))]
        new_poly = nodes[i] + nodes[j] if op == 0 else nodes[i] * nodes[j]
        is_success = (new_poly == target)
        reward = _step_reward(nodes, nodes + [new_poly], target, is_success, config)
        cum_reward += reward
        nodes = nodes + [new_poly]
        steps_taken += 1
        if is_success:
            success = True
            break

    return cum_reward, success


def _backup(leaf: _Node, rollout_return: float) -> None:
    """Walk from leaf to root updating visit/value counts.

    Args:
        leaf: the freshly-expanded (or selected terminal) node from which the
              rollout was performed.
        rollout_return: cumulative reward EARNED FROM leaf's state to terminal
                        (excludes the transition reward into leaf, which is
                        already captured as leaf.incoming_reward).
    """
    # Value at leaf = return-to-go from leaf = rollout_return.
    r_at_node = rollout_return
    node = leaf
    while node is not None:
        node.visit_count += 1
        node.value_sum += r_at_node
        if node.parent is not None:
            # Return-to-go at parent = parent->node transition reward + value at node.
            r_at_node = node.incoming_reward + r_at_node
        node = node.parent


def _run_one_simulation(
    root: _Node,
    target: FastPoly,
    config: Config,
    c_puct: float,
    rng: _random.Random,
) -> None:
    # Selection: descend until we hit a node with untried actions OR a terminal.
    cur = root
    while not cur.is_terminal and not cur.untried_actions:
        cur = _ucb_select(cur, c_puct)

    # Expansion: pop one untried action, create child (random — uniform prior).
    if not cur.is_terminal and cur.untried_actions:
        action = cur.untried_actions.pop(
            rng.randrange(len(cur.untried_actions))
        )
        cur = _expand_child(cur, action, target, config)

    # Simulation from cur to terminal.
    rollout_return, _ = _rollout(cur, target, config, rng)

    # Backup: rollout_return is from cur's state onward; leaf.incoming_reward
    # gets added on the way up.
    _backup(cur, rollout_return)


def solve_uniform_mcts(
    target_coeffs_flat: np.ndarray,
    config: Config,
    mcts_simulations: int = 32,
    c_puct: float = 1.4,
    seed: int = 0,
) -> Dict[str, float]:
    """Run uniform-prior UCB1 MCTS on a single target.

    Returns dict with success/num_steps/env_reward/success_only_return.
    """
    rng = _random.Random(seed)
    shape = (config.effective_max_degree + 1,) * config.n_variables
    target_poly = FastPoly(
        np.asarray(target_coeffs_flat, dtype=np.int64).reshape(shape),
        config.mod,
    )

    nodes = _initial_nodes(config)
    actions: List[Tuple[int, int, int]] = []
    env_reward = 0.0
    success = False

    # Episode loop: at each step, run mcts_simulations and pick the most-visited
    # root child. Then step the env (apply that action).
    for _ in range(config.max_steps):
        if len(nodes) >= config.max_nodes:
            break

        # Build a fresh tree from current state.
        root = _Node(
            nodes=list(nodes),
            steps_taken=len(actions),
            untried_actions=_enumerate_actions(len(nodes)),
        )
        if root.is_terminal or not root.untried_actions:
            break

        for _ in range(mcts_simulations):
            _run_one_simulation(root, target_poly, config, c_puct, rng)

        # Pick root child with best Q (incoming_reward + V), tie-break by visits.
        # Pure "most-visited" doesn't work at low budget: with branching ~12-20
        # and sims=32, many actions get only 1 visit and visit-count alone is
        # uninformative. Q-value selection cleanly prefers terminal-success
        # children whose incoming_reward is +9.9.
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

        # Apply.
        is_success = best_child.is_success
        new_poly = best_child.nodes[-1]
        env_reward += _step_reward(nodes, nodes + [new_poly], target_poly,
                                   is_success, config)
        nodes = nodes + [new_poly]
        actions.append(best_action)
        if is_success:
            success = True
            break

    return {
        "success": bool(success),
        "num_steps": len(actions),
        "actions": actions,
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
        result = solve_uniform_mcts(
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
            r = solve_uniform_mcts(
                poly.coeffs.flatten(), config,
                mcts_simulations=args.mcts_simulations,
                seed=args.seed + k,
            )
            succ += int(r["success"])
        print(f"Uniform MCTS (sims={args.mcts_simulations}) on N={args.num_trials} "
              f"random_C{args.complexity}: SR={succ / args.num_trials:.1%} "
              f"wallclock={time.time() - t0:.1f}s")
    else:
        print("Specify a mode: from-cache or quick")
        sys.exit(2)


if __name__ == "__main__":
    main()
