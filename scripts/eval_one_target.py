#!/usr/bin/env python3
"""Evaluate a trained PPO+MCTS (JAX) checkpoint against ONE specific target.

Loads a checkpoint, builds a target polynomial from user-supplied (coefficient,
exponent-tuple) pairs, and runs one greedy MCTS rollout. Prints the action
sequence the agent took, the polynomial it built at each step, and whether it
reached the target.

Example: evaluate (x0 + x1)^2 = x0^2 + 2*x0*x1 + x1^2 (mod 5):

    python scripts/eval_one_target.py \\
        --checkpoint results/ppo-mcts-jax_clean_onpath_SEQ_fixed_C2_p1_coeff3/checkpoint_00200.pkl \\
        --term 1,2,0 \\
        --term 2,1,1 \\
        --term 1,0,2

Each --term is "coeff,e_x0,e_x1[,e_x2,...]" — coefficient followed by the
exponent of each variable in order. So 1,2,0 means 1 * x0^2 * x1^0 = x0^2.

The agent treats the base nodes as: x0, x1, ..., x_{n-1}, then constant 1.
Action i,j corresponds to combining nodes at indices i and j (0-indexed).
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.algorithms.jax_env import (
    decode_action,
    get_observation,
    reset as env_reset,
    step as env_step,
)
from src.algorithms.ppo_mcts_jax import PPOMCTSJAXTrainer
from src.config import Config
from src.environment.fast_polynomial import FastPoly


def parse_term(term_str: str, n_vars: int) -> tuple[int, tuple[int, ...]]:
    parts = [int(p.strip()) for p in term_str.split(",")]
    if len(parts) != n_vars + 1:
        raise ValueError(
            f"--term '{term_str}' must have {n_vars + 1} integers "
            f"(1 coeff + {n_vars} exponents); got {len(parts)}."
        )
    coeff = parts[0]
    exps = tuple(parts[1:])
    if any(e < 0 for e in exps):
        raise ValueError(f"--term '{term_str}' has a negative exponent.")
    return coeff, exps


def build_target(
    terms: list[tuple[int, tuple[int, ...]]],
    n_vars: int,
    max_degree: int,
    mod: int,
) -> FastPoly:
    shape = (max_degree + 1,) * n_vars
    coeffs = np.zeros(shape, dtype=np.int64)
    for coeff, exps in terms:
        if any(e > max_degree for e in exps):
            raise ValueError(
                f"term coeff={coeff} exps={exps} exceeds max_degree={max_degree}"
            )
        coeffs[exps] = (coeffs[exps] + coeff) % mod
    return FastPoly(coeffs, mod)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, help="Path to checkpoint_XXXXX.pkl")
    p.add_argument(
        "--term", action="append", required=True,
        help="One term: 'coeff,e_x0,e_x1,...' (repeat for each term).",
    )
    p.add_argument("--mcts-simulations", type=int, default=64)
    p.add_argument(
        "--temperature", type=float, default=0.0,
        help="MCTS temperature (0 = argmax visit counts).",
    )
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def format_poly(poly: FastPoly, n_vars: int) -> str:
    coeffs = poly.coeffs
    nonzero = np.argwhere(coeffs != 0)
    if nonzero.size == 0:
        return "0"
    var_names = [f"x{k}" for k in range(n_vars)]
    terms = []
    for idx in nonzero:
        idx_t = tuple(int(x) for x in idx)
        c = int(coeffs[idx_t])
        parts = []
        for k, e in enumerate(idx_t):
            if e == 1:
                parts.append(var_names[k])
            elif e > 1:
                parts.append(f"{var_names[k]}^{e}")
        mono = "*".join(parts) if parts else "1"
        terms.append(f"{c}" if not parts else (mono if c == 1 else f"{c}*{mono}"))
    return " + ".join(terms)


def main() -> None:
    args = parse_args()

    with open(args.checkpoint, "rb") as f:
        state = pickle.load(f)
    config: Config = state["config"]

    # Eval is oracle-free: don't depend on the on-path cache being on disk.
    if getattr(config, "reward_mode", "legacy") == "clean_onpath":
        config.reward_mode = "clean_sparse"
        config.graph_onpath_cache_dir = None

    n_vars = config.n_variables
    mod = config.mod
    max_degree = config.effective_max_degree

    terms = [parse_term(t, n_vars) for t in args.term]
    target = build_target(terms, n_vars, max_degree, mod)
    print(f"Target ({n_vars} vars, mod {mod}, max_degree {max_degree}):")
    print(f"  {format_poly(target, n_vars)}")
    print(f"  raw coeffs shape={target.coeffs.shape}, nonzero={int((target.coeffs != 0).sum())}")
    print()

    trainer = PPOMCTSJAXTrainer(config, batch_size=1, fixed_complexities=None)
    trainer.load_checkpoint(args.checkpoint)
    config.mcts_simulations = args.mcts_simulations
    config.temperature_init = args.temperature

    env_config = trainer.env_config
    params = trainer.train_state.params

    target_arr = jnp.array(target.coeffs.flatten(), dtype=jnp.int32)[None, :]
    sgc, sga, sgl = trainer._prepare_initial_subgoals([target])
    library_coeffs, library_mask = trainer._export_library_cache()

    states = jax.vmap(
        lambda tc, c, a, l: env_reset(env_config, tc, c, a, l)
    )(target_arr, sgc, sga, sgl)

    # Initial node names for printing (base nodes: x0..x_{n-1}, then constant 1).
    base_names = [f"x{k}" for k in range(n_vars)] + ["1"]
    node_names = list(base_names)

    rng = jax.random.PRNGKey(args.seed)
    print(f"Initial nodes (idx -> poly):")
    for k, name in enumerate(base_names):
        print(f"  [{k}] {name}")
    print()
    print(f"Running greedy MCTS with {args.mcts_simulations} sims, temperature={args.temperature}")
    print()

    success = False
    for step_idx in range(config.max_steps):
        obs = jax.vmap(lambda s: get_observation(env_config, s))(states)
        rng, search_rng = jax.random.split(rng)
        policy = trainer._jit_batched_mcts(
            params, search_rng, obs, states, library_coeffs, library_mask,
        )
        action = int(jnp.argmax(policy.action_weights, axis=-1)[0])
        op, i, j = (int(x) for x in decode_action(jnp.int32(action), env_config.max_nodes))

        new_states, rewards, dones, successes_arr, *_rest = jax.vmap(
            lambda s, a: env_step(env_config, s, a, library_coeffs, library_mask)
        )(states, jnp.array([action], dtype=jnp.int32))

        # Read the freshly-built node from new_states.
        new_idx = int(new_states.num_nodes[0]) - 1
        new_coeffs = np.array(new_states.node_coeffs[0, new_idx]).reshape(target.coeffs.shape)
        new_poly = FastPoly(new_coeffs, mod)
        op_str = "+" if op == 0 else "*"
        new_name = f"({node_names[i]} {op_str} {node_names[j]})"
        node_names.append(new_name)

        reward = float(rewards[0])
        done = bool(dones[0])
        is_success = bool(successes_arr[0])
        print(
            f"step {step_idx + 1}: action={action} ({op_str} of [{i}]={node_names[i]} and [{j}]={node_names[j]})"
        )
        print(f"  built [{new_idx}] = {format_poly(new_poly, n_vars)}")
        print(f"  reward={reward:+.3f} done={done} success={is_success}")
        states = new_states

        if is_success:
            success = True
        if done:
            break

    print()
    if success:
        print("SUCCESS: agent constructed the target.")
        print(f"Final circuit: {node_names[-1]}")
    else:
        print("FAILURE: agent did not construct the target within max_steps.")
        final_idx = int(states.num_nodes[0]) - 1
        final_coeffs = np.array(states.node_coeffs[0, final_idx]).reshape(target.coeffs.shape)
        print(f"Last polynomial built: {format_poly(FastPoly(final_coeffs, mod), n_vars)}")


if __name__ == "__main__":
    main()
