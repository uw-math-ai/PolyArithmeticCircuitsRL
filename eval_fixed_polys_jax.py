#!/usr/bin/env python3
"""Evaluate JAX PPO+MCTS checkpoints on fixed compact polynomial targets.

This script is intentionally branch-local: it matches the older factor-library
JAX environment API where env_step returns factor/library/completion flags and
MCTS receives an exported library cache.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import random
import sys
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Config  # noqa: E402
from src.environment.fast_polynomial import FastPoly  # noqa: E402

jax = None
jnp = None
get_observation = None
make_env_config = None
env_reset = None
env_step = None
PPOMCTSJAXTrainer = None


def load_jax_deps() -> None:
    """Import JAX-only dependencies after CLI parsing.

    This keeps ``--help`` usable on login/local machines without JAX installed.
    """
    global jax, jnp, get_observation, make_env_config
    global env_reset, env_step, PPOMCTSJAXTrainer

    import jax as _jax
    import jax.numpy as _jnp

    import src.algorithms.jax_env as _jax_env
    from src.algorithms.jax_env import (
        get_observation as _get_observation,
        make_env_config as _make_env_config,
        reset as _env_reset,
        step as _env_step,
    )
    from src.algorithms.ppo_mcts_jax import (
        PPOMCTSJAXTrainer as _PPOMCTSJAXTrainer,
    )

    jax = _jax
    jnp = _jnp
    get_observation = _get_observation
    make_env_config = _make_env_config
    env_reset = _env_reset
    env_step = _env_step
    PPOMCTSJAXTrainer = _PPOMCTSJAXTrainer

    def _direct_poly_mul(a, b, mod: int, n_variables: int, max_degree: int):
        """Eval-only polynomial multiply that avoids cuDNN convolution."""
        lhs_idx, rhs_idx, out_idx = _direct_mul_indices(
            n_variables, max_degree
        )
        lhs = _jnp.array(lhs_idx, dtype=_jnp.int32)
        rhs = _jnp.array(rhs_idx, dtype=_jnp.int32)
        out = _jnp.array(out_idx, dtype=_jnp.int32)
        result = _jnp.zeros_like(a, dtype=_jnp.int32)
        prod = (a[lhs] * b[rhs]) % mod
        result = result.at[out].add(prod)
        return result % mod

    # The branch's default poly_mul uses lax.conv_general_dilated, which can
    # fail on some Hyak cuDNN stacks. Patch only this eval process.
    _jax_env.poly_mul = _direct_poly_mul


@lru_cache(maxsize=None)
def _direct_mul_indices(
    n_variables: int, max_degree: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Precompute sparse coefficient-product indices for direct multiplication."""
    dim = max_degree + 1
    shape = (dim,) * n_variables
    exponents = np.array(list(np.ndindex(shape)), dtype=np.int32)

    lhs: List[int] = []
    rhs: List[int] = []
    out: List[int] = []
    for i, exp_i in enumerate(exponents):
        for j, exp_j in enumerate(exponents):
            exp_out = exp_i + exp_j
            if np.all(exp_out <= max_degree):
                lhs.append(i)
                rhs.append(j)
                out.append(int(np.ravel_multi_index(tuple(exp_out), shape)))

    return (
        np.array(lhs, dtype=np.int32),
        np.array(rhs, dtype=np.int32),
        np.array(out, dtype=np.int32),
    )


@dataclass(frozen=True)
class TargetSpec:
    name: str
    complexity: int
    optimal_steps: int
    target: FastPoly
    recipe: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate fixed compact C5-C8 targets with JAX PPO+MCTS."
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="results/ppo-mcts-jax_fl_C5_C8",
        help="Directory containing checkpoint_XXXXX.pkl files.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        nargs="+",
        default=[50, 100, 150, 200, 250, 300, 350, 400, 450, 500],
        help="Checkpoint iterations to evaluate.",
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        default=100,
        help="Episodes per target per checkpoint.",
    )
    parser.add_argument(
        "--mcts-simulations",
        type=int,
        default=32,
        help="MCTS simulations per move.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Parallel eval episodes per batch.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="MCTS temperature. Actions are still chosen by argmax visits.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for eval RNGs.",
    )
    parser.add_argument(
        "--out-prefix",
        type=str,
        default=None,
        help="Output prefix for .csv and _trajectories.jsonl files.",
    )
    parser.add_argument(
        "--max-samples-per-outcome",
        type=int,
        default=1,
        help="Number of success/failure trajectories to save per target.",
    )
    return parser.parse_args()


def decode_action_int(action_idx: int, max_nodes: int) -> tuple[int, int, int]:
    """Decode this repo's upper-triangular action encoding."""
    op = action_idx % 2
    pair_idx = action_idx // 2
    discriminant = (2 * max_nodes + 1) ** 2 - 8 * pair_idx
    i = int((2 * max_nodes + 1 - math.isqrt(discriminant)) // 2)

    row_start = i * max_nodes - i * (i - 1) // 2
    while row_start > pair_idx and i > 0:
        i -= 1
        row_start = i * max_nodes - i * (i - 1) // 2

    next_row_start = (i + 1) * max_nodes - (i + 1) * i // 2
    while pair_idx >= next_row_start and i < max_nodes - 1:
        i += 1
        row_start = i * max_nodes - i * (i - 1) // 2
        next_row_start = (i + 1) * max_nodes - (i + 1) * i // 2

    j = i + (pair_idx - row_start)
    return op, i, j


def coeff_vector_to_fastpoly(config: Config, coeff_vector: np.ndarray) -> FastPoly:
    shape = (config.effective_max_degree + 1,) * config.n_variables
    coeffs = np.array(coeff_vector, dtype=np.int64).reshape(shape)
    return FastPoly(coeffs, config.mod)


def build_fixed_targets(config: Config) -> List[TargetSpec]:
    if config.n_variables < 2:
        raise ValueError("fixed polynomial eval requires at least two variables")

    max_deg = config.effective_max_degree
    mod = config.mod
    x0 = FastPoly.variable(0, config.n_variables, max_deg, mod)
    x1 = FastPoly.variable(1, config.n_variables, max_deg, mod)
    one = FastPoly.constant(1, config.n_variables, max_deg, mod)

    targets: List[TargetSpec] = []

    # C5: a=x0+x1; b=a*a; c=b+1; d=c*c; T=d+x0
    a = x0 + x1
    b = a * a
    c = b + one
    d = c * c
    t = d + x0
    targets.append(TargetSpec(
        name="C5_square_shift",
        complexity=5,
        optimal_steps=5,
        target=t,
        recipe="a=x0+x1; b=a*a; c=b+1; d=c*c; T=d+x0",
    ))

    # C6: a=x0+1; b=x1+1; c=a*a; d=b*b; e=c*d; T=e+x0
    a = x0 + one
    b = x1 + one
    c = a * a
    d = b * b
    e = c * d
    t = e + x0
    targets.append(TargetSpec(
        name="C6_product_of_squares",
        complexity=6,
        optimal_steps=6,
        target=t,
        recipe="a=x0+1; b=x1+1; c=a*a; d=b*b; e=c*d; T=e+x0",
    ))

    # C7: a=x0+x1; b=a*a; c=b*b; d=c+1; e=d*d; f=e+x1; T=f+x0
    a = x0 + x1
    b = a * a
    c = b * b
    d = c + one
    e = d * d
    f = e + x1
    t = f + x0
    targets.append(TargetSpec(
        name="C7_nested_power_shift",
        complexity=7,
        optimal_steps=7,
        target=t,
        recipe="a=x0+x1; b=a*a; c=b*b; d=c+1; e=d*d; f=e+x1; T=f+x0",
    ))

    # C8: a=x0+1; b=x1+1; c=a*a; d=b*b; e=c*d; f=e+1; g=f*f; T=g+x0
    a = x0 + one
    b = x1 + one
    c = a * a
    d = b * b
    e = c * d
    f = e + one
    g = f * f
    t = g + x0
    targets.append(TargetSpec(
        name="C8_product_then_square",
        complexity=8,
        optimal_steps=8,
        target=t,
        recipe="a=x0+1; b=x1+1; c=a*a; d=b*b; e=c*d; f=e+1; g=f*f; T=g+x0",
    ))

    return targets


def tree_take(tree: Any, index: int) -> Any:
    return jax.tree.map(lambda x: np.array(x[index]), tree)


def make_trace_step(
    config: Config,
    step_index: int,
    action: int,
    previous_num_nodes: int,
    next_state_i: Any,
    reward: float,
    done: bool,
    success: bool,
    factor_hit: bool,
    library_hit: bool,
    additive_complete: bool,
    mult_complete: bool,
) -> Dict[str, Any]:
    op, i, j = decode_action_int(action, config.max_nodes)
    result_vec = np.array(next_state_i.node_coeffs[previous_num_nodes])
    result_poly = coeff_vector_to_fastpoly(config, result_vec)
    return {
        "step": step_index + 1,
        "action": int(action),
        "op": "add" if op == 0 else "mul",
        "i": int(i),
        "j": int(j),
        "new_node": int(previous_num_nodes),
        "new_poly": repr(result_poly),
        "reward": float(reward),
        "done": bool(done),
        "success": bool(success),
        "factor_hit": bool(factor_hit),
        "library_hit": bool(library_hit),
        "additive_complete": bool(additive_complete),
        "mult_complete": bool(mult_complete),
    }


def evaluate_target(
    trainer: PPOMCTSJAXTrainer,
    spec: TargetSpec,
    num_trials: int,
    batch_size: int,
    rng: jax.Array,
    max_samples_per_outcome: int,
) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    config = trainer.config
    env_config = trainer.env_config
    params = trainer.train_state.params

    successes = 0
    reward_sum = 0.0
    successful_steps: List[int] = []
    all_steps: List[int] = []
    saved_successes: List[Dict[str, Any]] = []
    saved_failures: List[Dict[str, Any]] = []

    num_done = 0
    pbar = tqdm(total=num_trials, desc=f"    {spec.name}", unit="ep")

    while num_done < num_trials:
        bsz = min(batch_size, num_trials - num_done)
        targets = [spec.target.copy() for _ in range(bsz)]
        target_arrays = jnp.stack([
            jnp.array(t.coeffs.flatten(), dtype=jnp.int32)
            for t in targets
        ])
        subgoal_coeffs, subgoal_active, subgoal_library_known = (
            trainer._prepare_initial_subgoals(targets)
        )
        library_coeffs, library_mask = trainer._export_library_cache()

        states = jax.vmap(
            lambda tc, sgc, sga, sgl: env_reset(
                env_config, tc, sgc, sga, sgl
            )
        )(target_arrays, subgoal_coeffs, subgoal_active, subgoal_library_known)

        episode_rewards = np.zeros(bsz, dtype=np.float64)
        episode_successes = np.zeros(bsz, dtype=bool)
        episode_steps = np.zeros(bsz, dtype=np.int32)
        active = np.ones(bsz, dtype=bool)
        traces: List[List[Dict[str, Any]]] = [[] for _ in range(bsz)]

        for step_idx in range(config.max_steps):
            if not active.any():
                break

            obs_batch = jax.vmap(
                lambda s: get_observation(env_config, s)
            )(states)

            rng, search_rng = jax.random.split(rng)
            policy_output = trainer._jit_batched_mcts(
                params,
                search_rng,
                obs_batch,
                states,
                library_coeffs,
                library_mask,
            )
            actions = jnp.argmax(policy_output.action_weights, axis=-1)

            (
                next_states,
                rewards,
                dones,
                successes_batch,
                factor_hits,
                library_hits,
                additive_complete,
                mult_complete,
            ) = jax.vmap(
                lambda s, a: env_step(
                    env_config, s, a, library_coeffs, library_mask
                )
            )(states, actions)

            actions_np = np.array(actions)
            rewards_np = np.array(rewards)
            dones_np = np.array(dones)
            successes_np = np.array(successes_batch)
            factor_hits_np = np.array(factor_hits)
            library_hits_np = np.array(library_hits)
            additive_complete_np = np.array(additive_complete)
            mult_complete_np = np.array(mult_complete)
            previous_num_nodes_np = np.array(states.num_nodes)
            next_steps_np = np.array(next_states.steps_taken)

            for i in range(bsz):
                if not active[i]:
                    continue

                next_state_i = tree_take(next_states, i)
                traces[i].append(make_trace_step(
                    config=config,
                    step_index=step_idx,
                    action=int(actions_np[i]),
                    previous_num_nodes=int(previous_num_nodes_np[i]),
                    next_state_i=next_state_i,
                    reward=float(rewards_np[i]),
                    done=bool(dones_np[i]),
                    success=bool(successes_np[i]),
                    factor_hit=bool(factor_hits_np[i]),
                    library_hit=bool(library_hits_np[i]),
                    additive_complete=bool(additive_complete_np[i]),
                    mult_complete=bool(mult_complete_np[i]),
                ))
                episode_rewards[i] += rewards_np[i]
                episode_steps[i] = int(next_steps_np[i])
                if successes_np[i]:
                    episode_successes[i] = True
                if dones_np[i]:
                    active[i] = False

            states = next_states

        for i in range(bsz):
            reward_sum += float(episode_rewards[i])
            all_steps.append(int(episode_steps[i]))
            if episode_successes[i]:
                successes += 1
                successful_steps.append(int(episode_steps[i]))
                if len(saved_successes) < max_samples_per_outcome:
                    saved_successes.append({
                        "outcome": "success",
                        "steps": int(episode_steps[i]),
                        "reward": float(episode_rewards[i]),
                        "trace": traces[i],
                    })
            else:
                if len(saved_failures) < max_samples_per_outcome:
                    saved_failures.append({
                        "outcome": "failure",
                        "steps": int(episode_steps[i]),
                        "reward": float(episode_rewards[i]),
                        "trace": traces[i],
                    })

        num_done += bsz
        pbar.update(bsz)

    pbar.close()

    success_rate = successes / max(num_trials, 1)
    metrics = {
        "target": spec.name,
        "complexity": spec.complexity,
        "optimal_steps": spec.optimal_steps,
        "num_trials": num_trials,
        "successes": successes,
        "success_rate": success_rate,
        "avg_reward": reward_sum / max(num_trials, 1),
        "avg_steps_all": float(np.mean(all_steps)) if all_steps else float("nan"),
        "avg_success_steps": (
            float(np.mean(successful_steps)) if successful_steps else float("nan")
        ),
        "min_success_steps": min(successful_steps) if successful_steps else "",
        "target_poly": repr(spec.target),
        "recipe": spec.recipe,
    }
    return metrics, saved_successes + saved_failures


def load_checkpoint_config(path: Path) -> Config:
    with path.open("rb") as f:
        state = pickle.load(f)
    return state["config"]


def main() -> None:
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    iterations = sorted(set(args.iterations))
    checkpoint_dir = Path(args.checkpoint_dir)
    first_ckpt = checkpoint_dir / f"checkpoint_{iterations[0]:05d}.pkl"
    if not first_ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {first_ckpt}")

    config = load_checkpoint_config(first_ckpt)
    config.mcts_simulations = args.mcts_simulations
    config.temperature_init = args.temperature

    if config.max_complexity < 8:
        raise ValueError(
            f"checkpoint config max_complexity={config.max_complexity}; "
            "fixed C5-C8 eval expects max_complexity >= 8"
        )

    load_jax_deps()

    trainer = PPOMCTSJAXTrainer(
        config,
        batch_size=args.batch_size,
        fixed_complexities=None,
    )
    trainer.env_config = make_env_config(config)

    targets = build_fixed_targets(config)
    for spec in targets:
        if config.max_steps < spec.optimal_steps:
            print(
                f"WARNING: max_steps={config.max_steps} is below "
                f"{spec.name} optimal_steps={spec.optimal_steps}",
                flush=True,
            )

    out_prefix = Path(args.out_prefix) if args.out_prefix else (
        checkpoint_dir / "fixed_poly_eval"
    )
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = out_prefix.with_suffix(".csv")
    jsonl_path = out_prefix.with_name(out_prefix.name + "_trajectories.jsonl")

    rows: List[Dict[str, Any]] = []
    jsonl_records: List[Dict[str, Any]] = []

    print(
        f"Evaluating {checkpoint_dir} iterations={iterations} "
        f"num_trials={args.num_trials} mcts={args.mcts_simulations} "
        f"temperature={args.temperature}",
        flush=True,
    )

    for iteration in iterations:
        ckpt_path = checkpoint_dir / f"checkpoint_{iteration:05d}.pkl"
        if not ckpt_path.exists():
            print(f"WARNING: missing {ckpt_path}; skipping", flush=True)
            continue

        print(f"\n=== checkpoint {iteration} ===", flush=True)
        trainer.load_checkpoint(str(ckpt_path))

        for target_index, spec in enumerate(targets):
            rng = jax.random.PRNGKey(args.seed + iteration * 100 + target_index)
            start = time.time()
            metrics, samples = evaluate_target(
                trainer=trainer,
                spec=spec,
                num_trials=args.num_trials,
                batch_size=args.batch_size,
                rng=rng,
                max_samples_per_outcome=args.max_samples_per_outcome,
            )
            elapsed = time.time() - start
            metrics["iteration"] = iteration
            metrics["elapsed_sec"] = elapsed
            rows.append(metrics)

            for sample in samples:
                record = {
                    "iteration": iteration,
                    "target": spec.name,
                    "complexity": spec.complexity,
                    "optimal_steps": spec.optimal_steps,
                    "recipe": spec.recipe,
                    **sample,
                }
                jsonl_records.append(record)

            avg_success_steps = metrics["avg_success_steps"]
            avg_success_steps_str = (
                f"{avg_success_steps:.2f}"
                if isinstance(avg_success_steps, float)
                and not math.isnan(avg_success_steps)
                else "nan"
            )
            print(
                f"{spec.name} C{spec.complexity}: "
                f"SR={metrics['success_rate']:.1%} "
                f"successes={metrics['successes']}/{args.num_trials} "
                f"avg_success_steps={avg_success_steps_str} "
                f"min_success_steps={metrics['min_success_steps']} "
                f"avg_reward={metrics['avg_reward']:.3f} "
                f"({elapsed:.1f}s)",
                flush=True,
            )

    fieldnames = [
        "iteration",
        "target",
        "complexity",
        "optimal_steps",
        "num_trials",
        "successes",
        "success_rate",
        "avg_reward",
        "avg_steps_all",
        "avg_success_steps",
        "min_success_steps",
        "elapsed_sec",
        "target_poly",
        "recipe",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    with jsonl_path.open("w", encoding="utf-8") as f:
        for record in jsonl_records:
            f.write(json.dumps(record) + "\n")

    print(f"\nSaved {csv_path}", flush=True)
    print(f"Saved {jsonl_path}", flush=True)


if __name__ == "__main__":
    main()
