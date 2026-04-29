#!/usr/bin/env python3
"""Run baseline comparison harness on a shared target cache.

Reads a target cache produced by build_baseline_target_cache.py, runs each
selected method on every target, and writes one CSV per (method × generated
complexity) plus stratified rows by true_min_complexity for the PPO method.

CSV schema:
    method, generated_complexity, true_min_complexity, num_trials,
    success_rate, avg_steps_success, avg_steps_all, avg_reward, wallclock_sec

Methods (selected via --methods):
    greedy_1step
    uniform_mcts_32   (plain uniform-prior MCTS)
    onpath_mcts_32    (uniform MCTS + this branch's clean-onpath reward)
    beam_search_64    (deterministic beam; uses on-path reward if cache is supplied)
    ppo_mcts_checkpoint   (requires --checkpoint, JAX, mctx)
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Config  # noqa: E402

# Local baselines (CPU-only, no JAX needed):
import scripts.baseline_beam_search as bbeam  # noqa: E402
import scripts.baseline_greedy as bgreedy  # noqa: E402
import scripts.baseline_onpath_mcts as bonpath  # noqa: E402
import scripts.baseline_uniform_mcts as bmcts  # noqa: E402
from src.game_board.on_path import OnPathCache  # noqa: E402


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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target-cache", required=True)
    p.add_argument(
        "--on-path-cache-dir",
        default=None,
        help="Required for onpath_mcts. Optional for beam_search to use on-path reward.",
    )
    p.add_argument(
        "--methods",
        nargs="+",
        default=["greedy_1step", "uniform_mcts_32"],
        help="Methods to run. Add 'ppo_mcts_checkpoint' if --checkpoint is set.",
    )
    p.add_argument("--mcts-simulations", type=int, default=32)
    p.add_argument("--beam-width", type=int, default=64)
    p.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Required if 'ppo_mcts_checkpoint' is in --methods. "
             "E.g., results/ppo-mcts-jax_fl_C5_C8/checkpoint_00500.pkl",
    )
    p.add_argument("--ppo-batch-size", type=int, default=32)
    p.add_argument("--out-csv", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", type=str, default=None)
    p.add_argument("--wandb-run-name", type=str, default=None)
    return p.parse_args()


def _int_suffix(method: str, default: int) -> int:
    tail = method.split("_")[-1]
    return int(tail) if tail.isdigit() else default


def _load_cache(path: Path) -> Dict[str, Any]:
    data = np.load(path, allow_pickle=True)
    config = Config(
        n_variables=int(data["config_n_variables"]),
        mod=int(data["config_mod"]),
        max_complexity=int(np.max(data["generated_complexity"])),
        max_degree=int(data["config_max_degree"]),
        max_steps=int(data["config_max_steps"]),
    )
    if "source" in data.files and str(data["source"]) == "on_path_cache":
        config.reward_mode = "clean_onpath"
    return {
        "target_coeffs": data["target_coeffs"],
        "generated_complexity": data["generated_complexity"],
        "true_min_complexity": data["true_min_complexity"],
        "target_ids": data["target_ids"] if "target_ids" in data.files else None,
        "config": config,
    }


def _load_on_path_cache(
    cache: Dict[str, Any],
    on_path_cache_dir: Optional[str],
) -> Optional[OnPathCache]:
    if on_path_cache_dir is None:
        return None
    complexities = sorted(set(map(int, cache["generated_complexity"])))
    return OnPathCache.load(Path(on_path_cache_dir), cache["config"], complexities)


def _target_context(
    cache: Dict[str, Any],
    on_path_cache: Optional[OnPathCache],
    idx: int,
):
    if on_path_cache is None:
        return None
    target_ids = cache.get("target_ids")
    if target_ids is None:
        raise ValueError("target cache has no target_ids; rebuild from OnPathCache")
    target_id = int(target_ids[idx])
    if target_id < 0:
        raise ValueError("target cache target_ids are invalid for on-path baselines")
    c = int(cache["generated_complexity"][idx])
    return on_path_cache.by_complexity[c].target_context(target_id)


def _aggregate_rows(
    method_name: str,
    results: List[Dict[str, Any]],
    cache: Dict[str, Any],
    wallclock: float,
) -> List[Dict[str, Any]]:
    """One row per generated_complexity, plus per-(generated, true) pairs."""
    rows: List[Dict[str, Any]] = []

    gen_c = cache["generated_complexity"]
    true_c = cache["true_min_complexity"]

    by_gen: Dict[int, List[Dict[str, Any]]] = {}
    by_pair: Dict[tuple, List[Dict[str, Any]]] = {}

    for r in results:
        idx = r["target_idx"]
        g = int(gen_c[idx])
        t = int(true_c[idx])
        by_gen.setdefault(g, []).append(r)
        if t >= 0:
            by_pair.setdefault((g, t), []).append(r)

    def _summarize(rows_subset: List[Dict[str, Any]]) -> Dict[str, float]:
        n = len(rows_subset)
        succ = [r for r in rows_subset if r["success"]]
        return {
            "num_trials": n,
            "success_rate": len(succ) / n if n else 0.0,
            "avg_steps_success": (
                float(np.mean([r["num_steps"] for r in succ])) if succ else 0.0
            ),
            "avg_steps_all": (
                float(np.mean([r["num_steps"] for r in rows_subset])) if n else 0.0
            ),
            "avg_reward": (
                float(np.mean([r["env_reward"] for r in rows_subset])) if n else 0.0
            ),
        }

    for g in sorted(by_gen):
        agg = _summarize(by_gen[g])
        rows.append({
            "method": method_name,
            "generated_complexity": g,
            "true_min_complexity": -1,  # -1 indicates "all true_complexities"
            "wallclock_sec": wallclock if g == sorted(by_gen)[0] else 0.0,
            **agg,
        })

    for (g, t) in sorted(by_pair):
        agg = _summarize(by_pair[(g, t)])
        rows.append({
            "method": method_name,
            "generated_complexity": g,
            "true_min_complexity": t,
            "wallclock_sec": 0.0,
            **agg,
        })

    return rows


def _run_greedy(cache: Dict[str, Any]) -> List[Dict[str, Any]]:
    config = cache["config"]
    coeffs = cache["target_coeffs"]
    results: List[Dict[str, Any]] = []
    for idx in range(coeffs.shape[0]):
        r = bgreedy.solve_greedy(coeffs[idx], config)
        r["target_idx"] = idx
        results.append(r)
    return results


def _run_uniform_mcts(
    cache: Dict[str, Any],
    mcts_simulations: int,
    seed: int,
) -> List[Dict[str, Any]]:
    config = cache["config"]
    coeffs = cache["target_coeffs"]
    results: List[Dict[str, Any]] = []
    for idx in range(coeffs.shape[0]):
        r = bmcts.solve_uniform_mcts(
            coeffs[idx],
            config,
            mcts_simulations=mcts_simulations,
            seed=seed + idx,
        )
        r["target_idx"] = idx
        results.append(r)
    return results


def _run_onpath_mcts(
    cache: Dict[str, Any],
    on_path_cache: OnPathCache,
    mcts_simulations: int,
    seed: int,
) -> List[Dict[str, Any]]:
    config = cache["config"]
    results: List[Dict[str, Any]] = []
    for idx in range(cache["target_coeffs"].shape[0]):
        context = _target_context(cache, on_path_cache, idx)
        r = bonpath.solve_onpath_mcts(
            context,
            config,
            mcts_simulations=mcts_simulations,
            seed=seed + idx,
        )
        r["target_idx"] = idx
        results.append(r)
    return results


def _run_beam_search(
    cache: Dict[str, Any],
    beam_width: int,
    on_path_cache: Optional[OnPathCache] = None,
) -> List[Dict[str, Any]]:
    config = cache["config"]
    coeffs = cache["target_coeffs"]
    results: List[Dict[str, Any]] = []
    for idx in range(coeffs.shape[0]):
        context = _target_context(cache, on_path_cache, idx) if on_path_cache else None
        r = bbeam.solve_beam_search(
            coeffs[idx],
            config,
            beam_width=beam_width,
            context=context,
        )
        r["target_idx"] = idx
        results.append(r)
    return results


def _run_ppo_checkpoint(
    cache: Dict[str, Any],
    checkpoint_path: str,
    batch_size: int,
    mcts_simulations: int,
) -> List[Dict[str, Any]]:
    """Inference-only PPO eval over the shared cache. Imports JAX lazily."""
    import pickle
    import jax
    import jax.numpy as jnp
    import src.algorithms.jax_env as jax_env
    from src.algorithms.jax_env import (
        make_env_config, reset as env_reset, step as env_step,
        get_observation,
    )
    from src.algorithms.ppo_mcts_jax import PPOMCTSJAXTrainer

    def _direct_poly_mul(a, b, mod: int, n_variables: int, max_degree: int):
        """Eval-only polynomial multiply that avoids cuDNN convolution."""
        lhs_idx, rhs_idx, out_idx = _direct_mul_indices(
            n_variables, max_degree
        )
        lhs = jnp.array(lhs_idx, dtype=jnp.int32)
        rhs = jnp.array(rhs_idx, dtype=jnp.int32)
        out = jnp.array(out_idx, dtype=jnp.int32)
        result = jnp.zeros_like(a, dtype=jnp.int32)
        prod = (a[lhs] * b[rhs]) % mod
        result = result.at[out].add(prod)
        return result % mod

    # The branch's default poly_mul uses lax.conv_general_dilated, which can
    # fail on some Hyak cuDNN stacks. Patch only this eval process.
    jax_env.poly_mul = _direct_poly_mul

    config = cache["config"]
    # Load the checkpoint's saved config to ensure shapes match training.
    with open(checkpoint_path, "rb") as f:
        ckpt = pickle.load(f)
    ckpt_config: Config = ckpt["config"]
    # Override evaluation-relevant fields from cache where appropriate.
    ckpt_config.max_steps = config.max_steps

    trainer = PPOMCTSJAXTrainer(
        ckpt_config,
        batch_size=batch_size,
        fixed_complexities=ckpt.get("fixed_complexities"),
    )
    trainer.load_checkpoint(checkpoint_path)
    trainer.config.mcts_simulations = mcts_simulations
    trainer.config.temperature_init = 0.0  # near-greedy

    env_config = trainer.env_config
    coeffs = cache["target_coeffs"]
    n = coeffs.shape[0]
    results: List[Dict[str, Any]] = [None] * n

    rng = jax.random.PRNGKey(0)
    cursor = 0
    while cursor < n:
        bsz = min(batch_size, n - cursor)
        # Build target coeffs in JAX-flattened form (already flat in the cache).
        target_arrays = jnp.array(coeffs[cursor:cursor + bsz], dtype=jnp.int32)
        from src.environment.fast_polynomial import FastPoly
        shape = (config.effective_max_degree + 1,) * config.n_variables
        target_polys = [
            FastPoly(np.asarray(coeffs[cursor + i], dtype=np.int64).reshape(shape),
                     config.mod)
            for i in range(bsz)
        ]
        sgc, sga, sgl = trainer._prepare_initial_subgoals(target_polys)
        library_coeffs, library_mask = trainer._export_library_cache()

        states = jax.vmap(
            lambda tc, c, a, lk: env_reset(env_config, tc, c, a, lk)
        )(target_arrays, sgc, sga, sgl)

        episode_rewards = np.zeros(bsz, dtype=np.float64)
        episode_successes = np.zeros(bsz, dtype=bool)
        episode_steps = np.zeros(bsz, dtype=np.int32)
        active = np.ones(bsz, dtype=bool)

        for _ in range(env_config.max_steps):
            if not active.any():
                break
            obs = jax.vmap(
                lambda s: get_observation(env_config, s)
            )(states)
            rng, search_rng = jax.random.split(rng)
            policy_output = trainer._jit_batched_mcts(
                trainer.train_state.params, search_rng, obs, states,
                library_coeffs, library_mask,
            )
            actions = jnp.argmax(policy_output.action_weights, axis=-1)

            step_out = jax.vmap(
                lambda s, a: env_step(env_config, s, a, library_coeffs, library_mask)
            )(states, actions)
            next_states = step_out[0]
            rewards = np.array(step_out[1])
            dones = np.array(step_out[2])
            successes = np.array(step_out[3])

            for i in range(bsz):
                if not active[i]:
                    continue
                episode_rewards[i] += rewards[i]
                episode_steps[i] += 1
                if successes[i]:
                    episode_successes[i] = True
                if dones[i]:
                    active[i] = False
            states = next_states

        for i in range(bsz):
            results[cursor + i] = {
                "target_idx": cursor + i,
                "success": bool(episode_successes[i]),
                "num_steps": int(episode_steps[i]),
                "env_reward": float(episode_rewards[i]),
            }
        cursor += bsz

    return results


CSV_FIELDS = [
    "method", "generated_complexity", "true_min_complexity",
    "num_trials", "success_rate", "avg_steps_success", "avg_steps_all",
    "avg_reward", "wallclock_sec",
]


def main() -> None:
    args = parse_args()
    cache = _load_cache(Path(args.target_cache))
    on_path_cache = _load_on_path_cache(cache, args.on_path_cache_dir)

    print(
        f"Loaded {cache['target_coeffs'].shape[0]} targets across "
        f"{len(np.unique(cache['generated_complexity']))} generated complexities."
    )

    all_rows: List[Dict[str, Any]] = []

    for method in args.methods:
        t0 = time.time()
        if method == "greedy_1step":
            results = _run_greedy(cache)
        elif method.startswith("uniform_mcts"):
            sims = _int_suffix(method, args.mcts_simulations)
            results = _run_uniform_mcts(cache, sims, args.seed)
        elif method.startswith("onpath_mcts"):
            if on_path_cache is None:
                print("ERROR: onpath_mcts requires --on-path-cache-dir.")
                sys.exit(2)
            sims = _int_suffix(method, args.mcts_simulations)
            results = _run_onpath_mcts(cache, on_path_cache, sims, args.seed)
        elif method.startswith("beam_search"):
            beam_width = _int_suffix(method, args.beam_width)
            results = _run_beam_search(cache, beam_width, on_path_cache)
        elif method == "ppo_mcts_checkpoint":
            if args.checkpoint is None:
                print("ERROR: ppo_mcts_checkpoint requires --checkpoint.")
                sys.exit(2)
            results = _run_ppo_checkpoint(
                cache, args.checkpoint, args.ppo_batch_size,
                args.mcts_simulations,
            )
        else:
            print(f"WARNING: unknown method '{method}', skipping.")
            continue
        dt = time.time() - t0
        rows = _aggregate_rows(method, results, cache, dt)
        all_rows.extend(rows)
        # Quick summary to stdout.
        for r in rows:
            if r["true_min_complexity"] == -1:
                print(
                    f"  {r['method']:<28s} C{r['generated_complexity']} "
                    f"SR={r['success_rate']:.1%} "
                    f"avg_steps={r['avg_steps_all']:.1f} "
                    f"avg_reward={r['avg_reward']:+.2f} "
                    f"({r['wallclock_sec']:.1f}s)"
                )

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nWrote {out_csv} ({len(all_rows)} rows)")

    if args.wandb:
        import wandb
        wandb.init(
            project=args.wandb_project or "PolyArithmeticCircuitsRL",
            name=args.wandb_run_name or f"baselines_{out_csv.stem}",
            config={
                "target_cache": args.target_cache,
                "methods": args.methods,
                "mcts_simulations": args.mcts_simulations,
                "beam_width": args.beam_width,
                "checkpoint": args.checkpoint,
            },
        )
        for r in all_rows:
            if r["true_min_complexity"] == -1:
                wandb.log({
                    f"baseline/{r['method']}/C{r['generated_complexity']}/SR":
                        r["success_rate"],
                    f"baseline/{r['method']}/C{r['generated_complexity']}/avg_reward":
                        r["avg_reward"],
                })
        wandb.finish()


if __name__ == "__main__":
    main()
