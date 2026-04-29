"""Supervised oracle diagnostic — train the policy network with cross-entropy
against shortest-path oracle labels, with no MCTS and no PPO.

This answers a single question: can the current PolicyValueNet architecture
fit the optimal-action distribution at all? If yes (high top-1 accuracy on
held-out targets), then the bottleneck in RL training is the bootstrap, not
the architecture. If no, the architecture/representation is the bottleneck
and needs to be improved before further RL work.

Usage:
    python -u scripts/supervised_oracle_diagnostic.py \\
      --cache-dir on_path_cache/c1_c5 \\
      --complexities 2 3 \\
      --n-vars 2 --mod 5 --max-degree 6 \\
      --max-build-complexity 8 --build-complexity-slack 3 --max-steps 8 \\
      --max-targets-per-complexity 256 \\
      --steps 10000 --batch-size 256 --lr 3e-4 \\
      --results-dir results/supervised_oracle_c2c3
"""

from __future__ import annotations

import argparse
import functools
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state

from src.config import Config
from src.algorithms.jax_env import (
    EnvConfig, make_env_config,
    reset as env_reset,
    step as env_step,
    get_observation,
)
from src.algorithms.jax_net import (
    PolicyValueNet, create_network, init_params,
)
from src.algorithms.oracle_labels import (
    oracle_action_scores, oracle_action_distribution,
)
from src.game_board.on_path import OnPathCache, OnPathTargetContext


# ---------------------------------------------------------------------------
# Args & config
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--complexities", type=int, nargs="+", default=[2, 3])
    p.add_argument("--n-vars", type=int, default=2)
    p.add_argument("--mod", type=int, default=5)
    p.add_argument("--max-degree", type=int, default=6)
    p.add_argument("--max-build-complexity", type=int, default=8)
    p.add_argument("--build-complexity-slack", type=int, default=3)
    p.add_argument("--max-steps", type=int, default=8)
    p.add_argument("--on-path-max-size", type=int, default=4096)
    p.add_argument("--on-path-num-routes", type=int, default=32)
    p.add_argument("--on-path-phi-mode", default="depth_weighted",
                   choices=["count", "max_step", "depth_weighted"])
    p.add_argument("--on-path-route-consistency-mode", default="best_route_phi",
                   choices=["best_route_phi", "lock_on_first_hit", "off"])
    p.add_argument("--on-path-depth-weight-power", type=float, default=1.0)
    p.add_argument("--max-targets-per-complexity", type=int, default=256)
    p.add_argument("--max-rollouts-per-target", type=int, default=4,
                   help="Re-roll each target this many times to cover branching "
                        "oracle actions (the oracle picks one action uniformly "
                        "at each step, so multiple rollouts cover more states).")
    p.add_argument("--train-fraction", type=float, default=0.8)
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-interval", type=int, default=200)
    p.add_argument("--results-dir", default="results/supervised_oracle")
    p.add_argument("--save-checkpoint", action="store_true",
                   help="Save final params to <results-dir>/checkpoint.pkl")
    return p.parse_args()


def build_config(args: argparse.Namespace) -> Config:
    cfg = Config()
    cfg.n_variables = args.n_vars
    cfg.mod = args.mod
    cfg.max_degree = args.max_degree
    cfg.max_complexity = max(args.complexities)
    cfg.max_build_complexity = args.max_build_complexity
    cfg.build_complexity_slack = args.build_complexity_slack
    cfg.max_steps = args.max_steps
    cfg.on_path_max_size = args.on_path_max_size
    cfg.on_path_num_routes = args.on_path_num_routes
    cfg.on_path_phi_mode = args.on_path_phi_mode
    cfg.on_path_route_consistency_mode = args.on_path_route_consistency_mode
    cfg.on_path_depth_weight_power = args.on_path_depth_weight_power
    cfg.reward_mode = "clean_onpath"
    cfg.factor_library_enabled = False
    cfg.curriculum_enabled = False
    cfg.seed = args.seed
    return cfg


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------

def _fastpoly_to_jax(poly, target_size: int) -> jnp.ndarray:
    return jnp.array(poly.coeffs.flatten(), dtype=jnp.int32)


def _pack_context(
    ctx: OnPathTargetContext, max_size: int, target_size: int
) -> Tuple[jnp.ndarray, ...]:
    """Pack a single OnPathTargetContext into JAX arrays for env_reset."""
    coeffs = np.zeros((max_size, target_size), dtype=np.int32)
    hashes = np.zeros((max_size,), dtype=np.uint32)
    steps = np.zeros((max_size,), dtype=np.int32)
    route_masks = np.zeros((max_size,), dtype=np.uint32)
    active = np.zeros((max_size,), dtype=bool)
    n = len(ctx.on_path_ids)
    coeffs[:n] = ctx.on_path_coeffs.astype(np.int32, copy=False)
    hashes[:n] = ctx.on_path_hashes
    steps[:n] = ctx.on_path_steps
    route_masks[:n] = ctx.on_path_route_masks
    active[:n] = True
    return (
        jnp.array(coeffs),
        jnp.array(hashes),
        jnp.array(steps),
        jnp.array(route_masks),
        jnp.array(active),
        jnp.int32(ctx.target_board_step),
    )


def collect_dataset(
    env_config: EnvConfig,
    cache: OnPathCache,
    complexities: List[int],
    max_targets: int,
    max_rollouts_per_target: int,
    train_fraction: float,
    rng_seed: int,
) -> Dict[str, np.ndarray]:
    """Roll out oracle-driven trajectories and collect (obs, label, complexity).

    Returns a dict of stacked numpy arrays. Observations are flattened
    into individual keys with a leading batch dimension.
    """
    rng = np.random.default_rng(rng_seed)
    key = jax.random.PRNGKey(rng_seed)

    # JIT closures over env_config (it contains JAX arrays so it must be a closure).
    @jax.jit
    def jit_reset(target_arr, sg_c, sg_a, sg_l, op_c, op_h, op_s, op_r, op_a, tbs):
        return env_reset(env_config, target_arr, sg_c, sg_a, sg_l,
                         op_c, op_h, op_s, op_r, op_a, tbs)

    @jax.jit
    def jit_step(state, action):
        library_coeffs = jnp.zeros((1, env_config.target_size), dtype=jnp.int32)
        library_mask = jnp.zeros((1,), dtype=jnp.bool_)
        return env_step(env_config, state, action, library_coeffs, library_mask)

    @jax.jit
    def jit_obs(state):
        return get_observation(env_config, state)

    @jax.jit
    def jit_oracle(state):
        return oracle_action_distribution(env_config, state)

    obs_examples: List[Dict[str, np.ndarray]] = []
    label_examples: List[np.ndarray] = []
    complexity_labels: List[int] = []
    is_train_flags: List[bool] = []

    target_size = env_config.target_size
    max_subgoals = env_config.max_subgoals
    on_path_max = env_config.on_path_max_size

    empty_sg_coeffs = jnp.zeros((max_subgoals, target_size), dtype=jnp.int32)
    empty_sg_active = jnp.zeros((max_subgoals,), dtype=jnp.bool_)
    empty_sg_known = jnp.zeros((max_subgoals,), dtype=jnp.bool_)

    n_skipped_no_label = 0
    n_oracle_action_set_sizes: List[int] = []

    for complexity in complexities:
        comp_cache = cache.by_complexity[int(complexity)]
        train_ids = comp_cache.train_target_ids
        val_ids = comp_cache.val_target_ids
        # Subsample if we have more than max_targets.
        train_take = min(max_targets, len(train_ids))
        val_take = min(max(8, max_targets // 8), len(val_ids))
        train_pick = rng.choice(train_ids, size=train_take, replace=False)
        val_pick = rng.choice(val_ids, size=val_take, replace=False)

        for split_name, target_ids, is_train in (
            ("train", train_pick, True),
            ("val", val_pick, False),
        ):
            for target_id in target_ids:
                ctx = comp_cache.target_context(int(target_id))
                target_arr = jnp.array(
                    ctx.target_poly.coeffs.flatten().astype(np.int32),
                    dtype=jnp.int32,
                )
                op_c, op_h, op_s, op_r, op_a, tbs = _pack_context(
                    ctx, on_path_max, target_size
                )

                for _rollout in range(max_rollouts_per_target):
                    state = jit_reset(
                        target_arr,
                        empty_sg_coeffs, empty_sg_active, empty_sg_known,
                        op_c, op_h, op_s, op_r, op_a, tbs,
                    )

                    for _step in range(env_config.max_steps):
                        if bool(state.done):
                            break
                        oracle_dist, has_label = jit_oracle(state)
                        if not bool(has_label):
                            n_skipped_no_label += 1
                            break

                        obs = jit_obs(state)
                        # Convert to numpy so we can stack across heterogeneous
                        # episode lengths.
                        obs_np = jax.tree.map(lambda x: np.asarray(x), obs)
                        oracle_dist_np = np.asarray(oracle_dist, dtype=np.float32)
                        obs_examples.append(obs_np)
                        label_examples.append(oracle_dist_np)
                        complexity_labels.append(int(complexity))
                        is_train_flags.append(is_train)
                        n_oracle_action_set_sizes.append(int((oracle_dist_np > 0).sum()))

                        # Sample one oracle action and step.
                        oracle_probs = oracle_dist_np
                        action = int(rng.choice(
                            len(oracle_probs), p=oracle_probs / oracle_probs.sum()
                        ))
                        out = jit_step(state, jnp.int32(action))
                        state = out[0]

    # Stack into batched arrays.
    n = len(obs_examples)
    if n == 0:
        raise RuntimeError("No (obs, oracle_dist) pairs collected.")
    keys = list(obs_examples[0].keys())
    stacked_obs = {
        k: np.stack([ex[k] for ex in obs_examples], axis=0) for k in keys
    }
    stacked_labels = np.stack(label_examples, axis=0)
    complexity_arr = np.array(complexity_labels, dtype=np.int32)
    is_train_arr = np.array(is_train_flags, dtype=np.bool_)

    print(f"[dataset] collected {n} (obs, label) pairs", flush=True)
    print(
        f"[dataset] train={int(is_train_arr.sum())} "
        f"val={int((~is_train_arr).sum())} "
        f"skipped_no_label={n_skipped_no_label}",
        flush=True,
    )
    if n_oracle_action_set_sizes:
        sizes = np.array(n_oracle_action_set_sizes)
        print(
            f"[dataset] |oracle_action_set| mean={sizes.mean():.2f} "
            f"min={sizes.min()} max={sizes.max()} "
            f"median={int(np.median(sizes))}",
            flush=True,
        )
    for c in complexities:
        mask_c = complexity_arr == c
        print(
            f"[dataset] C{c}: total={int(mask_c.sum())} "
            f"train={int((mask_c & is_train_arr).sum())} "
            f"val={int((mask_c & ~is_train_arr).sum())}",
            flush=True,
        )

    return {
        "obs": stacked_obs,
        "labels": stacked_labels,
        "complexity": complexity_arr,
        "is_train": is_train_arr,
    }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def make_train_state(network: PolicyValueNet, env_config: EnvConfig,
                     lr: float, max_grad_norm: float, seed: int):
    rng = jax.random.PRNGKey(seed)
    params = init_params(network, env_config, rng)
    tx = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adam(lr),
    )
    return train_state.TrainState.create(
        apply_fn=network.apply, params=params, tx=tx,
    )


def make_step_fn(network: PolicyValueNet):
    @jax.jit
    def supervised_step(state, batch_obs, batch_labels):
        def loss_fn(params):
            logits, _values = jax.vmap(
                lambda o: network.apply(params, o)
            )(batch_obs)
            log_probs = jax.nn.log_softmax(logits)
            ce = -(batch_labels * log_probs).sum(axis=-1).mean()
            # Top-1: argmax of logits matches an oracle action (label > 0).
            argmax_a = jnp.argmax(logits, axis=-1)
            top1_in_set = jnp.take_along_axis(
                batch_labels, argmax_a[:, None], axis=-1
            ).squeeze(-1) > 0.0
            top1_acc = top1_in_set.astype(jnp.float32).mean()
            # KL(oracle || network)
            kl_terms = jnp.where(
                batch_labels > 0,
                batch_labels * (jnp.log(batch_labels + 1e-12) - log_probs),
                0.0,
            )
            kl = kl_terms.sum(axis=-1).mean()
            return ce, (top1_acc, kl)

        (ce, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        state = state.apply_gradients(grads=grads)
        return state, ce, aux[0], aux[1]

    return supervised_step


def make_eval_fn(network: PolicyValueNet):
    @jax.jit
    def supervised_eval(params, batch_obs, batch_labels):
        logits, _values = jax.vmap(lambda o: network.apply(params, o))(batch_obs)
        log_probs = jax.nn.log_softmax(logits)
        ce = -(batch_labels * log_probs).sum(axis=-1).mean()
        argmax_a = jnp.argmax(logits, axis=-1)
        top1_in_set = jnp.take_along_axis(
            batch_labels, argmax_a[:, None], axis=-1
        ).squeeze(-1) > 0.0
        kl_terms = jnp.where(
            batch_labels > 0,
            batch_labels * (jnp.log(batch_labels + 1e-12) - log_probs),
            0.0,
        )
        kl = kl_terms.sum(axis=-1).mean()
        return ce, top1_in_set, kl
    return supervised_eval


def slice_dataset(dataset: dict, mask: np.ndarray) -> dict:
    return {
        "obs": {k: v[mask] for k, v in dataset["obs"].items()},
        "labels": dataset["labels"][mask],
        "complexity": dataset["complexity"][mask],
    }


def evaluate(
    eval_fn,
    params,
    eval_data: dict,
    complexities: List[int],
    batch_size: int,
) -> Dict[str, float]:
    """Run eval over the full validation set, return per-complexity metrics."""
    n = len(eval_data["labels"])
    if n == 0:
        return {}
    all_top1: List[np.ndarray] = []
    all_ce: List[float] = []
    all_kl: List[float] = []
    weights: List[int] = []

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_obs = {k: jnp.asarray(v[start:end]) for k, v in eval_data["obs"].items()}
        batch_labels = jnp.asarray(eval_data["labels"][start:end])
        ce, top1_in_set, kl = eval_fn(params, batch_obs, batch_labels)
        all_top1.append(np.asarray(top1_in_set))
        all_ce.append(float(ce))
        all_kl.append(float(kl))
        weights.append(end - start)

    weights_arr = np.array(weights, dtype=np.float32)
    top1_arr = np.concatenate(all_top1)
    overall = {
        "ce": float(np.average(all_ce, weights=weights_arr)),
        "top1_in_set": float(top1_arr.mean()),
        "kl": float(np.average(all_kl, weights=weights_arr)),
    }
    by_c: Dict[str, float] = {}
    for c in complexities:
        mask_c = eval_data["complexity"] == c
        if mask_c.sum() == 0:
            continue
        by_c[f"top1_C{c}"] = float(top1_arr[mask_c].mean())
    return {**overall, **by_c}


def main():
    args = parse_args()
    cfg = build_config(args)
    env_config = make_env_config(cfg)
    print(
        f"[config] n_vars={cfg.n_variables} mod={cfg.mod} "
        f"max_degree={cfg.effective_max_degree} target_size={cfg.target_size} "
        f"max_nodes={cfg.max_nodes} max_actions={cfg.max_actions} "
        f"max_steps={cfg.max_steps}",
        flush=True,
    )

    cache = OnPathCache.load(args.cache_dir, cfg, args.complexities)
    print(f"[cache] loaded {args.cache_dir} for C{args.complexities}", flush=True)

    t0 = time.time()
    dataset = collect_dataset(
        env_config, cache, args.complexities,
        max_targets=args.max_targets_per_complexity,
        max_rollouts_per_target=args.max_rollouts_per_target,
        train_fraction=args.train_fraction,
        rng_seed=args.seed,
    )
    print(f"[dataset] built in {time.time() - t0:.1f}s", flush=True)

    train_data = slice_dataset(dataset, dataset["is_train"])
    val_data = slice_dataset(dataset, ~dataset["is_train"])

    network = create_network(cfg)
    state = make_train_state(network, env_config,
                             lr=args.lr, max_grad_norm=args.max_grad_norm,
                             seed=args.seed)
    step_fn = make_step_fn(network)
    eval_fn = make_eval_fn(network)

    n_train = len(train_data["labels"])
    print(f"[train] start: {n_train} train examples, {args.steps} steps, "
          f"batch={args.batch_size}", flush=True)

    rng = np.random.default_rng(args.seed + 1)
    os.makedirs(args.results_dir, exist_ok=True)

    best_val_top1 = -1.0
    history = []

    for step_idx in range(1, args.steps + 1):
        idx = rng.integers(0, n_train, size=args.batch_size)
        batch_obs = {k: jnp.asarray(v[idx]) for k, v in train_data["obs"].items()}
        batch_labels = jnp.asarray(train_data["labels"][idx])
        state, ce, top1, kl = step_fn(state, batch_obs, batch_labels)

        if step_idx % args.eval_interval == 0 or step_idx == args.steps:
            val_metrics = evaluate(
                eval_fn, state.params, val_data, args.complexities,
                batch_size=args.batch_size,
            )
            line = (
                f"[step {step_idx:>5}] "
                f"train_ce={float(ce):.4f} train_top1={float(top1):.3f} "
                f"train_kl={float(kl):.3f} "
                f"val_ce={val_metrics.get('ce', 0):.4f} "
                f"val_top1={val_metrics.get('top1_in_set', 0):.3f} "
                f"val_kl={val_metrics.get('kl', 0):.3f}"
            )
            for c in args.complexities:
                k = f"top1_C{c}"
                if k in val_metrics:
                    line += f" {k}={val_metrics[k]:.3f}"
            print(line, flush=True)

            history.append({
                "step": step_idx,
                "train_ce": float(ce),
                "train_top1": float(top1),
                "train_kl": float(kl),
                **val_metrics,
            })
            cur = val_metrics.get("top1_in_set", 0.0)
            if cur > best_val_top1:
                best_val_top1 = cur

    # Final summary.
    print("\n=== FINAL ===", flush=True)
    final = evaluate(
        eval_fn, state.params, val_data, args.complexities,
        batch_size=args.batch_size,
    )
    print(f"val_ce={final.get('ce', 0):.4f}", flush=True)
    print(f"val_top1_in_set={final.get('top1_in_set', 0):.3f}", flush=True)
    print(f"val_kl={final.get('kl', 0):.3f}", flush=True)
    for c in args.complexities:
        k = f"top1_C{c}"
        if k in final:
            print(f"val_top1_C{c}={final[k]:.3f}", flush=True)
    print(f"best_val_top1_in_set_so_far={best_val_top1:.3f}", flush=True)

    history_path = Path(args.results_dir) / "history.pkl"
    with open(history_path, "wb") as f:
        pickle.dump({"history": history, "args": vars(args), "final": final}, f)
    print(f"[saved] history -> {history_path}", flush=True)

    if args.save_checkpoint:
        ckpt_path = Path(args.results_dir) / "checkpoint.pkl"
        with open(ckpt_path, "wb") as f:
            pickle.dump({"params": state.params, "config": cfg}, f)
        print(f"[saved] params -> {ckpt_path}", flush=True)


if __name__ == "__main__":
    main()
