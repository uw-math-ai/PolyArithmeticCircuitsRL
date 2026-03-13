#!/usr/bin/env python3
"""Evaluate PPO+MCTS (JAX) checkpoints on held-out targets.

Loads saved .pkl checkpoints, runs greedy MCTS evaluation on freshly
sampled targets at specified complexities, and reports success rate,
average reward, and policy entropy.

Outputs:
- Text + CSV summary files
- Optionally logs eval/* metrics to wandb

Example usage:
    python eval_ppo_mcts_jax.py \
        --checkpoint-dir results/ppo-mcts-jax_C6 \
        --iterations 50 100 150 200 \
        --complexities 5 6 \
        --num-trials 1000 \
        --wandb
"""

from __future__ import annotations

import argparse
import os
import pickle
import random
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import mctx
import numpy as np
from tqdm import tqdm

# Ensure project root is importable when script is run directly.
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.algorithms.ppo_mcts_jax import PPOMCTSJAXTrainer
from src.config import Config
from src.environment.fast_polynomial import FastPoly
from src.game_board.generator import generate_random_circuit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate PPO+MCTS JAX checkpoints on held-out targets."
    )
    parser.add_argument(
        "--checkpoint-dir", type=str, required=True,
        help="Directory containing checkpoint_XXXXX.pkl files.",
    )
    parser.add_argument(
        "--iterations", type=int, nargs="+", default=[50, 100, 150, 200],
        help="Checkpoint iterations to evaluate (default: 50 100 150 200).",
    )
    parser.add_argument(
        "--complexities", type=int, nargs="+", default=[5, 6],
        help="Target complexities to evaluate (default: 5 6).",
    )
    parser.add_argument(
        "--num-trials", type=int, default=1000,
        help="Episodes per complexity per checkpoint (default: 1000).",
    )
    parser.add_argument(
        "--mcts-simulations", type=int, default=16,
        help="MCTS simulations per action during eval (default: 16).",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.1,
        help="MCTS temperature for eval (low = greedy, default: 0.1).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=256,
        help="Batch size for parallel eval episodes (default: 256).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for target sampling.",
    )
    parser.add_argument(
        "--out-prefix", type=str, default=None,
        help="Output file prefix (writes .txt and .csv). "
             "Defaults to {checkpoint-dir}/eval_results.",
    )
    parser.add_argument(
        "--wandb", action="store_true",
        help="Log eval metrics to wandb.",
    )
    parser.add_argument(
        "--wandb-project", type=str, default="PolyArithmeticCircuitsRL",
        help="W&B project name.",
    )
    parser.add_argument(
        "--wandb-entity", type=str, default=None,
        help="W&B entity (team or user).",
    )
    parser.add_argument(
        "--wandb-run-name", type=str, default=None,
        help="W&B run name (default: auto-generated).",
    )
    return parser.parse_args()


def evaluate_checkpoint(
    trainer: PPOMCTSJAXTrainer,
    complexities: list[int],
    num_trials: int,
    batch_size: int,
    mcts_simulations: int,
    temperature: float,
) -> dict:
    """Run eval episodes and return per-complexity metrics.

    Args:
        trainer: Trainer with loaded checkpoint params.
        complexities: List of target complexities to evaluate.
        num_trials: Episodes per complexity.
        batch_size: Parallel eval batch size.
        mcts_simulations: MCTS simulations per action.
        temperature: MCTS temperature (low = near-greedy).

    Returns:
        Dict mapping complexity -> {success_rate_pct, avg_reward, avg_entropy}.
    """
    from src.algorithms.jax_env import (
        reset as env_reset, step as env_step,
        get_observation, get_valid_actions_mask,
    )

    env_config = trainer.env_config
    config = trainer.config
    params = trainer.train_state.params
    network = trainer.network

    # Override MCTS settings for eval.
    orig_sims = config.mcts_simulations
    orig_temp = config.temperature_init
    config.mcts_simulations = mcts_simulations
    config.temperature_init = temperature

    results = {}

    for c in complexities:
        successes = 0
        reward_sum = 0.0
        entropy_sum = 0.0
        entropy_count = 0

        num_done = 0
        pbar = tqdm(total=num_trials, desc=f"  C{c}", unit="ep")

        while num_done < num_trials:
            B = min(batch_size, num_trials - num_done)
            rng = jax.random.PRNGKey(np.random.randint(0, 2**31))

            # Sample targets.
            targets = []
            for _ in range(B):
                poly, _ = generate_random_circuit(config, c)
                targets.append(
                    jnp.array(poly.coeffs.flatten(), dtype=jnp.int32)
                )
            target_arrays = jnp.stack(targets, axis=0)

            # Reset envs.
            states = jax.vmap(
                lambda tc: env_reset(env_config, tc)
            )(target_arrays)

            episode_rewards = np.zeros(B)
            episode_successes = np.zeros(B, dtype=bool)
            active = np.ones(B, dtype=bool)

            for step_idx in range(config.max_steps):
                if not active.any():
                    break

                obs_batch = jax.vmap(
                    lambda s: get_observation(env_config, s)
                )(states)

                # Run MCTS.
                rng, search_rng = jax.random.split(rng)
                policy_output = trainer._jit_batched_mcts(
                    params, search_rng, obs_batch, states,
                )

                # Near-greedy: argmax of MCTS visit counts.
                actions = jnp.argmax(policy_output.action_weights, axis=-1)

                # Compute entropy of the network policy for logging.
                logits, _ = jax.vmap(
                    lambda obs: network.apply(params, obs)
                )(obs_batch)
                log_probs = jax.nn.log_softmax(logits)
                probs = jax.nn.softmax(logits)
                entropy_batch = -(probs * log_probs).sum(axis=-1)
                entropy_np = np.array(entropy_batch)

                for i in range(B):
                    if active[i]:
                        entropy_sum += float(entropy_np[i])
                        entropy_count += 1

                # Step envs.
                next_states, rewards, dones, successes_batch = jax.vmap(
                    lambda s, a: env_step(env_config, s, a)
                )(states, actions)

                rewards_np = np.array(rewards)
                dones_np = np.array(dones)
                successes_np = np.array(successes_batch)

                for i in range(B):
                    if active[i]:
                        episode_rewards[i] += rewards_np[i]
                        if successes_np[i]:
                            episode_successes[i] = True
                        if dones_np[i]:
                            active[i] = False

                states = next_states

            successes += int(episode_successes.sum())
            reward_sum += float(episode_rewards.sum())
            num_done += B
            pbar.update(B)

        pbar.close()

        results[c] = {
            "success_rate_pct": 100.0 * successes / num_trials,
            "avg_reward": reward_sum / num_trials,
            "avg_entropy": entropy_sum / max(entropy_count, 1),
        }

    # Restore original settings.
    config.mcts_simulations = orig_sims
    config.temperature_init = orig_temp

    return results


def main() -> None:
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    iterations = sorted(set(args.iterations))
    complexities = sorted(set(args.complexities))

    # Load first checkpoint to get config.
    first_ckpt = os.path.join(
        args.checkpoint_dir, f"checkpoint_{iterations[0]:05d}.pkl"
    )
    if not os.path.exists(first_ckpt):
        print(f"Checkpoint not found: {first_ckpt}")
        sys.exit(1)

    with open(first_ckpt, "rb") as f:
        first_state = pickle.load(f)
    config: Config = first_state["config"]
    fixed_c = first_state.get("fixed_complexities")

    # Build trainer once (network + JIT compilation reused across checkpoints).
    trainer = PPOMCTSJAXTrainer(
        config,
        batch_size=args.batch_size,
        fixed_complexities=fixed_c,
    )

    if args.wandb:
        import wandb
        run_name = args.wandb_run_name or f"eval_{Path(args.checkpoint_dir).name}"
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            config={
                "eval_complexities": complexities,
                "num_trials": args.num_trials,
                "mcts_simulations": args.mcts_simulations,
                "temperature": args.temperature,
                "n_variables": config.n_variables,
                "mod": config.mod,
                **{f"train_{k}": v for k, v in vars(config).items()
                   if isinstance(v, (int, float, bool, str))},
            },
        )

    all_rows = []

    for it in iterations:
        ckpt_path = os.path.join(
            args.checkpoint_dir, f"checkpoint_{it:05d}.pkl"
        )
        if not os.path.exists(ckpt_path):
            print(f"WARNING: Checkpoint not found for iter {it}: {ckpt_path}, skipping.")
            continue

        print(f"\n=== Evaluating iteration {it} ===")
        trainer.load_checkpoint(ckpt_path)

        t0 = time.time()
        metrics = evaluate_checkpoint(
            trainer=trainer,
            complexities=complexities,
            num_trials=args.num_trials,
            batch_size=args.batch_size,
            mcts_simulations=args.mcts_simulations,
            temperature=args.temperature,
        )
        elapsed = time.time() - t0

        row = {"iter": it}
        log_dict = {"eval/iteration": it}
        parts = []
        for c in complexities:
            m = metrics.get(c, {
                "success_rate_pct": float("nan"),
                "avg_reward": float("nan"),
                "avg_entropy": float("nan"),
            })
            row[f"c{c}_success_rate_pct"] = m["success_rate_pct"]
            row[f"c{c}_avg_reward"] = m["avg_reward"]
            row[f"c{c}_avg_entropy"] = m["avg_entropy"]
            log_dict[f"eval/success_rate_C{c}"] = m["success_rate_pct"] / 100.0
            log_dict[f"eval/avg_reward_C{c}"] = m["avg_reward"]
            log_dict[f"eval/avg_entropy_C{c}"] = m["avg_entropy"]
            parts.append(
                f"C{c} SR={m['success_rate_pct']:.2f}% "
                f"R={m['avg_reward']:.2f} H={m['avg_entropy']:.2f}"
            )

        all_rows.append(row)
        print(f"Iter {it} ({elapsed:.1f}s): " + " | ".join(parts))

        if args.wandb:
            import wandb
            wandb.log(log_dict, step=it)

    # Write output files.
    out_prefix = args.out_prefix or os.path.join(
        args.checkpoint_dir, "eval_results"
    )
    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    out_txt = out_prefix.with_suffix(".txt")
    out_csv = out_prefix.with_suffix(".csv")

    # Build header from complexities.
    header_parts = ["Iter"]
    for c in complexities:
        header_parts.extend([
            f"C{c} Success Rate (%)", f"C{c} Avg Reward", f"C{c} Entropy"
        ])
    header = " | ".join(header_parts)
    sep = "-" * len(header)

    lines = [header, sep]
    for r in all_rows:
        row_parts = [f"{r['iter']}"]
        for c in complexities:
            row_parts.extend([
                f"{r[f'c{c}_success_rate_pct']:.2f}",
                f"{r[f'c{c}_avg_reward']:.2f}",
                f"{r[f'c{c}_avg_entropy']:.2f}",
            ])
        lines.append(" | ".join(row_parts))
    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

    csv_header_parts = ["iter"]
    for c in complexities:
        csv_header_parts.extend([
            f"c{c}_success_rate_pct", f"c{c}_avg_reward", f"c{c}_avg_entropy"
        ])
    csv_lines = [",".join(csv_header_parts)]
    for r in all_rows:
        vals = [str(r["iter"])]
        for c in complexities:
            vals.extend([
                f"{r[f'c{c}_success_rate_pct']:.2f}",
                f"{r[f'c{c}_avg_reward']:.2f}",
                f"{r[f'c{c}_avg_entropy']:.2f}",
            ])
        csv_lines.append(",".join(vals))
    out_csv.write_text("\n".join(csv_lines) + "\n", encoding="utf-8")

    print(f"\nSaved {out_txt}")
    print(f"Saved {out_csv}")

    if args.wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
