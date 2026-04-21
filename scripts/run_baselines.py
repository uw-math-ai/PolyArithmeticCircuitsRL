#!/usr/bin/env python3
"""Evaluate symbolic/search baselines on the same target distribution the RL agent sees.

Usage:
    python scripts/run_baselines.py --n_vars 2 --max_ops 6 --num_targets 500 \
        --out runs/baselines.json
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from poly_circuit_rl.config import Config
from poly_circuit_rl.env.samplers import FrozenSplitSampler, RandomCircuitSampler
from poly_circuit_rl.baselines.exhaustive import ExhaustiveSearch
from poly_circuit_rl.baselines.greedy import GreedyBaseline
from poly_circuit_rl.baselines.factorization import FactorizationBaseline
from poly_circuit_rl.baselines.horner import HornerBaseline
from poly_circuit_rl.baselines.memoized import MemoizedCSEBaseline


def sample_targets_from_split(jsonl_path: str, n_vars: int, max_ops: int):
    """Use every target in the held-out eval split whose optimal length fits."""
    sampler = FrozenSplitSampler(jsonl_path, n_vars=n_vars)
    targets = []
    for length, entries in sampler.by_length.items():
        if length <= max_ops:
            targets.extend(poly for poly, _ in entries)
    return targets


def sample_targets_random(config: Config, num_targets: int, seed: int):
    rng = random.Random(seed)
    sampler = RandomCircuitSampler(n_vars=config.n_vars, max_steps=config.max_ops)
    targets, seen = [], set()
    attempts = 0
    while len(targets) < num_targets and attempts < num_targets * 10:
        poly, _ = sampler.sample(rng)
        key = tuple(sorted(poly.items()))
        if key not in seen:
            seen.add(key)
            targets.append(poly)
        attempts += 1
    return targets


def run_one(name, baseline, targets, max_ops):
    t0 = time.time()
    stats = baseline.evaluate_batch(targets, max_ops=max_ops)
    stats["name"] = name
    stats["wall_sec"] = round(time.time() - t0, 2)
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_vars", type=int, default=2)
    parser.add_argument("--max_ops", type=int, default=6)
    parser.add_argument("--num_targets", type=int, default=500,
                        help="Only used when --eval_jsonl is not given.")
    parser.add_argument("--eval_jsonl", type=str, default=None,
                        help="Evaluate on the held-out split (recommended).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_exhaustive", action="store_true",
                        help="Exhaustive BFS gets expensive past max_ops=4.")
    parser.add_argument("--out", type=str, default="runs/baselines.json")
    args = parser.parse_args()

    if args.max_ops > 4 and not args.skip_exhaustive:
        print("max_ops > 4 detected; auto-enabling --skip_exhaustive.")
        args.skip_exhaustive = True

    config = Config(n_vars=args.n_vars, max_ops=args.max_ops)
    if args.eval_jsonl:
        print(f"Loading held-out targets from {args.eval_jsonl}")
        targets = sample_targets_from_split(args.eval_jsonl, args.n_vars, args.max_ops)
    else:
        print(f"Sampling {args.num_targets} random targets")
        targets = sample_targets_random(config, args.num_targets, args.seed)
    print(f"  got {len(targets)} unique targets")

    results = []

    if not args.skip_exhaustive:
        print("Running ExhaustiveSearch...")
        ex = ExhaustiveSearch(config)
        t0 = time.time()
        ex.build(args.max_ops)
        build_sec = round(time.time() - t0, 2)
        optimals = [ex.find_optimal(t) for t in targets]
        reachable = [o for o in optimals if o is not None]
        results.append({
            "name": "exhaustive",
            "build_sec": build_sec,
            "reachable_count": ex.reachable_count(),
            "targets_reachable": len(reachable),
            "avg_optimal_ops": sum(reachable) / max(len(reachable), 1),
        })

    for name, cls in [
        ("greedy", GreedyBaseline),
        ("factorization", FactorizationBaseline),
        ("horner", HornerBaseline),
        ("memoized_cse", MemoizedCSEBaseline),
    ]:
        print(f"Running {name}...")
        baseline = cls(config)
        try:
            results.append(run_one(name, baseline, targets, args.max_ops))
        except Exception as e:
            results.append({"name": name, "error": str(e)})

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "config": {"n_vars": args.n_vars, "max_ops": args.max_ops,
                       "num_targets": len(targets), "seed": args.seed},
            "results": results,
        }, f, indent=2)

    print("\n=== Baseline summary ===")
    for r in results:
        print(json.dumps(r, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
