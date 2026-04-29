#!/usr/bin/env python3
"""Build a deterministic, shared target cache for baseline comparisons.

Samples ``N`` random-circuit targets per generated complexity, caches their
flat coefficient arrays + canonical keys, and (optionally) joins per-target
``true_min_complexity`` from a precomputed histogram.

For this branch's on-path baselines, prefer ``--on-path-cache-dir``.  That
samples exact cached on-path targets and stores ``target_ids`` so reward-shaped
MCTS/beam search can reconstruct ``OnPathTargetContext`` later.

Usage:

    python scripts/build_baseline_target_cache.py \\
        --complexities 5 6 7 8 \\
        --num-trials 50 \\
        --seed 42 \\
        --out results/baselines/target_cache.npz \\
        --true-complexity-json results/random_circuit_true_complexity.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Config  # noqa: E402
from src.environment.fast_polynomial import FastPoly  # noqa: E402
from src.game_board.generator import generate_random_circuit  # noqa: E402
from src.game_board.on_path import OnPathCache  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--complexities", type=int, nargs="+", default=[5, 6, 7, 8])
    p.add_argument("--num-trials", type=int, default=50,
                   help="Targets sampled per complexity (deduplicated). v1 uses 50.")
    p.add_argument("--n-variables", type=int, default=2)
    p.add_argument("--mod", type=int, default=5)
    p.add_argument("--max-degree", type=int, default=6)
    p.add_argument("--max-steps", type=int, default=14)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--on-path-cache-dir",
        type=str,
        default=None,
        help="Sample targets from this OnPathCache directory instead of random circuits.",
    )
    p.add_argument(
        "--split",
        choices=["train", "val", "test", "all"],
        default="test",
        help="OnPathCache split to sample when --on-path-cache-dir is set.",
    )
    p.add_argument("--out", required=True,
                   help="Output .npz path (e.g., results/baselines/target_cache.npz).")
    p.add_argument(
        "--true-complexity-json",
        type=str,
        default=None,
        help="Optional JSON output of check_random_circuit_true_complexity.py. "
             "Used to join true_min_complexity per canonical_key. If omitted, "
             "all targets get true_min_complexity=-1 (= unknown).",
    )
    return p.parse_args()


def make_config(args: argparse.Namespace) -> Config:
    return Config(
        n_variables=args.n_variables,
        mod=args.mod,
        max_complexity=max(args.complexities),
        max_degree=args.max_degree,
        max_steps=args.max_steps,
    )


def load_true_complexity_lookup(json_path: Optional[str]) -> Dict[bytes, int]:
    """Load (canonical_key -> true_min_complexity) from histogram JSON if any.

    The check_random_circuit_true_complexity.py JSON stores per-label histograms
    but not per-key complexities. We derive the lookup by re-sampling the same
    seed/labels/samples from that JSON's ``args`` block and re-running the route
    index. To keep this script lightweight, we instead expose only the histogram
    summary; a per-key lookup requires running the full enumeration.

    For v1 we accept this and return an empty dict if no per-key data is
    provided. The CSV will report ``true_min_complexity=-1`` which the analyst
    can join later via a separate script.
    """
    if json_path is None:
        return {}
    json_path = Path(json_path)
    if not json_path.exists():
        print(f"WARNING: {json_path} not found; no true_min_complexity join.")
        return {}

    with json_path.open("r") as f:
        data = json.load(f)

    # The stock histogram JSON does not include per-key min costs. If a future
    # version adds a "min_cost_by_key" mapping (hex -> cost), we use it.
    raw = data.get("min_cost_by_key", {})
    if not raw:
        print(
            f"INFO: {json_path} has no per-key 'min_cost_by_key' field; "
            "true_min_complexity will be -1 in the cache. Re-run the histogram "
            "with --emit-min-cost-by-key (if available) to populate it."
        )
        return {}

    return {bytes.fromhex(k): int(v) for k, v in raw.items()}


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    config = make_config(args)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    true_lookup = load_true_complexity_lookup(args.true_complexity_json)

    target_size = config.target_size

    all_coeffs: List[np.ndarray] = []
    all_canonical_keys: List[bytes] = []
    all_generated_complexity: List[int] = []
    all_true_complexity: List[int] = []
    all_idx_within_complexity: List[int] = []
    all_target_ids: List[int] = []
    all_target_board_steps: List[int] = []

    rng = np.random.default_rng(args.seed)

    if args.on_path_cache_dir is not None:
        cache = OnPathCache.load(Path(args.on_path_cache_dir), config, args.complexities)

        for c in args.complexities:
            comp = cache.by_complexity[int(c)]
            if args.split == "train":
                ids = comp.train_target_ids
            elif args.split == "val":
                ids = comp.val_target_ids
            elif args.split == "test":
                ids = comp.test_target_ids
            else:
                ids = comp.target_ids
            if ids.size == 0:
                ids = comp.target_ids
            if ids.size == 0:
                print(f"complexity={c}: no cached targets, skipping")
                continue

            replace = ids.size < args.num_trials
            chosen = rng.choice(ids, size=args.num_trials, replace=replace)
            for idx_within, target_id in enumerate(chosen):
                context = comp.target_context(int(target_id))
                poly = context.target_poly
                key = poly.canonical_key()
                all_coeffs.append(poly.coeffs.flatten().astype(np.int32, copy=False))
                all_canonical_keys.append(key)
                all_generated_complexity.append(int(c))
                all_true_complexity.append(int(context.target_board_step))
                all_idx_within_complexity.append(idx_within)
                all_target_ids.append(int(target_id))
                all_target_board_steps.append(int(context.target_board_step))

            print(
                f"complexity={c}: sampled {len(chosen)} {args.split} targets "
                f"from {args.on_path_cache_dir}"
            )

        if not all_coeffs:
            raise SystemExit("No targets collected from OnPathCache.")

        coeffs_arr = np.stack(all_coeffs, axis=0).astype(np.int32)
        keys_arr = np.array([k for k in all_canonical_keys], dtype=object)
        gen_c_arr = np.array(all_generated_complexity, dtype=np.int32)
        true_c_arr = np.array(all_true_complexity, dtype=np.int32)
        idx_arr = np.array(all_idx_within_complexity, dtype=np.int32)
        target_ids_arr = np.array(all_target_ids, dtype=np.int32)
        target_board_step_arr = np.array(all_target_board_steps, dtype=np.int32)

        np.savez(
            out_path,
            target_coeffs=coeffs_arr,
            canonical_keys=keys_arr,
            generated_complexity=gen_c_arr,
            true_min_complexity=true_c_arr,
            idx_within_complexity=idx_arr,
            target_ids=target_ids_arr,
            target_board_step=target_board_step_arr,
            source=np.array("on_path_cache"),
            split=np.array(args.split),
            on_path_cache_dir=np.array(str(args.on_path_cache_dir)),
            config_n_variables=np.int32(config.n_variables),
            config_mod=np.int32(config.mod),
            config_max_degree=np.int32(config.effective_max_degree),
            config_max_steps=np.int32(config.max_steps),
            config_target_size=np.int32(target_size),
            seed=np.int32(args.seed),
        )
        print(f"\nWrote {out_path} with {coeffs_arr.shape[0]} on-path targets")
        return

    for c in args.complexities:
        seen: Dict[bytes, int] = {}
        attempts = 0
        # Try up to 4x num_trials attempts to find unique targets, but accept
        # whatever we get if generation runs out of distinct polys.
        max_attempts = 4 * args.num_trials
        while len(seen) < args.num_trials and attempts < max_attempts:
            poly, _actions = generate_random_circuit(config, c)
            key = poly.canonical_key()
            if key not in seen:
                seen[key] = len(all_coeffs)
                all_coeffs.append(
                    poly.coeffs.flatten().astype(np.int32, copy=False)
                )
                all_canonical_keys.append(key)
                all_generated_complexity.append(int(c))
                all_true_complexity.append(int(true_lookup.get(key, -1)))
                all_idx_within_complexity.append(len(seen) - 1)
                all_target_ids.append(-1)
                all_target_board_steps.append(-1)
            attempts += 1

        print(
            f"complexity={c}: collected {len(seen)} unique targets "
            f"in {attempts} attempts"
        )

    coeffs_arr = np.stack(all_coeffs, axis=0).astype(np.int32)
    keys_arr = np.array([k for k in all_canonical_keys], dtype=object)
    gen_c_arr = np.array(all_generated_complexity, dtype=np.int32)
    true_c_arr = np.array(all_true_complexity, dtype=np.int32)
    idx_arr = np.array(all_idx_within_complexity, dtype=np.int32)
    target_ids_arr = np.array(all_target_ids, dtype=np.int32)
    target_board_step_arr = np.array(all_target_board_steps, dtype=np.int32)

    np.savez(
        out_path,
        target_coeffs=coeffs_arr,
        canonical_keys=keys_arr,
        generated_complexity=gen_c_arr,
        true_min_complexity=true_c_arr,
        idx_within_complexity=idx_arr,
        target_ids=target_ids_arr,
        target_board_step=target_board_step_arr,
        source=np.array("random_circuit"),
        config_n_variables=np.int32(config.n_variables),
        config_mod=np.int32(config.mod),
        config_max_degree=np.int32(config.effective_max_degree),
        config_max_steps=np.int32(config.max_steps),
        config_target_size=np.int32(target_size),
        seed=np.int32(args.seed),
    )

    n_known = int(np.sum(true_c_arr >= 0))
    print(
        f"\nWrote {out_path} with {coeffs_arr.shape[0]} targets "
        f"({n_known} have true_min_complexity, {coeffs_arr.shape[0] - n_known} unknown)"
    )


if __name__ == "__main__":
    main()
