#!/usr/bin/env python3
"""Generate the interesting-polynomial dataset once, stratify by shortest_length,
then split train/eval deterministically with hash ordering, and write JSONL files.
Subsequent training runs load these
instead of rebuilding the game graph (the build is slow and non-deterministic
when caps bite — freezing gives reproducibility + speed).

Usage:
    python scripts/build_dataset.py --n_vars 2 --max_ops 6 \
        --out_dir data/ --eval_frac 0.2 --seed 0

Output files:
    data/polys_nvars{N}_maxops{K}.train.jsonl
    data/polys_nvars{N}_maxops{K}.eval.jsonl
    data/polys_nvars{N}_maxops{K}.meta.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import warnings
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from poly_circuit_rl.core.poly import poly_hashkey
from poly_circuit_rl.env.graph_enumeration import analyze_graph, build_game_graph
from poly_circuit_rl.env.samplers import _sympy_expr_to_poly


def stable_hash_value(key_bytes: bytes, salt: bytes) -> int:
    """Deterministic integer hash from poly key + salt."""
    h = hashlib.sha256(salt + key_bytes).digest()
    return int.from_bytes(h[:8], "big")


def eval_size_for_group(group_size: int, eval_frac: float) -> int:
    """Choose eval-set size for one shortest-length stratum."""
    if group_size <= 0 or eval_frac <= 0.0:
        return 0
    if eval_frac >= 1.0:
        return group_size

    n_eval = int(round(eval_frac * group_size))
    if n_eval == 0:
        n_eval = 1
    if n_eval == group_size and group_size > 1:
        n_eval = group_size - 1
    return n_eval


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_vars", type=int, default=2)
    p.add_argument("--max_ops", type=int, default=6)
    p.add_argument("--out_dir", type=str, default="data/")
    p.add_argument("--eval_frac", type=float, default=0.2,
                   help="Fraction held out for eval within each shortest_length bucket.")
    p.add_argument("--seed", type=int, default=0,
                   help="Salt for hash bucketing — lets you reshuffle.")
    p.add_argument(
        "--only_shortcut",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep only shortcut polynomials (default true). Use --no-only_shortcut to disable.",
    )
    p.add_argument("--min_shortcut_gap", type=int, default=2)
    p.add_argument("--max_graph_nodes", type=int, default=500_000,
                   help="Higher than training default — we only pay this once.")
    p.add_argument("--max_successors", type=int, default=None)
    p.add_argument("--max_seconds", type=float, default=900.0,
                   help="Wall-clock cap; raise freely since this is one-shot.")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"polys_nvars{args.n_vars}_maxops{args.max_ops}"
    train_path = out_dir / f"{tag}.train.jsonl"
    eval_path = out_dir / f"{tag}.eval.jsonl"
    meta_path = out_dir / f"{tag}.meta.json"

    print(f"Building game graph: n_vars={args.n_vars} steps={args.max_ops}")
    t0 = time.time()
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always")
        G = build_game_graph(
            steps=args.max_ops,
            num_vars=args.n_vars,
            max_nodes=args.max_graph_nodes,
            max_successors_per_node=args.max_successors,
            max_seconds=args.max_seconds,
        )
        cap_warnings = [str(w.message) for w in wlist if "exceeded" in str(w.message)]
    build_sec = time.time() - t0
    print(f"  graph: {len(G)} nodes in {build_sec:.1f}s")
    if cap_warnings:
        print("  WARNING: caps bit during build:")
        for w in cap_warnings:
            print(f"    - {w}")

    print("Analyzing graph...")
    records, _, _ = analyze_graph(
        G,
        only_multipath=False,
        only_shortcut=args.only_shortcut,
        min_shortcut_gap=args.min_shortcut_gap,
        max_step=args.max_ops,
        num_vars=args.n_vars,
    )
    print(f"  {len(records)} candidate polynomials")

    var_names = ["x"] if args.n_vars == 1 else [f"x{i}" for i in range(args.n_vars)]

    if not 0.0 <= args.eval_frac <= 1.0:
        raise ValueError(f"--eval_frac must be in [0, 1], got {args.eval_frac}")

    salt = args.seed.to_bytes(8, "big", signed=False)
    train_by_len: dict = defaultdict(int)
    eval_by_len: dict = defaultdict(int)
    rows_by_len: dict[int, list[tuple[int, dict]]] = defaultdict(list)
    seen_keys: set = set()
    dropped_parse = 0
    dropped_dup = 0
    t_train = train_path.open("w")
    t_eval = eval_path.open("w")

    try:
        for rec in records:
            sl = rec.get("shortest_length")
            if sl is None or sl <= 0:
                continue
            expr_str = rec.get("expr_str")
            if not expr_str:
                continue
            try:
                poly = _sympy_expr_to_poly(expr_str, var_names)
            except Exception:
                dropped_parse += 1
                continue
            key = poly_hashkey(poly)
            if key in seen_keys:
                dropped_dup += 1
                continue
            seen_keys.add(key)

            out_rec = {
                "expr_str": expr_str,
                "shortest_length": sl,
                "naive_ops": rec.get("naive_ops"),
                "shortcut_gap": rec.get("shortcut_gap"),
                "shortest_path_count": rec.get("shortest_path_count"),
                "multiple_shortest_paths": rec.get("multiple_shortest_paths", False),
                # Canonical poly key as a JSON-safe list-of-[monom, num, den]
                "poly_key": [[list(m), int(c.numerator), int(c.denominator)]
                             for m, c in [(mono, Fraction(coef))
                                          for mono, coef in sorted(poly.items())]],
            }

            key_bytes = json.dumps(out_rec["poly_key"], sort_keys=True).encode()
            rows_by_len[int(sl)].append((stable_hash_value(key_bytes, salt), out_rec))

        for sl, rows in sorted(rows_by_len.items()):
            rows.sort(key=lambda item: item[0])
            n_eval = eval_size_for_group(len(rows), args.eval_frac)
            for idx, (_, out_rec) in enumerate(rows):
                if idx < n_eval:
                    t_eval.write(json.dumps(out_rec) + "\n")
                    eval_by_len[sl] += 1
                else:
                    t_train.write(json.dumps(out_rec) + "\n")
                    train_by_len[sl] += 1
    finally:
        t_train.close()
        t_eval.close()

    meta = {
        "n_vars": args.n_vars,
        "max_ops": args.max_ops,
        "eval_frac_requested": args.eval_frac,
        "seed": args.seed,
        "build_sec": round(build_sec, 2),
        "graph_nodes": len(G),
        "cap_warnings": cap_warnings,
        "min_shortcut_gap": args.min_shortcut_gap,
        "only_shortcut": args.only_shortcut,
        "train_count": sum(train_by_len.values()),
        "eval_count": sum(eval_by_len.values()),
        "train_by_length": dict(sorted(train_by_len.items())),
        "eval_by_length": dict(sorted(eval_by_len.items())),
        "dropped_parse": dropped_parse,
        "dropped_duplicate": dropped_dup,
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    print("\n=== dataset summary ===")
    print(json.dumps(meta, indent=2))
    print(f"\nTrain file: {train_path}")
    print(f"Eval  file: {eval_path}")
    print(f"Meta  file: {meta_path}")

    if meta["eval_count"] == 0:
        print("\nERROR: eval split is empty. Raise --eval_frac or check filters.",
              file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
