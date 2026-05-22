#!/usr/bin/env python3
"""Generate a curriculum bucketed by exact circuit complexity Ck.

Each target is labeled with ``ck`` = the op count of the shortest circuit found
by ``decomp_rl.complexity`` (exact search over whole-poly factorization +
additive splits for small polys; multi-baseline upper bound for large ones).
Targets are then binned into exact Ck buckets (C2, C3, C4, ...), keeping up to
``--per-bucket`` per value within ``[--ck-min, --ck-max]``.

To populate a spread of complexities the candidate pool mixes:
  * random sparse polynomials (small support / degree),
  * products of 2-3 random low-degree factors (some squared) — these are the
    factorable, low-Ck targets like (x+y)^2.

Output JSONL fields (consumed by run_ppo_curriculum.py):
    {"prime", "variables", "terms", "ck", "method", "level", "source"}
``level`` mirrors ``ck`` so the launcher groups eval metrics per complexity.

Example:
    python scripts/generate_ck_curriculum.py \
        --output artifacts/ck_curriculum/targets.jsonl \
        --prime 3 --variables x y z \
        --ck-min 2 --ck-max 9 --per-bucket 10 --seed 0
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from random import Random

from decomp_rl.baseline_cost import BaselineCostModel
from decomp_rl.baselines import BaselineBundle
from decomp_rl.complexity import circuit_complexity
from decomp_rl.config import FactorizerConfig
from decomp_rl.factor_fp import FiniteFieldFactorizer
from decomp_rl.family_generators import random_sparse_polynomial
from decomp_rl.polynomial import SparsePolynomial


def _random_factor(rng: Random, prime: int, variables: tuple[str, ...]) -> SparsePolynomial:
    """A small low-degree factor: 2-3 terms, degree <= 2."""
    support = rng.randint(2, 3)
    return random_sparse_polynomial(rng, prime, variables, support_size=support, max_degree=2)


def _candidate_pool(rng: Random, prime: int, variables: tuple[str, ...], n: int):
    """Yield ~n candidate polynomials mixing random and factored constructions."""
    for _ in range(n):
        kind = rng.random()
        if kind < 0.45:
            # Product of 2-3 factors, occasionally squaring one (gives perfect
            # powers like (x+y)^2 with low Ck).
            n_factors = rng.randint(2, 3)
            poly = SparsePolynomial.one(prime, variables)
            for _ in range(n_factors):
                f = _random_factor(rng, prime, variables)
                if rng.random() < 0.3:
                    f = f * f
                poly = poly * f
            yield poly
        else:
            support = rng.randint(2, 7)
            max_deg = rng.randint(1, 4)
            yield random_sparse_polynomial(rng, prime, variables, support, max_deg)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=Path, default=Path("artifacts/ck_curriculum/targets.jsonl"))
    p.add_argument("--prime", type=int, default=3)
    p.add_argument("--variables", nargs="+", default=["x", "y", "z"])
    p.add_argument("--ck-min", type=int, default=2)
    p.add_argument("--ck-max", type=int, default=9)
    p.add_argument("--per-bucket", type=int, default=10)
    p.add_argument("--pool-size", type=int, default=6000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    variables = tuple(args.variables)
    rng = Random(args.seed)
    bm = BaselineCostModel()
    bundle = BaselineBundle(baseline_model=bm)
    fac = FiniteFieldFactorizer(FactorizerConfig())

    buckets: dict[int, list[dict]] = defaultdict(list)
    seen: set[str] = set()
    targets_needed = (args.ck_max - args.ck_min + 1) * args.per_bucket

    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        for poly in _candidate_pool(rng, args.prime, variables, args.pool_size):
            if poly.is_zero or poly.is_constant or poly.support_size < 2:
                continue
            key = poly.to_key()
            if key in seen:
                continue
            seen.add(key)
            ck, method = circuit_complexity(poly, fac, bm, bundle=bundle)
            if ck < args.ck_min or ck > args.ck_max:
                continue
            if len(buckets[ck]) >= args.per_bucket:
                continue
            buckets[ck].append({
                "prime": poly.p,
                "variables": list(poly.variables),
                "terms": [[int(c), [int(e) for e in exp]] for c, exp in poly.terms],
                "ck": ck,
                "method": method,
                "level": ck,
                "source": "product" if poly.support_size <= 1 else "pool",
            })
            filled = sum(min(len(v), args.per_bucket) for v in buckets.values())
            if filled >= targets_needed and all(
                len(buckets[c]) >= args.per_bucket for c in range(args.ck_min, args.ck_max + 1)
            ):
                break
    finally:
        fac.close()

    records = [rec for ck in sorted(buckets) for rec in buckets[ck]]
    with args.output.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")

    print(f"Wrote {len(records)} targets to {args.output}")
    print("Per-Ck counts (method mix):")
    for ck in range(args.ck_min, args.ck_max + 1):
        recs = buckets.get(ck, [])
        n_exact = sum(1 for r in recs if r["method"] == "exact")
        print(f"  C{ck}: {len(recs):>2}  (exact={n_exact}, heuristic={len(recs)-n_exact})")


if __name__ == "__main__":
    main()
