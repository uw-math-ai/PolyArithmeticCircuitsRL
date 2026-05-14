"""Bucket random 2-var F_3 polys by 'efficient cost' = min over my baselines,
then show how far each individual baseline is from that minimum at each level.

User definition: c_k = polynomial constructible in k operations.
                 (x+y)^2 is c2 -- one add + one square = 2 ops.

CAVEAT: these baselines only consider ADDITIVE decompositions of the target.
        Polynomials with strong multiplicative structure (like (x+y)^2) will be
        scored too high by min(baselines). The DecompEnv used by training does
        consider multiplicative factorization, so a trained RL model can beat
        these numbers on those families.
"""

import sys, statistics, time
from random import Random

sys.path.insert(0, "/Users/kj/Documents/Poly/top-down/src")

from decomp_rl.baseline_cost import BaselineCostModel
from decomp_rl.baselines import (
    BivariateHornerBaseline,
    CSEBaseline,
    TopDownSearchBaseline,
)
from decomp_rl.family_generators import random_sparse_polynomial


PRIME = 3
VARS = ("x", "y")
LEVELS = list(range(1, 11))            # c1 .. c10
TARGET_PER_LEVEL = 200
HARD_CAP = 200_000


def best_upper_bound(poly, base, mv, cse, td):
    """min over the 5 baselines = our 'efficient cost' proxy."""
    return min(
        base.sparse_direct_cost(poly),
        base.horner_upper_bound(poly),
        mv.cost(poly),
        cse.cost(poly),
        td.cost(poly),
    )


def fmt_example(poly):
    """Compact poly repr, shortened."""
    s = repr(poly)
    return s if len(s) <= 60 else s[:57] + "..."


base = BaselineCostModel()
mv = BivariateHornerBaseline()
cse = CSEBaseline()
td = TopDownSearchBaseline()

rng = Random(0)
buckets = {k: [] for k in LEVELS}

print(f"Sampling random 2-var F_{PRIME} polys, bucketing by min(baselines)...",
      flush=True)
print(f"Target: {TARGET_PER_LEVEL} polys in each of c1..c{LEVELS[-1]} "
      f"(hard cap {HARD_CAP} attempts).", flush=True)
print(flush=True)

t_start = time.time()
sampled = 0
last_log = 0
LOG_EVERY = 500

while any(len(buckets[k]) < TARGET_PER_LEVEL for k in LEVELS) and sampled < HARD_CAP:
    sampled += 1
    support = rng.randint(1, 8)
    max_deg = rng.randint(1, 5)
    poly = random_sparse_polynomial(rng, PRIME, VARS, support, max_deg)

    sparse_v = base.sparse_direct_cost(poly)
    horn_v   = base.horner_upper_bound(poly)
    mv_v     = mv.cost(poly)
    cse_v    = cse.cost(poly)
    td_v     = td.cost(poly)
    c = min(sparse_v, horn_v, mv_v, cse_v, td_v)

    if c in buckets and len(buckets[c]) < TARGET_PER_LEVEL:
        buckets[c].append({
            "poly": poly,
            "support": poly.support_size,
            "total_degree": poly.total_degree,
            "sparse_direct":   sparse_v,
            "horner_one_step": horn_v,
            "mv_horner":       mv_v,
            "cse":             cse_v,
            "top_down":        td_v,
        })

    if sampled - last_log >= LOG_EVERY:
        last_log = sampled
        fill = " ".join(f"c{k}:{len(buckets[k])}" for k in LEVELS)
        elapsed = time.time() - t_start
        print(f"[{elapsed:6.1f}s] sampled={sampled:>6}  {fill}", flush=True)

elapsed = time.time() - t_start

print(f"\nSampled {sampled} random 2-var F_{PRIME} polys in {elapsed:.1f}s.")
print(f"c_k = level where min over all baselines == k (= our upper bound on efficient cost).\n")

# Counts table
print("Bucket fill:")
for k in LEVELS:
    print(f"  c{k:<2}: {len(buckets[k]):>4} polys")
print()

print("Mean number of steps per baseline at each complexity level:")
print(f"{'level':<6} {'n':>5}  {'support':>8} {'tot_deg':>8}  "
      f"{'sparse':>8} {'horner':>8} {'mvHorn':>8} {'cse':>8} {'topdn':>8}")
print("-" * 84)
for k in LEVELS:
    rows = buckets[k]
    if not rows:
        print(f"c{k:<5} {0:>5}")
        continue
    def m(field):
        return statistics.mean(r[field] for r in rows)
    print(
        f"c{k:<5} {len(rows):>5}  "
        f"{m('support'):>8.2f} {m('total_degree'):>8.2f}  "
        f"{m('sparse_direct'):>8.2f} {m('horner_one_step'):>8.2f} "
        f"{m('mv_horner'):>8.2f} {m('cse'):>8.2f} {m('top_down'):>8.2f}"
    )

print("\nOne representative example per level (smallest support found):")
for k in LEVELS:
    rows = buckets[k]
    if not rows:
        continue
    ex = min(rows, key=lambda r: (r["support"], r["total_degree"]))
    print(f"  c{k:<2}  support={ex['support']}  deg={ex['total_degree']:>2}  "
          f"poly = {fmt_example(ex['poly'])}")
