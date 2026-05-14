#!/usr/bin/env python3
"""Run the cost baselines against the same targets training will see.

Generates targets via either the supervised pretraining mixture or the
curriculum-progress sampler used by ``run_full_experiment.py``, then prices
each target under five upper-bound cost models:

  * ``sparse_direct``    -- per-monomial build + final adds (loosest bound)
  * ``horner_one_step``  -- existing ``BaselineCostModel.horner_upper_bound``
                            (recursive, one-step pivot per variable)
  * ``mv_horner``        -- ``BivariateHornerBaseline`` (gap-aware multivariate
                            Horner with both orderings tried)
  * ``cse``              -- ``CSEBaseline`` (per-var shared power chain + term
                            combine + final adds)
  * ``top_down_search``  -- ``TopDownSearchBaseline`` (memoised B&B over
                            power-pivot splits, k = 1..max_branches)

Each row of the output JSONL contains per-target metadata, every baseline's
integer cost, and the wall-clock per-baseline time. A summary table is
printed at the end so this can also serve as a printable comparison sheet.

Examples
--------

Curriculum-style sweep at three progress levels (matches the targets
``run_full_experiment.py`` actually trains on):

    python scripts/run_baseline_sweep.py \\
        --source curriculum \\
        --progress 0.0,0.5,1.0 \\
        --count 200 \\
        --output artifacts/baseline_sweep_curriculum.jsonl

Pretraining-mixture sweep (40/25/20/15 planted/horner/elementary/exact):

    python scripts/run_baseline_sweep.py \\
        --source mixture \\
        --count 500 \\
        --output artifacts/baseline_sweep_mixture.jsonl

Skip the slow search-based baseline if you just want the closed-form bounds:

    python scripts/run_baseline_sweep.py --no-top-down-search ...
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from random import Random

from decomp_rl.baseline_cost import BaselineCostModel
from decomp_rl.baselines import (
    BivariateHornerBaseline,
    CSEBaseline,
    TopDownSearchBaseline,
)
from decomp_rl.family_generators import (
    elementary_symmetric_example,
    exact_small_example,
    horner_example,
    multivariate_horner_example,
    planted_factorable_example,
    pretraining_mixture,
)
from decomp_rl.polynomial import SparsePolynomial


# -----------------------------------------------------------------------------
# Curriculum helpers copied verbatim from scripts/run_full_experiment.py so we
# don't have to import that module (which imports torch at top level). If those
# definitions ever change there, mirror the changes here.
# -----------------------------------------------------------------------------


def curriculum_prime_pool(base_prime: int, extra_primes: str, progress: float) -> list[int]:
    primes = [base_prime]
    extras = [int(piece.strip()) for piece in extra_primes.split(",") if piece.strip()]
    if progress >= 0.34 and extras:
        primes.append(extras[0])
    if progress >= 0.67:
        primes.extend(extras[1:])
    return sorted({prime for prime in primes if prime > 1})


def make_variable_tuple(count: int) -> tuple[str, ...]:
    if count <= 1:
        return ("x",)
    if count == 2:
        return ("x", "y")
    return tuple(f"x{i + 1}" for i in range(count))


def curriculum_profile(
    progress: float,
    base_prime: int,
    prime_pool: list[int],
    max_var_count: int,
    max_support: int,
    max_degree: int,
    max_horner_degree: int,
) -> dict[str, object]:
    capped_progress = max(0.0, min(1.0, progress))
    planted_support = min(3 + int(round((max_support - 3) * capped_progress)), max_support)
    planted_degree = min(2 + int(round((max_degree - 2) * capped_progress)), max_degree)
    horner_degree_min = 3 + int(round(2 * capped_progress))
    horner_degree_max = min(5 + int(round((max_horner_degree - 5) * capped_progress)), max_horner_degree)
    variable_count = min(max_var_count, 2 + int(round((max_var_count - 2) * capped_progress)))
    family_weights = (
        ("planted", 0.50 - 0.10 * capped_progress),
        ("horner", 0.25 - 0.05 * capped_progress),
        ("elementary", 0.20 + 0.05 * capped_progress),
        ("exact_small", 0.05 + 0.10 * capped_progress),
    )
    return {
        "base_prime": base_prime,
        "prime_pool": prime_pool,
        "variable_count": variable_count,
        "planted_support": planted_support,
        "planted_degree": planted_degree,
        "horner_degree_min": horner_degree_min,
        "horner_degree_max": horner_degree_max,
        "family_weights": family_weights,
    }


def generate_cycle_targets(
    rng: Random,
    count: int,
    cycle: int,
    base_prime: int,
    prime_pool: list[int],
    max_var_count: int,
    max_support: int,
    max_degree: int,
    max_horner_degree: int,
    max_inner_support: int,
) -> list[SparsePolynomial]:
    targets: list[SparsePolynomial] = []
    profile = curriculum_profile(
        min(1.0, cycle / max(1, cycle + 2)),
        base_prime=base_prime,
        prime_pool=prime_pool,
        max_var_count=max_var_count,
        max_support=max_support,
        max_degree=max_degree,
        max_horner_degree=max_horner_degree,
    )
    for index in range(count):
        for _attempt in range(32):
            prime = rng.choice(prime_pool)
            selector = index % 4
            try:
                if selector == 0:
                    target = planted_factorable_example(
                        rng,
                        prime,
                        make_variable_tuple(int(profile["variable_count"])),
                        support_size=int(profile["planted_support"]),
                        max_degree=int(profile["planted_degree"]),
                    ).target
                elif selector == 1:
                    variables = make_variable_tuple(int(profile["variable_count"]))
                    if len(variables) > 1:
                        target = multivariate_horner_example(
                            rng,
                            prime,
                            variables,
                            outer_degree=min(max_horner_degree, int(profile["horner_degree_min"]) + 1),
                            inner_support_size=max(1, min(max_inner_support, int(profile["planted_support"]) - 1)),
                            inner_max_degree=max(1, int(profile["planted_degree"])),
                        ).target
                    else:
                        degree = rng.randint(int(profile["horner_degree_min"]), int(profile["horner_degree_max"]))
                        coefficients = [rng.randint(0, prime - 1) for _ in range(degree + 1)]
                        if all(coeff == 0 for coeff in coefficients):
                            coefficients[0] = 1
                        if coefficients[0] == 0:
                            coefficients[0] = 1
                        target = horner_example(coefficients, prime).target
                elif selector == 2:
                    variable_count = max(4, int(profile["variable_count"]))
                    target = elementary_symmetric_example(
                        variable_count=variable_count, degree=2, prime=prime
                    ).target
                else:
                    target = exact_small_example(rng, prime, variables=("x", "y")).target
            except RuntimeError:
                continue
            targets.append(target)
            break
        else:
            raise RuntimeError(
                f"Failed to generate a cycle target after repeated attempts for profile {profile}"
            )
    return targets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--source",
        choices=("mixture", "curriculum"),
        default="curriculum",
        help="Target generator: 'mixture' uses pretraining_mixture, "
        "'curriculum' uses generate_cycle_targets at one or more progress levels.",
    )
    parser.add_argument("--prime", type=int, default=3)
    parser.add_argument(
        "--variables",
        nargs="+",
        default=["x", "y"],
        help="Variable names for the mixture source. Curriculum scales variable count itself.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=200,
        help="Targets per progress level (curriculum) or total (mixture).",
    )
    parser.add_argument(
        "--progress",
        default="0.0,0.5,1.0",
        help="Comma-separated curriculum progress values in [0, 1]. Ignored when --source=mixture.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/baseline_sweep.jsonl"),
        help="Per-target JSONL output path.",
    )
    parser.add_argument(
        "--no-top-down-search",
        action="store_true",
        help="Skip the TopDownSearch baseline (slowest; quadratic in support_size).",
    )
    parser.add_argument(
        "--top-down-max-branches",
        type=int,
        default=3,
        help="max_branches_per_var for TopDownSearchBaseline.",
    )

    # Curriculum knobs -- defaults match run_full_experiment.py:91-96.
    parser.add_argument("--curriculum-extra-primes", default="5,7")
    parser.add_argument("--curriculum-max-vars", type=int, default=5)
    parser.add_argument("--curriculum-max-support", type=int, default=6)
    parser.add_argument("--curriculum-max-degree", type=int, default=4)
    parser.add_argument("--curriculum-max-horner-degree", type=int, default=8)
    parser.add_argument("--curriculum-max-inner-support", type=int, default=4)
    return parser.parse_args()


def collect_targets(args: argparse.Namespace) -> list[tuple[float, SparsePolynomial]]:
    """Return a list of ``(progress, target)`` pairs.

    For mixture, ``progress`` is reported as ``-1.0`` so the column still
    exists in the JSONL but is clearly tagged as 'not a curriculum sample'.
    """
    rng = Random(args.seed)
    if args.source == "mixture":
        examples = pretraining_mixture(
            rng,
            args.prime,
            args.count,
            variables=tuple(args.variables),
        )
        return [(-1.0, ex.target) for ex in examples]

    progress_levels = [float(p) for p in args.progress.split(",") if p.strip()]
    if not progress_levels:
        raise ValueError("--progress must be a non-empty comma-separated list")

    pairs: list[tuple[float, SparsePolynomial]] = []
    for level in progress_levels:
        prime_pool = curriculum_prime_pool(
            args.prime,
            args.curriculum_extra_primes,
            progress=level,
        )
        targets = generate_cycle_targets(
            rng,
            count=args.count,
            cycle=max(1, int(round(level * 10))),
            base_prime=args.prime,
            prime_pool=prime_pool,
            max_var_count=args.curriculum_max_vars,
            max_support=args.curriculum_max_support,
            max_degree=args.curriculum_max_degree,
            max_horner_degree=args.curriculum_max_horner_degree,
            max_inner_support=args.curriculum_max_inner_support,
        )
        pairs.extend((level, target) for target in targets)
    return pairs


def _timed(fn, *args):
    t0 = time.perf_counter()
    result = fn(*args)
    return result, (time.perf_counter() - t0) * 1e6  # microseconds


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.unlink(missing_ok=True)

    print({"stage": "generate_targets", "source": args.source, "count": args.count}, flush=True)
    pairs = collect_targets(args)
    print({"stage": "generate_done", "total_targets": len(pairs)}, flush=True)

    # Instantiate baselines once so per-instance caches accumulate across the sweep.
    base = BaselineCostModel()
    mv_horner = BivariateHornerBaseline()
    cse = CSEBaseline()
    top_down = (
        None
        if args.no_top_down_search
        else TopDownSearchBaseline(max_branches_per_var=args.top_down_max_branches)
    )

    by_baseline_costs: dict[str, list[int]] = {
        "sparse_direct": [],
        "horner_one_step": [],
        "mv_horner": [],
        "cse": [],
    }
    by_baseline_times: dict[str, list[float]] = {k: [] for k in by_baseline_costs}
    if top_down is not None:
        by_baseline_costs["top_down_search"] = []
        by_baseline_times["top_down_search"] = []

    with args.output.open("w") as fh:
        for index, (progress, target) in enumerate(pairs):
            sparse_cost, sparse_us = _timed(base.sparse_direct_cost, target)
            # horner_upper_bound's recursion shares cache with direct_construction_cost,
            # so call the umbrella method to be fair: it returns min(sparse, horner).
            horner_cost, horner_us = _timed(base.horner_upper_bound, target)
            mv_cost, mv_us = _timed(mv_horner.cost, target)
            cse_cost, cse_us = _timed(cse.cost, target)

            row = {
                "index": index,
                "progress": progress,
                "prime": target.p,
                "variables": list(target.variables),
                "support": target.support_size,
                "total_degree": target.total_degree,
                "max_degrees": list(target.max_degrees),
                "poly_key": target.to_key(),
                "sparse_direct": int(sparse_cost),
                "horner_one_step": int(horner_cost),
                "mv_horner": int(mv_cost),
                "cse": int(cse_cost),
                "sparse_direct_us": sparse_us,
                "horner_one_step_us": horner_us,
                "mv_horner_us": mv_us,
                "cse_us": cse_us,
            }

            by_baseline_costs["sparse_direct"].append(int(sparse_cost))
            by_baseline_costs["horner_one_step"].append(int(horner_cost))
            by_baseline_costs["mv_horner"].append(int(mv_cost))
            by_baseline_costs["cse"].append(int(cse_cost))
            by_baseline_times["sparse_direct"].append(sparse_us)
            by_baseline_times["horner_one_step"].append(horner_us)
            by_baseline_times["mv_horner"].append(mv_us)
            by_baseline_times["cse"].append(cse_us)

            if top_down is not None:
                td_cost, td_us = _timed(top_down.cost, target)
                row["top_down_search"] = int(td_cost)
                row["top_down_search_us"] = td_us
                by_baseline_costs["top_down_search"].append(int(td_cost))
                by_baseline_times["top_down_search"].append(td_us)

            fh.write(json.dumps(row) + "\n")

    print_summary(args, pairs, by_baseline_costs, by_baseline_times)


def print_summary(
    args: argparse.Namespace,
    pairs: list[tuple[float, SparsePolynomial]],
    costs: dict[str, list[int]],
    times: dict[str, list[float]],
) -> None:
    print()
    print(f"Wrote {len(pairs)} rows to {args.output}")
    print()
    print("                       cost                         time (microseconds)")
    print(f"{'baseline':<20}  {'mean':>8} {'median':>8} {'p95':>8} {'max':>8}   "
          f"{'mean':>8} {'p50':>8} {'p95':>8} {'max':>10}")
    print("-" * 100)
    for name in costs:
        cs = costs[name]
        ts = times[name]
        cs_sorted = sorted(cs)
        ts_sorted = sorted(ts)

        def pct(xs, q):
            return xs[min(int(len(xs) * q), len(xs) - 1)] if xs else 0

        print(
            f"{name:<20}  "
            f"{statistics.mean(cs):>8.2f} {pct(cs_sorted, 0.5):>8} "
            f"{pct(cs_sorted, 0.95):>8} {max(cs):>8}   "
            f"{statistics.mean(ts):>8.1f} {pct(ts_sorted, 0.5):>8.1f} "
            f"{pct(ts_sorted, 0.95):>8.1f} {max(ts):>10.1f}"
        )

    # Improvement table: how often each baseline strictly beats sparse_direct.
    sparse = costs["sparse_direct"]
    print()
    print("Improvement vs sparse_direct (per target):")
    print(f"{'baseline':<20}  {'strict <':>10} {'tied =':>10} {'mean Δ':>10} {'best Δ':>10}")
    print("-" * 70)
    for name in costs:
        if name == "sparse_direct":
            continue
        diffs = [sparse[i] - costs[name][i] for i in range(len(sparse))]
        strict = sum(1 for d in diffs if d > 0)
        tied = sum(1 for d in diffs if d == 0)
        print(
            f"{name:<20}  "
            f"{strict:>10} {tied:>10} {statistics.mean(diffs):>10.2f} {max(diffs):>10}"
        )


if __name__ == "__main__":
    main()
