#!/usr/bin/env python3
"""Evaluate the generated target distribution with the project baselines.

This script mirrors the labeled curriculum generator used by
``scripts/run_full_experiment.py`` without importing that training entrypoint,
which would pull in optional Torch/W&B dependencies. It keeps the supervised
family label for each generated target, scores every target with the five
baseline upper bounds, and writes both per-target rows and paper-oriented
summary tables.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Iterable

from decomp_rl.baseline_cost import BaselineCostModel
from decomp_rl.baselines import (
    BivariateHornerBaseline,
    CSEBaseline,
    TopDownSearchBaseline,
)
from decomp_rl.factor_fp import FiniteFieldFactorizer
from decomp_rl.family_generators import (
    SupervisedExample,
    elementary_symmetric_example,
    exact_small_example,
    horner_example,
    multivariate_horner_example,
    planted_factorable_example,
)
from decomp_rl.polynomial import SparsePolynomial


BASELINE_NAMES = (
    "sparse_direct",
    "horner_one_step",
    "mv_horner",
    "cse",
    "top_down_search",
)


@dataclass(frozen=True)
class GeneratedExample:
    progress: float
    example: SupervisedExample


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
    horner_degree_max = min(
        5 + int(round((max_horner_degree - 5) * capped_progress)),
        max_horner_degree,
    )
    variable_count = min(
        max_var_count,
        2 + int(round((max_var_count - 2) * capped_progress)),
    )
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


def generate_curriculum_examples(
    rng: Random,
    count: int,
    progress: float,
    base_prime: int,
    prime_pool: list[int],
    max_var_count: int,
    max_support: int,
    max_degree: int,
    max_horner_degree: int,
    max_inner_support: int,
    factor_cache_clear_interval: int = 512,
) -> list[SupervisedExample]:
    if count <= 0:
        return []
    profile = curriculum_profile(
        progress,
        base_prime,
        prime_pool,
        max_var_count,
        max_support,
        max_degree,
        max_horner_degree,
    )
    family_weights = profile["family_weights"]
    families = [name for name, _ in family_weights]
    weights = [weight for _, weight in family_weights]
    examples: list[SupervisedExample] = []
    baseline_model = BaselineCostModel()
    factorizer = FiniteFieldFactorizer()
    try:
        for index in range(count):
            if (
                factor_cache_clear_interval > 0
                and index > 0
                and index % factor_cache_clear_interval == 0
            ):
                factorizer.clear()
            for _attempt in range(32):
                family = rng.choices(families, weights=weights, k=1)[0]
                prime = rng.choice(prime_pool)
                variable_count = int(profile["variable_count"])
                variables = make_variable_tuple(variable_count)
                try:
                    if family == "planted":
                        example = planted_factorable_example(
                            rng,
                            prime,
                            variables,
                            support_size=int(profile["planted_support"]),
                            max_degree=int(profile["planted_degree"]),
                            baseline_model=baseline_model,
                            factorizer=factorizer,
                        )
                    elif family == "horner":
                        if len(variables) > 1:
                            example = multivariate_horner_example(
                                rng,
                                prime,
                                variables,
                                outer_degree=min(
                                    max_horner_degree,
                                    int(profile["horner_degree_min"]) + 1,
                                ),
                                inner_support_size=max(
                                    1,
                                    min(
                                        max_inner_support,
                                        int(profile["planted_support"]) - 1,
                                    ),
                                ),
                                inner_max_degree=max(1, int(profile["planted_degree"])),
                            )
                        else:
                            degree = rng.randint(
                                int(profile["horner_degree_min"]),
                                int(profile["horner_degree_max"]),
                            )
                            coefficients = [
                                rng.randint(0, prime - 1) for _ in range(degree + 1)
                            ]
                            if all(coeff == 0 for coeff in coefficients):
                                coefficients[0] = 1
                            if coefficients[0] == 0:
                                coefficients[0] = 1
                            example = horner_example(coefficients, prime)
                    elif family == "elementary":
                        example = elementary_symmetric_example(
                            variable_count=max(4, variable_count),
                            degree=2,
                            prime=prime,
                        )
                    else:
                        example = exact_small_example(
                            rng,
                            prime,
                            variables=("x", "y"),
                            baseline_model=baseline_model,
                            factorizer=factorizer,
                        )
                except RuntimeError:
                    continue
                examples.append(example)
                break
            else:
                raise RuntimeError(
                    "Failed to generate a curriculum example after repeated "
                    f"attempts for profile {profile}"
                )
    finally:
        factorizer.close()
    return examples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--progress", default="0.0,0.5,1.0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prime", type=int, default=3)
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=Path("artifacts/target_distribution.jsonl"),
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=Path("artifacts/target_distribution_summary.json"),
    )
    parser.add_argument(
        "--table-csv",
        type=Path,
        default=Path("artifacts/target_distribution_table.csv"),
    )
    parser.add_argument("--latex-output", type=Path, default=None)
    parser.add_argument("--top-down-max-branches", type=int, default=3)

    # Defaults match scripts/run_full_experiment.py.
    parser.add_argument("--curriculum-extra-primes", default="5,7")
    parser.add_argument("--curriculum-max-vars", type=int, default=5)
    parser.add_argument("--curriculum-max-support", type=int, default=6)
    parser.add_argument("--curriculum-max-degree", type=int, default=4)
    parser.add_argument("--curriculum-max-horner-degree", type=int, default=8)
    parser.add_argument("--curriculum-max-inner-support", type=int, default=4)
    parser.add_argument("--factor-cache-clear-interval", type=int, default=512)
    return parser.parse_args()


def parse_progress_levels(progress: str) -> list[float]:
    levels = [float(piece.strip()) for piece in progress.split(",") if piece.strip()]
    if not levels:
        raise ValueError("--progress must be a non-empty comma-separated list")
    for level in levels:
        if level < 0.0 or level > 1.0:
            raise ValueError(f"progress levels must be in [0, 1], got {level}")
    return levels


def collect_examples(args: argparse.Namespace) -> tuple[list[GeneratedExample], dict[str, object]]:
    rng = Random(args.seed)
    generated: list[GeneratedExample] = []
    profiles: dict[str, object] = {}
    for progress in parse_progress_levels(args.progress):
        prime_pool = curriculum_prime_pool(
            args.prime,
            args.curriculum_extra_primes,
            progress=progress,
        )
        profile = curriculum_profile(
            progress,
            base_prime=args.prime,
            prime_pool=prime_pool,
            max_var_count=args.curriculum_max_vars,
            max_support=args.curriculum_max_support,
            max_degree=args.curriculum_max_degree,
            max_horner_degree=args.curriculum_max_horner_degree,
        )
        profiles[str(progress)] = profile
        print(
            {
                "stage": "generate_progress",
                "progress": progress,
                "count": args.count,
                "profile": profile,
            },
            flush=True,
        )
        examples = generate_curriculum_examples(
            rng,
            count=args.count,
            progress=progress,
            base_prime=args.prime,
            prime_pool=prime_pool,
            max_var_count=args.curriculum_max_vars,
            max_support=args.curriculum_max_support,
            max_degree=args.curriculum_max_degree,
            max_horner_degree=args.curriculum_max_horner_degree,
            max_inner_support=args.curriculum_max_inner_support,
            factor_cache_clear_interval=args.factor_cache_clear_interval,
        )
        generated.extend(GeneratedExample(progress, example) for example in examples)
    return generated, profiles


def baseline_costs(
    target: SparsePolynomial,
    base: BaselineCostModel,
    mv_horner: BivariateHornerBaseline,
    cse: CSEBaseline,
    top_down: TopDownSearchBaseline,
) -> dict[str, int]:
    return {
        "sparse_direct": int(base.sparse_direct_cost(target)),
        "horner_one_step": int(base.horner_upper_bound(target)),
        "mv_horner": int(mv_horner.cost(target)),
        "cse": int(cse.cost(target)),
        "top_down_search": int(top_down.cost(target)),
    }


def evaluate_examples(generated: list[GeneratedExample], args: argparse.Namespace) -> list[dict[str, object]]:
    base = BaselineCostModel()
    mv_horner = BivariateHornerBaseline()
    cse = CSEBaseline()
    top_down = TopDownSearchBaseline(max_branches_per_var=args.top_down_max_branches)
    rows: list[dict[str, object]] = []
    for index, item in enumerate(generated):
        example = item.example
        target = example.target
        costs = baseline_costs(target, base, mv_horner, cse, top_down)
        best_cost = min(costs.values())
        winners = [name for name in BASELINE_NAMES if costs[name] == best_cost]
        target_cost = float(example.total_cost_target)
        rows.append(
            {
                "index": index,
                "progress": item.progress,
                "family": example.family,
                "prime": target.p,
                "variables": list(target.variables),
                "var_count": len(target.variables),
                "support": target.support_size,
                "total_degree": target.total_degree,
                "max_degrees": list(target.max_degrees),
                "poly_key": target.to_key(),
                "value_target": example.value_target,
                "target_cost": target_cost,
                **costs,
                "best_baseline": best_cost,
                "best_baseline_winners": winners,
                "best_baseline_winner_count": len(winners),
                "saving_vs_best_baseline": best_cost - target_cost,
                "target_beats_all_baselines": target_cost < best_cost,
                "target_ties_or_beats_best_baseline": target_cost <= best_cost,
            }
        )
    return rows


def pct(part: int | float, whole: int) -> float:
    if whole <= 0:
        return 0.0
    return 100.0 * float(part) / float(whole)


def mean(values: Iterable[int | float]) -> float:
    values = list(values)
    return float(statistics.mean(values)) if values else 0.0


def percentile(values: Iterable[int | float], q: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    index = max(0, min(math.ceil(q * len(ordered)) - 1, len(ordered) - 1))
    return ordered[index]


def summarize_group(
    group_type: str,
    group: str,
    rows: list[dict[str, object]],
    total_count: int,
) -> dict[str, object]:
    n = len(rows)
    target_keys = [str(row["poly_key"]) for row in rows]
    summary: dict[str, object] = {
        "group_type": group_type,
        "group": group,
        "n": n,
        "pct": pct(n, total_count),
        "unique_target_pct": pct(len(set(target_keys)), n),
        "mean_vars": mean(row["var_count"] for row in rows),
        "mean_support": mean(row["support"] for row in rows),
        "mean_total_degree": mean(row["total_degree"] for row in rows),
        "mean_best_baseline": mean(row["best_baseline"] for row in rows),
        "median_best_baseline": percentile((row["best_baseline"] for row in rows), 0.5),
        "p90_best_baseline": percentile((row["best_baseline"] for row in rows), 0.9),
        "mean_target_cost": mean(row["target_cost"] for row in rows),
        "target_beats_all_baselines_pct": pct(
            sum(1 for row in rows if row["target_beats_all_baselines"]),
            n,
        ),
        "mean_saving_vs_best_baseline": mean(
            row["saving_vs_best_baseline"] for row in rows
        ),
    }
    for name in BASELINE_NAMES:
        summary[f"mean_{name}"] = mean(row[name] for row in rows)
    for name in BASELINE_NAMES:
        summary[f"winner_pct_{name}"] = pct(
            sum(1 for row in rows if name in row["best_baseline_winners"]),
            n,
        )
    return summary


def build_table(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    total = len(rows)
    table = [summarize_group("overall", "all", rows, total)]

    progress_values = sorted({float(row["progress"]) for row in rows})
    for progress in progress_values:
        group_rows = [row for row in rows if float(row["progress"]) == progress]
        table.append(summarize_group("progress", str(progress), group_rows, total))

    family_values = sorted({str(row["family"]) for row in rows})
    for family in family_values:
        group_rows = [row for row in rows if str(row["family"]) == family]
        table.append(summarize_group("family", family, group_rows, total))

    return table


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def write_summary_json(
    path: Path,
    rows: list[dict[str, object]],
    table: list[dict[str, object]],
    profiles: dict[str, object],
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {
            "count_per_progress": args.count,
            "progress": parse_progress_levels(args.progress),
            "seed": args.seed,
            "prime": args.prime,
            "curriculum_extra_primes": args.curriculum_extra_primes,
            "curriculum_max_vars": args.curriculum_max_vars,
            "curriculum_max_support": args.curriculum_max_support,
            "curriculum_max_degree": args.curriculum_max_degree,
            "curriculum_max_horner_degree": args.curriculum_max_horner_degree,
            "curriculum_max_inner_support": args.curriculum_max_inner_support,
            "top_down_max_branches": args.top_down_max_branches,
            "baseline_names": BASELINE_NAMES,
        },
        "total_targets": len(rows),
        "unique_targets": len({row["poly_key"] for row in rows}),
        "family_counts": dict(Counter(str(row["family"]) for row in rows)),
        "progress_counts": dict(Counter(str(row["progress"]) for row in rows)),
        "profiles": profiles,
        "table": table,
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def write_csv(path: Path, table: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(table[0].keys()) if table else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(table)


def format_latex_value(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value).replace("_", r"\_")


def write_latex(path: Path, table: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "group_type",
        "group",
        "n",
        "pct",
        "unique_target_pct",
        "mean_support",
        "mean_total_degree",
        "mean_best_baseline",
        "p90_best_baseline",
        "mean_target_cost",
        "target_beats_all_baselines_pct",
        "mean_saving_vs_best_baseline",
    ]
    headers = [
        "Type",
        "Group",
        "n",
        "Pct.",
        "Unique",
        "Supp.",
        "Deg.",
        "Best",
        "P90 Best",
        "Target",
        "Beats",
        "Saving",
    ]
    lines = [
        r"\begin{tabular}{llrrrrrrrrrr}",
        r"\toprule",
        " & ".join(headers) + r" \\",
        r"\midrule",
    ]
    for row in table:
        lines.append(" & ".join(format_latex_value(row[column]) for column in columns) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    with path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def print_table(table: list[dict[str, object]]) -> None:
    print()
    print(
        f"{'type':<10} {'group':<24} {'n':>5} {'pct':>7} {'unique':>8} "
        f"{'supp':>7} {'deg':>7} {'best':>8} {'target':>8} {'saving':>8}"
    )
    print("-" * 100)
    for row in table:
        print(
            f"{row['group_type']:<10} {row['group']:<24} {row['n']:>5} "
            f"{float(row['pct']):>6.1f}% {float(row['unique_target_pct']):>7.1f}% "
            f"{float(row['mean_support']):>7.2f} "
            f"{float(row['mean_total_degree']):>7.2f} "
            f"{float(row['mean_best_baseline']):>8.2f} "
            f"{float(row['mean_target_cost']):>8.2f} "
            f"{float(row['mean_saving_vs_best_baseline']):>8.2f}"
        )


def main() -> None:
    args = parse_args()
    if args.count < 0:
        raise ValueError("--count must be non-negative")

    print(
        {
            "stage": "start",
            "count_per_progress": args.count,
            "progress": parse_progress_levels(args.progress),
            "seed": args.seed,
        },
        flush=True,
    )
    generated, profiles = collect_examples(args)
    print({"stage": "evaluate", "total_targets": len(generated)}, flush=True)
    rows = evaluate_examples(generated, args)
    table = build_table(rows)

    write_jsonl(args.output_jsonl, rows)
    write_summary_json(args.summary_json, rows, table, profiles, args)
    write_csv(args.table_csv, table)
    if args.latex_output is not None:
        write_latex(args.latex_output, table)

    print_table(table)
    print()
    print(
        {
            "stage": "done",
            "jsonl": str(args.output_jsonl),
            "summary_json": str(args.summary_json),
            "table_csv": str(args.table_csv),
            "latex_output": str(args.latex_output) if args.latex_output else None,
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
