#!/usr/bin/env python3
"""Generate a graded target curriculum for top-down PPO training.

Unlike the two trivial built-in targets in ``run_ppo_finetune.py``, every
polynomial emitted here is constructed so that a *beneficial* additive
decomposition exists (the planted / Horner family generators only return an
example when its decomposed circuit cost beats the sparse-direct baseline).
That gives PPO an actual cost-savings signal to learn.

Each line is JSON with the fields ``run_ppo_finetune.load_targets`` consumes
(``prime``, ``variables``, ``terms``) plus two extra tags used by
``run_ppo_curriculum.py`` for per-bucket metrics:

    {"prime": 3, "variables": ["x","y","z"],
     "terms": [[c, [e0,e1,e2]], ...],
     "level": 2, "family": "planted_factorable"}

Complexity ``level`` (1..4) scales support size and degree, so the W&B
dashboard can show success rate broken out by difficulty.

Example:
    python scripts/generate_target_curriculum.py \
        --output artifacts/curriculum/targets.jsonl \
        --prime 3 --variables x y z --per-level 12 --seed 0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from random import Random

from decomp_rl.baseline_cost import BaselineCostModel
from decomp_rl.factor_fp import FiniteFieldFactorizer
from decomp_rl.family_generators import (
    multivariate_horner_example,
    planted_factorable_example,
)
from decomp_rl.polynomial import SparsePolynomial

# Per-level recipe: a list of (family, kwargs) generators sampled round-robin to
# fill that level's quota. Difficulty grows with support size / degree.
LEVEL_RECIPES = {
    1: [("planted", dict(support_size=2, max_degree=2))],
    2: [("planted", dict(support_size=3, max_degree=2))],
    3: [
        ("planted", dict(support_size=3, max_degree=3)),
        ("mv_horner", dict(outer_degree=2, inner_support_size=2, inner_max_degree=2)),
    ],
    4: [
        ("planted", dict(support_size=4, max_degree=3)),
        ("mv_horner", dict(outer_degree=3, inner_support_size=2, inner_max_degree=2)),
    ],
}


def _serialize(poly: SparsePolynomial, level: int, family: str) -> dict:
    return {
        "prime": poly.p,
        "variables": list(poly.variables),
        "terms": [[int(coeff), [int(e) for e in exponent]] for coeff, exponent in poly.terms],
        "level": level,
        "family": family,
    }


def _generate_one(
    family: str,
    kwargs: dict,
    rng: Random,
    prime: int,
    variables: tuple[str, ...],
    baseline_model: BaselineCostModel,
    factorizer: FiniteFieldFactorizer,
) -> SparsePolynomial:
    if family == "planted":
        example = planted_factorable_example(
            rng, prime, variables,
            baseline_model=baseline_model, factorizer=factorizer, **kwargs,
        )
    elif family == "mv_horner":
        example = multivariate_horner_example(rng, prime, variables, **kwargs)
    else:
        raise ValueError(f"unknown family {family!r}")
    return example.target


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=Path, default=Path("artifacts/curriculum/targets.jsonl"))
    p.add_argument("--prime", type=int, default=3)
    p.add_argument("--variables", nargs="+", default=["x", "y", "z"])
    p.add_argument("--per-level", type=int, default=12, help="Targets to emit per complexity level.")
    p.add_argument("--levels", nargs="+", type=int, default=[1, 2, 3, 4])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-attempts-mult", type=int, default=20,
                   help="Per-target generation retries = per_level * this before giving up on a level.")
    args = p.parse_args()

    variables = tuple(args.variables)
    rng = Random(args.seed)
    baseline_model = BaselineCostModel()
    factorizer = FiniteFieldFactorizer(library=None)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    seen: set[str] = set()

    try:
        for level in args.levels:
            recipe = LEVEL_RECIPES[level]
            produced = 0
            attempts = 0
            attempt_budget = max(args.per_level * args.max_attempts_mult, 50)
            while produced < args.per_level and attempts < attempt_budget:
                family, kwargs = recipe[produced % len(recipe)]
                attempts += 1
                try:
                    target = _generate_one(
                        family, kwargs, rng, args.prime, variables, baseline_model, factorizer
                    )
                except (RuntimeError, ValueError):
                    continue
                key = target.to_key()
                if key in seen or target.is_zero or target.support_size < 2:
                    continue
                seen.add(key)
                records.append(_serialize(target, level, family))
                produced += 1
            print(f"level {level}: produced {produced}/{args.per_level} "
                  f"(after {attempts} attempts)", flush=True)
    finally:
        factorizer.close()

    with args.output.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")

    by_level: dict[int, int] = {}
    for rec in records:
        by_level[rec["level"]] = by_level.get(rec["level"], 0) + 1
    print(f"Wrote {len(records)} targets to {args.output}")
    print(f"Per-level counts: {dict(sorted(by_level.items()))}")


if __name__ == "__main__":
    main()
