#!/usr/bin/env python3
"""Run a tiny end-to-end symbolic evaluation suite."""

from __future__ import annotations

from random import Random

from decomp_rl.andor_search import AndOrSearch
from decomp_rl.baseline_cost import BaselineCostModel
from decomp_rl.family_generators import (
    elementary_symmetric_example,
    horner_example,
    planted_factorable_example,
)


def main() -> None:
    rng = Random(0)
    search = AndOrSearch()
    baseline = BaselineCostModel()
    examples = [
        planted_factorable_example(rng, 3, ("x", "y")),
        horner_example([1, 0, 2, 1], 3),
        elementary_symmetric_example(4, 2, 3),
    ]
    for example in examples:
        result = search.search(example.target)
        before = baseline.direct_construction_cost(example.target)
        print(
            {
                "family": example.family,
                "baseline_cost": before,
                "best_cost": result.best_cost,
                "root_value": result.root_value,
            }
        )


if __name__ == "__main__":
    main()
