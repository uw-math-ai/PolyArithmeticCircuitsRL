#!/usr/bin/env python3
"""Run one small search-distillation training round."""

from __future__ import annotations

import argparse
from random import Random

from decomp_rl.andor_search import AndOrSearch
from decomp_rl.baseline_cost import BaselineCostModel
from decomp_rl.evaluate import summarize_search_results
from decomp_rl.family_generators import planted_factorable_example
from decomp_rl.model import HeuristicPolicyValueModel
from decomp_rl.train_search_distill import distill_targets, make_distillation_training_examples
from decomp_rl.train_supervised import TorchTrainConfig, train_torch_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prime", type=int, default=3)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    rng = Random(args.seed)
    search = AndOrSearch(model=HeuristicPolicyValueModel(BaselineCostModel()))
    baseline_model = BaselineCostModel()
    targets = [
        planted_factorable_example(rng, args.prime, ("x", "y")).target
        for _ in range(args.count)
    ]
    baselines = [float(baseline_model.direct_construction_cost(target)) for target in targets]
    distilled = distill_targets(targets, search)
    before = summarize_search_results([search.search(target) for target in targets], baselines)

    training_examples = make_distillation_training_examples(distilled)
    result = train_torch_model(
        training_examples,
        TorchTrainConfig(epochs=args.epochs, seed=args.seed),
    )

    trained_search = AndOrSearch(model=result.wrapper)
    after = summarize_search_results([trained_search.search(target) for target in targets], baselines)
    print({"before": before.__dict__, "after": after.__dict__, "batch_size": result.batch_size_stats.__dict__})
    print({"epoch_history": [epoch.__dict__ for epoch in result.history]})


if __name__ == "__main__":
    main()
