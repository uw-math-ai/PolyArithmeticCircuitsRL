#!/usr/bin/env python3
"""Run a lightweight end-to-end training smoke test."""

from __future__ import annotations

import argparse
from random import Random

from decomp_rl.andor_search import AndOrSearch
from decomp_rl.baseline_cost import BaselineCostModel
from decomp_rl.evaluate import summarize_search_results, summarize_supervised
from decomp_rl.family_generators import planted_factorable_example, pretraining_mixture
from decomp_rl.model import HeuristicPolicyValueModel
from decomp_rl.train_search_distill import distill_targets, make_distillation_training_examples
from decomp_rl.train_supervised import (
    TorchTrainConfig,
    build_default_model,
    evaluate_training_examples,
    make_training_examples,
    train_torch_model,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prime", type=int, default=3)
    parser.add_argument("--supervised-count", type=int, default=18)
    parser.add_argument("--distill-count", type=int, default=6)
    parser.add_argument("--supervised-epochs", type=int, default=6)
    parser.add_argument("--distill-epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = Random(args.seed)
    supervised_examples = pretraining_mixture(rng, args.prime, args.supervised_count, variables=("x", "y"))
    supervised_training = make_training_examples(supervised_examples)
    heuristic_metrics = evaluate_training_examples(supervised_training, build_default_model())
    supervised_result = train_torch_model(
        supervised_training,
        TorchTrainConfig(epochs=args.supervised_epochs, seed=args.seed),
    )

    distill_targets_list = [
        planted_factorable_example(rng, args.prime, ("x", "y")).target
        for _ in range(args.distill_count)
    ]
    baseline_model = BaselineCostModel()
    baselines = [float(baseline_model.direct_construction_cost(target)) for target in distill_targets_list]

    search_before = AndOrSearch(model=HeuristicPolicyValueModel(BaselineCostModel()))
    distill_examples = distill_targets(distill_targets_list, search_before)
    distill_training = make_distillation_training_examples(distill_examples)

    pre_distill_search = AndOrSearch(model=supervised_result.wrapper)
    pre_distill_summary = summarize_search_results(
        [pre_distill_search.search(target) for target in distill_targets_list],
        baselines,
    )

    distill_result = train_torch_model(
        distill_training,
        TorchTrainConfig(epochs=args.distill_epochs, seed=args.seed),
        network=supervised_result.network,
    )
    post_distill_search = AndOrSearch(model=distill_result.wrapper)
    post_distill_summary = summarize_search_results(
        [post_distill_search.search(target) for target in distill_targets_list],
        baselines,
    )

    print(
        {
            "supervised_before": summarize_supervised(heuristic_metrics),
            "supervised_after": summarize_supervised(supervised_result.final_metrics),
            "supervised_batch_size": supervised_result.batch_size_stats.__dict__,
            "search_before_distill": pre_distill_summary.__dict__,
            "search_after_distill": post_distill_summary.__dict__,
            "distill_batch_size": distill_result.batch_size_stats.__dict__,
        }
    )
    print({"supervised_epochs": [epoch.__dict__ for epoch in supervised_result.history]})
    print({"distill_epochs": [epoch.__dict__ for epoch in distill_result.history]})


if __name__ == "__main__":
    main()
