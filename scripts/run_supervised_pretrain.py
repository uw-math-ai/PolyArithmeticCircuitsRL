#!/usr/bin/env python3
"""Train a small policy/value network on synthetic supervision."""

from __future__ import annotations

import argparse
from random import Random

from decomp_rl.evaluate import summarize_supervised
from decomp_rl.family_generators import pretraining_mixture
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
    parser.add_argument("--count", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=6)
    args = parser.parse_args()

    rng = Random(args.seed)
    examples = pretraining_mixture(rng, args.prime, args.count, variables=("x", "y"))
    training_examples = make_training_examples(examples)
    before = evaluate_training_examples(training_examples, build_default_model())
    result = train_torch_model(
        training_examples,
        TorchTrainConfig(epochs=args.epochs, seed=args.seed),
    )
    print(
        {
            "before": summarize_supervised(before),
            "after": summarize_supervised(result.final_metrics),
            "batch_size": result.batch_size_stats.__dict__,
        }
    )
    print({"epoch_history": [epoch.__dict__ for epoch in result.history]})


if __name__ == "__main__":
    main()
