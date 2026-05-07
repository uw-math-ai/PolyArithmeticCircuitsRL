#!/usr/bin/env python3
"""Generate a small supervised pretraining dataset."""

from __future__ import annotations

import argparse
from random import Random

from decomp_rl.family_generators import (
    pretraining_mixture,
)
from decomp_rl.train_supervised import save_examples_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts/pretrain_dataset.jsonl")
    parser.add_argument("--prime", type=int, default=3)
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = Random(args.seed)
    examples = pretraining_mixture(rng, args.prime, args.count, variables=("x", "y"))
    save_examples_jsonl(examples, args.output)
    print(f"Wrote {len(examples)} examples to {args.output}")


if __name__ == "__main__":
    main()
