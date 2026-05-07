from random import Random

import pytest

from decomp_rl.family_generators import (
    elementary_symmetric_example,
    horner_example,
    planted_factorable_example,
)
from decomp_rl.train_supervised import (
    TorchTrainConfig,
    build_default_model,
    evaluate_training_examples,
    make_training_examples,
    train_torch_model,
)


def test_torch_training_smoke():
    torch = pytest.importorskip("torch")
    assert torch is not None

    rng = Random(0)
    examples = [
        planted_factorable_example(rng, 3, ("x", "y")),
        horner_example([1, 2, 0, 1], 3),
        elementary_symmetric_example(variable_count=4, degree=2, prime=3),
    ]
    training_examples = make_training_examples(examples)
    before = evaluate_training_examples(training_examples, build_default_model())
    result = train_torch_model(
        training_examples,
        TorchTrainConfig(epochs=1, seed=0),
    )

    assert result.history
    assert result.final_metrics.example_count == 3
    assert result.final_metrics.average_policy_loss <= before.average_policy_loss
    assert result.batch_size_stats.resolved_batch_size == 3
    assert result.batch_size_stats.device in {"cpu", "cuda"}
