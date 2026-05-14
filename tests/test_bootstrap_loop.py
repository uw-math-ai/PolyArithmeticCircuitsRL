from pathlib import Path

import pytest

from lgs.data.curriculum import FixedCurriculum
from lgs.data.target_generators import (
    make_tiny_train_instances,
    make_tiny_validation_instances,
)
from lgs.env.circuit_state import CircuitState
from lgs.env.verification import verify_trace
from lgs.eval.evaluate_search import SearchEvalMetrics, evaluate_beam_search
from lgs.poly.fast_poly import Polynomial
from lgs.search.beam_search import beam_search
import lgs.training.bootstrap_loop as bootstrap_loop
from lgs.training.preference_dataset import extract_preferences as real_extract_preferences
from lgs.training.bootstrap_loop import (
    BootstrapConfig,
    next_lambda,
    run_bootstrap_training,
    should_promote_lambda,
)


def make_curriculum() -> FixedCurriculum:
    return FixedCurriculum(
        train_instances=make_tiny_train_instances(field_p=17, degree_cap=2),
        validation_instances=make_tiny_validation_instances(field_p=17, degree_cap=2),
    )


def make_fast_config() -> BootstrapConfig:
    return BootstrapConfig(
        num_rounds=1,
        beam_width=4,
        candidate_k=16,
        tier2_m=64,
        epochs_per_round=5,
        batch_size=16,
        seed=0,
        lambda_values=(0.0, 0.25),
    )


def test_target_generators_produce_verifiable_instances():
    instances = [
        *make_tiny_train_instances(field_p=17, degree_cap=2),
        *make_tiny_validation_instances(field_p=17, degree_cap=2),
    ]

    for instance in instances:
        assert isinstance(instance.target, Polynomial)
        assert instance.op_budget > 0
        state = CircuitState.initial(instance)
        assert state.num_nodes() == len(instance.variables) + 1
        beam_search(instance, beam_width=4, candidate_k=16, tier2_m=64)


def test_evaluation_utility_works():
    instances = make_tiny_validation_instances(field_p=17, degree_cap=2)

    metrics = evaluate_beam_search(
        instances,
        beam_width=4,
        candidate_k=16,
        tier2_m=64,
    )

    assert metrics.num_instances == len(instances)
    assert 0.0 <= metrics.success_rate <= 1.0
    assert metrics.avg_expansions >= 0.0
    if metrics.success_rate == 0.0:
        assert metrics.avg_best_ops is None
    else:
        assert metrics.avg_best_ops is not None


def test_bootstrap_loop_runs_one_round():
    result = run_bootstrap_training(make_curriculum(), make_fast_config())

    assert len(result.metrics) == 1
    assert result.metrics[0].total_preferences > 0
    assert result.metrics[0].num_preferences_added > 0
    assert result.ranker is not None
    assert result.encoder is not None


def test_bootstrap_loop_metrics_are_deterministic():
    first = run_bootstrap_training(make_curriculum(), make_fast_config())
    second = run_bootstrap_training(make_curriculum(), make_fast_config())

    first_metric = first.metrics[0]
    second_metric = second.metrics[0]
    assert first_metric.num_preferences_added == second_metric.num_preferences_added
    assert first_metric.total_preferences == second_metric.total_preferences
    assert first_metric.heuristic_val_success_rate == second_metric.heuristic_val_success_rate
    assert first_metric.guided_val_success_rate == second_metric.guided_val_success_rate
    assert first.final_lambda_model == second.final_lambda_model
    assert first_metric.train_loss_final == pytest.approx(second_metric.train_loss_final)
    assert first_metric.train_accuracy_final == pytest.approx(
        second_metric.train_accuracy_final
    )


def test_lambda_promotion_is_gated():
    heuristic = SearchEvalMetrics(
        success_rate=0.5,
        avg_best_ops=2.0,
        avg_expansions=10.0,
        num_instances=2,
    )

    guided_worse = SearchEvalMetrics(
        success_rate=0.0,
        avg_best_ops=None,
        avg_expansions=5.0,
        num_instances=2,
    )
    guided_equal_less_work = SearchEvalMetrics(
        success_rate=0.5,
        avg_best_ops=2.0,
        avg_expansions=10.0,
        num_instances=2,
    )
    guided_better = SearchEvalMetrics(
        success_rate=1.0,
        avg_best_ops=2.0,
        avg_expansions=50.0,
        num_instances=2,
    )

    assert not should_promote_lambda(heuristic, guided_worse)
    assert should_promote_lambda(heuristic, guided_equal_less_work)
    assert should_promote_lambda(heuristic, guided_better)
    assert next_lambda(0.0, (0.0, 0.25, 0.5)) == 0.25
    assert next_lambda(0.5, (0.0, 0.25, 0.5)) == 0.5


def test_new_modules_do_not_import_gumbel():
    module_paths = [
        Path("src/lgs/training/bootstrap_loop.py"),
        Path("src/lgs/data/target_generators.py"),
        Path("src/lgs/data/curriculum.py"),
        Path("src/lgs/eval/evaluate_search.py"),
        Path("scripts/train_bootstrap.py"),
    ]

    for path in module_paths:
        assert "gumbel" not in path.read_text()


def test_bootstrap_does_not_extract_validation_preferences_and_guided_traces_verify(
    monkeypatch,
):
    curriculum = make_curriculum()
    config = make_fast_config()
    extraction_ids: list[str] = []

    def recording_extract_preferences(history, delta=1.0):
        extraction_ids.append(history.instance.metadata["target_id"])
        return real_extract_preferences(history, delta=delta)

    monkeypatch.setattr(
        bootstrap_loop,
        "extract_preferences",
        recording_extract_preferences,
    )

    result = run_bootstrap_training(curriculum, config)
    train_ids = {instance.metadata["target_id"] for instance in curriculum.train_instances}
    validation_ids = {
        instance.metadata["target_id"]
        for instance in curriculum.validation_instances
    }

    assert extraction_ids
    assert set(extraction_ids) <= train_ids
    assert not set(extraction_ids) & validation_ids
    assert result.final_lambda_model in config.lambda_values

    for instance in curriculum.validation_instances:
        history = beam_search(
            instance,
            ranker=result.ranker,
            encoder=result.encoder,
            lambda_model=result.final_lambda_model,
            beam_width=config.beam_width,
            candidate_k=config.candidate_k,
            tier2_m=config.tier2_m,
        )
        for finished in history.finished:
            assert verify_trace(instance, finished.actions)
