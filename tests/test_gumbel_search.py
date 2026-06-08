import pytest
import torch

from lgs.data.target_generators import (
    make_common_factor_instance,
    make_product_of_sums_instance,
    make_square_instance,
)
from lgs.env.verification import verify_trace
from lgs.models.candidate_ranker import CandidateRanker
from lgs.models.feature_encoder import CandidateFeatureEncoder
from lgs.search.candidate_generator import generate_candidates
from lgs.search.beam_search import beam_search, recover_trace
from lgs.search.gumbel_search import gumbel_search
from lgs.search.search_history import history_expansion_count
from lgs.env.circuit_state import CircuitState
from lgs.training.preference_dataset import extract_preferences
from lgs.training.train_ranker import train_ranker_on_preferences


def easy_instances():
    return (
        make_square_instance(("x", "y"), field_p=17, degree_cap=4, op_budget=2),
        make_common_factor_instance(("a", "b", "c"), field_p=17, degree_cap=4, op_budget=2),
        make_product_of_sums_instance(
            ("a", "b"),
            ("c", "d"),
            field_p=17,
            degree_cap=4,
            op_budget=3,
        ),
    )


def train_tiny_ranker():
    preferences = []
    for instance in easy_instances():
        history = beam_search(instance, beam_width=8, candidate_k=32, tier2_m=128)
        assert history.success()
        preferences.extend(extract_preferences(history))
    encoder = CandidateFeatureEncoder()
    torch.manual_seed(0)
    ranker = CandidateRanker(
        input_dim=len(encoder.feature_names),
        hidden_dim=48,
        num_layers=3,
    )
    train_ranker_on_preferences(
        ranker,
        encoder,
        preferences,
        epochs=20,
        batch_size=16,
        seed=0,
    )
    return ranker, encoder


def test_gumbel_search_solves_easy_targets_with_exact_traces():
    for instance in easy_instances():
        history = gumbel_search(
            instance,
            candidate_k=64,
            tier2_m=128,
            initial_width=32,
            num_rounds=3,
            rollout_depth=1,
            seed=0,
        )
        best = history.best_finished()

        assert best is not None
        assert history.success()
        assert verify_trace(instance, recover_trace(best))


def test_gumbel_search_is_deterministic_for_same_seed():
    instance = make_square_instance(("x", "y"), field_p=17, degree_cap=4, op_budget=2)

    first = gumbel_search(instance, candidate_k=64, initial_width=16, seed=123)
    second = gumbel_search(instance, candidate_k=64, initial_width=16, seed=123)

    assert recover_trace(first.best_finished()) == recover_trace(second.best_finished())
    assert history_expansion_count(first) == history_expansion_count(second)


def test_different_seed_finished_traces_still_verify():
    instance = make_product_of_sums_instance(
        ("a", "b"),
        ("c", "d"),
        field_p=17,
        degree_cap=4,
        op_budget=3,
    )

    for seed in (0, 9):
        history = gumbel_search(
            instance,
            candidate_k=64,
            initial_width=24,
            rollout_depth=1,
            seed=seed,
        )
        for state in history.finished:
            assert verify_trace(instance, state.actions)


def test_multi_seed_gumbel_finished_traces_verify_exactly():
    instance = make_product_of_sums_instance(
        ("a", "b"),
        ("c", "d"),
        field_p=17,
        degree_cap=4,
        op_budget=3,
    )

    for seed in range(10):
        history = gumbel_search(
            instance,
            candidate_k=64,
            tier2_m=128,
            initial_width=16,
            rollout_depth=1,
            expansion_budget=64,
            seed=seed,
        )
        for state in history.finished:
            assert verify_trace(instance, state.actions)


def test_gumbel_search_requires_ranker_and_encoder_when_lambda_positive():
    instance = make_square_instance(("x", "y"), field_p=17, degree_cap=4, op_budget=2)
    encoder = CandidateFeatureEncoder()
    ranker = CandidateRanker(input_dim=len(encoder.feature_names), hidden_dim=8, num_layers=2)

    with pytest.raises(ValueError, match="ranker is required"):
        gumbel_search(instance, encoder=encoder, lambda_model=1.0)
    with pytest.raises(ValueError, match="encoder is required"):
        gumbel_search(instance, ranker=ranker, lambda_model=1.0)


def test_gumbel_search_respects_expansion_budget():
    instance = make_square_instance(("x", "y"), field_p=17, degree_cap=4, op_budget=2)

    history = gumbel_search(
        instance,
        candidate_k=64,
        initial_width=16,
        expansion_budget=3,
        seed=0,
    )

    assert history_expansion_count(history) <= 3
    assert len(history.records) <= history_expansion_count(history)


def test_actual_expansion_count_includes_rollout_expansions():
    instance = make_square_instance(("x", "y"), field_p=17, degree_cap=4, op_budget=2)

    no_rollout = gumbel_search(
        instance,
        candidate_k=64,
        tier2_m=128,
        initial_width=1,
        keep_per_round=1,
        num_rounds=1,
        rollout_depth=0,
        gumbel_scale=0.0,
        seed=0,
    )
    with_rollout = gumbel_search(
        instance,
        candidate_k=64,
        tier2_m=128,
        initial_width=1,
        keep_per_round=1,
        num_rounds=1,
        rollout_depth=1,
        gumbel_scale=0.0,
        seed=0,
    )

    assert history_expansion_count(no_rollout) == len(no_rollout.records)
    assert history_expansion_count(with_rollout) == len(with_rollout.records)
    assert history_expansion_count(with_rollout) > history_expansion_count(no_rollout)
    assert with_rollout.metadata["root_expansions"] > 0
    assert with_rollout.metadata["rollout_expansions"] > 0
    assert (
        with_rollout.metadata["root_expansions"]
        + with_rollout.metadata["rollout_expansions"]
        == history_expansion_count(with_rollout)
    )
    assert with_rollout.metadata["candidate_generation_calls"] > 0
    assert with_rollout.metadata["total_candidates_generated"] > 0


def test_rollout_found_finished_trace_verifies_exactly_under_budget():
    instance = make_square_instance(("x", "y"), field_p=17, degree_cap=4, op_budget=2)

    history = gumbel_search(
        instance,
        candidate_k=64,
        tier2_m=128,
        initial_width=1,
        keep_per_round=1,
        num_rounds=1,
        rollout_depth=1,
        gumbel_scale=0.0,
        seed=0,
        expansion_budget=2,
    )

    best = history.best_finished()
    assert best is not None
    assert history_expansion_count(history) == 2
    assert verify_trace(instance, best.actions)
    assert len(best.actions) == 2


class CountingRanker(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(0.0))
        self.calls = 0

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        raise AssertionError("ranker should not be called when lambda_model is zero")


def test_lambda_zero_does_not_call_ranker():
    instance = make_square_instance(("x", "y"), field_p=17, degree_cap=4, op_budget=2)
    ranker = CountingRanker()

    history = gumbel_search(
        instance,
        ranker=ranker,
        encoder=CandidateFeatureEncoder(),
        lambda_model=0.0,
        candidate_k=64,
    )

    assert history.success()
    assert ranker.calls == 0


def test_guided_gumbel_populates_model_scores_and_verifies_finished_trace():
    ranker, encoder = train_tiny_ranker()
    instance = make_square_instance(("u", "v"), field_p=17, degree_cap=4, op_budget=2)

    history = gumbel_search(
        instance,
        ranker=ranker,
        encoder=encoder,
        lambda_model=1.0,
        candidate_k=64,
        initial_width=24,
        rollout_depth=1,
        seed=0,
    )

    assert any(abs(record.candidate.model_score) > 1e-8 for record in history.records)
    best = history.best_finished()
    assert best is not None
    assert verify_trace(instance, best.actions)


def test_guided_gumbel_preserves_ranker_mode():
    ranker, encoder = train_tiny_ranker()
    instance = make_square_instance(("u", "v"), field_p=17, degree_cap=4, op_budget=2)

    ranker.train()
    gumbel_search(
        instance,
        ranker=ranker,
        encoder=encoder,
        lambda_model=1.0,
        candidate_k=64,
        initial_width=8,
        seed=0,
    )
    assert ranker.training

    ranker.eval()
    gumbel_search(
        instance,
        ranker=ranker,
        encoder=encoder,
        lambda_model=1.0,
        candidate_k=64,
        initial_width=8,
        seed=0,
    )
    assert not ranker.training


def test_gumbel_does_not_mutate_source_state_or_candidate_features():
    instance = make_square_instance(("x", "y"), field_p=17, degree_cap=4, op_budget=2)
    state = CircuitState.initial(instance)
    state_nodes = state.nodes
    state_actions = state.actions
    candidate = generate_candidates(instance, state, K=64, tier2_m=128)[0]
    feature_snapshot = dict(candidate.features)

    gumbel_search(
        instance,
        candidate_k=64,
        tier2_m=128,
        initial_width=8,
        seed=0,
    )

    assert state.nodes == state_nodes
    assert state.actions == state_actions
    assert candidate.features == feature_snapshot


def test_gumbel_history_supports_preference_extraction_when_trace_steps_recorded():
    instance = make_square_instance(("x", "y"), field_p=17, degree_cap=4, op_budget=2)
    history = gumbel_search(
        instance,
        candidate_k=64,
        tier2_m=128,
        initial_width=1,
        keep_per_round=1,
        num_rounds=1,
        rollout_depth=1,
        gumbel_scale=0.0,
        seed=0,
        expansion_budget=2,
    )

    assert history.success()
    preferences = extract_preferences(history)
    assert preferences
