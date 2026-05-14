import pytest
import torch

from lgs.env.problem_instance import ProblemInstance
from lgs.env.verification import verify_trace
from lgs.eval.compare_rankers import compare_heuristic_vs_ranker
from lgs.models.candidate_ranker import CandidateRanker
from lgs.models.feature_encoder import CandidateFeatureEncoder
from lgs.poly.fast_poly import FastPoly
from lgs.search.beam_search import beam_search, recover_trace
from lgs.training.preference_dataset import extract_preferences
from lgs.training.train_ranker import train_ranker_on_preferences


def make_square_instance(names=("x", "y")) -> ProblemInstance:
    x = FastPoly.variable(0, 2, 2, 17)
    y = FastPoly.variable(1, 2, 2, 17)
    return ProblemInstance(
        target=(x + y) * (x + y),
        variables=tuple(names),
        field_p=17,
        degree_cap=2,
        op_budget=2,
        family_name="guided_square",
    )


def make_common_factor_instance(names=("a", "b", "c")) -> ProblemInstance:
    a = FastPoly.variable(0, 3, 2, 17)
    b = FastPoly.variable(1, 3, 2, 17)
    c = FastPoly.variable(2, 3, 2, 17)
    return ProblemInstance(
        target=a * b + a * c,
        variables=tuple(names),
        field_p=17,
        degree_cap=2,
        op_budget=2,
        family_name="guided_common_factor",
    )


def make_product_of_sums_instance(names=("a", "b", "c", "d")) -> ProblemInstance:
    a = FastPoly.variable(0, 4, 2, 17)
    b = FastPoly.variable(1, 4, 2, 17)
    c = FastPoly.variable(2, 4, 2, 17)
    d = FastPoly.variable(3, 4, 2, 17)
    return ProblemInstance(
        target=(a + b) * (c + d),
        variables=tuple(names),
        field_p=17,
        degree_cap=2,
        op_budget=3,
        family_name="guided_product_of_sums",
    )


def train_tiny_ranker():
    preferences = []
    for instance in (
        make_square_instance(),
        make_common_factor_instance(),
        make_product_of_sums_instance(),
    ):
        history = beam_search(instance, beam_width=8, candidate_k=32)
        assert history.success()
        preferences.extend(extract_preferences(history))
    encoder = CandidateFeatureEncoder()
    torch.manual_seed(0)
    ranker = CandidateRanker(
        input_dim=len(encoder.feature_names),
        hidden_dim=64,
        num_layers=3,
        dropout=0.0,
    )
    train_ranker_on_preferences(
        ranker,
        encoder,
        preferences,
        epochs=80,
        lr=1e-3,
        batch_size=16,
        seed=0,
    )
    return ranker, encoder


def test_heuristic_only_backward_compatibility_with_lambda_zero():
    instance = make_square_instance()
    ranker, encoder = train_tiny_ranker()

    heuristic = beam_search(instance, beam_width=4, candidate_k=16)
    lambda_zero = beam_search(
        instance,
        ranker=ranker,
        encoder=encoder,
        lambda_model=0.0,
        beam_width=4,
        candidate_k=16,
    )

    assert heuristic.success()
    assert lambda_zero.success()
    assert recover_trace(heuristic.best_finished()) == recover_trace(lambda_zero.best_finished())


class CountingRanker(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(0.0))
        self.calls = 0

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        raise AssertionError("ranker should not be called when lambda_model is zero")


def test_lambda_zero_does_not_call_ranker():
    instance = make_square_instance()
    ranker = CountingRanker()

    history = beam_search(
        instance,
        ranker=ranker,
        encoder=CandidateFeatureEncoder(),
        lambda_model=0.0,
        beam_width=4,
        candidate_k=16,
    )

    assert history.success()
    assert ranker.calls == 0


def test_guided_search_preserves_ranker_mode():
    instance = make_square_instance()
    ranker, encoder = train_tiny_ranker()

    ranker.train()
    beam_search(
        instance,
        ranker=ranker,
        encoder=encoder,
        lambda_model=1.0,
        beam_width=4,
        candidate_k=16,
    )
    assert ranker.training

    ranker.eval()
    beam_search(
        instance,
        ranker=ranker,
        encoder=encoder,
        lambda_model=1.0,
        beam_width=4,
        candidate_k=16,
    )
    assert not ranker.training


def test_missing_ranker_or_encoder_validation():
    instance = make_square_instance()
    encoder = CandidateFeatureEncoder()
    ranker = CandidateRanker(input_dim=len(encoder.feature_names), hidden_dim=8, num_layers=2)

    with pytest.raises(ValueError, match="ranker is required"):
        beam_search(instance, encoder=encoder, lambda_model=1.0)
    with pytest.raises(ValueError, match="encoder is required"):
        beam_search(instance, ranker=ranker, lambda_model=1.0)


def test_model_scores_are_populated_in_history_records():
    instance = make_square_instance()
    ranker, encoder = train_tiny_ranker()
    lambda_model = 1.0

    history = beam_search(
        instance,
        ranker=ranker,
        encoder=encoder,
        lambda_model=lambda_model,
        beam_width=4,
        candidate_k=16,
    )

    scored = [record.candidate for record in history.records]
    assert any(abs(candidate.model_score) > 1e-8 for candidate in scored)
    for candidate in scored:
        expected = candidate.heuristic_score + lambda_model * candidate.model_score
        assert candidate.total_score == pytest.approx(expected)


def test_ranker_guided_search_solves_heldout_feature_isomorphic_targets():
    ranker, encoder = train_tiny_ranker()
    instances = (
        make_square_instance(("u", "v")),
        make_common_factor_instance(("u", "v", "w")),
        make_product_of_sums_instance(("p", "q", "r", "s")),
    )

    for instance in instances:
        history = beam_search(
            instance,
            ranker=ranker,
            encoder=encoder,
            lambda_model=1.0,
            beam_width=8,
            candidate_k=32,
        )
        best = history.best_finished()
        assert best is not None
        assert history.success()
        assert verify_trace(instance, recover_trace(best))


class ConstantHighRanker(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(1000.0))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.bias.expand(features.shape[0])


def test_exact_verification_remains_required_for_success():
    instance = make_square_instance()
    impossible = ProblemInstance(
        target=instance.target,
        variables=instance.variables,
        field_p=instance.field_p,
        degree_cap=instance.degree_cap,
        op_budget=1,
        family_name="guided_impossible_square",
    )

    history = beam_search(
        impossible,
        ranker=ConstantHighRanker(),
        encoder=CandidateFeatureEncoder(),
        lambda_model=1000.0,
        beam_width=8,
        candidate_k=32,
    )

    assert not history.success()
    assert all(state.contains(impossible.target) for state in history.finished)


def test_comparison_helper_returns_metrics():
    ranker, encoder = train_tiny_ranker()
    instances = [
        make_square_instance(("u", "v")),
        make_common_factor_instance(("u", "v", "w")),
    ]

    results = compare_heuristic_vs_ranker(
        instances,
        ranker,
        encoder,
        lambda_model=1.0,
        beam_width=4,
        candidate_k=16,
        tier2_m=128,
    )

    assert len(results) == len(instances)
    for result in results:
        assert result.heuristic_expansions >= 0
        assert result.guided_expansions >= 0
        assert result.heuristic_best_ops is None or isinstance(result.heuristic_best_ops, int)
        assert result.guided_best_ops is None or isinstance(result.guided_best_ops, int)


def test_comparison_helper_is_deterministic():
    ranker, encoder = train_tiny_ranker()
    instances = [
        make_square_instance(("u", "v")),
        make_common_factor_instance(("u", "v", "w")),
    ]

    first = compare_heuristic_vs_ranker(
        instances,
        ranker,
        encoder,
        lambda_model=1.0,
        beam_width=4,
        candidate_k=16,
        tier2_m=128,
    )
    second = compare_heuristic_vs_ranker(
        instances,
        ranker,
        encoder,
        lambda_model=1.0,
        beam_width=4,
        candidate_k=16,
        tier2_m=128,
    )

    assert first == second
