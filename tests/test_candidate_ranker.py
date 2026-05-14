import math

import torch

from lgs.env.problem_instance import ProblemInstance
from lgs.poly.fast_poly import FastPoly
from lgs.search.beam_search import beam_search
from lgs.training.preference_dataset import extract_preferences
from lgs.models.candidate_ranker import CandidateRanker
from lgs.models.feature_encoder import CandidateFeatureEncoder
from lgs.training.train_ranker import (
    load_ranker,
    pairwise_ranking_loss,
    save_ranker,
    train_ranker_on_preferences,
)


def make_square_instance() -> ProblemInstance:
    x = FastPoly.variable(0, 2, 2, 17)
    y = FastPoly.variable(1, 2, 2, 17)
    return ProblemInstance(
        target=(x + y) * (x + y),
        variables=("x", "y"),
        field_p=17,
        degree_cap=2,
        op_budget=2,
        family_name="ranker_square",
    )


def make_common_factor_instance() -> ProblemInstance:
    a = FastPoly.variable(0, 3, 2, 17)
    b = FastPoly.variable(1, 3, 2, 17)
    c = FastPoly.variable(2, 3, 2, 17)
    return ProblemInstance(
        target=a * b + a * c,
        variables=("a", "b", "c"),
        field_p=17,
        degree_cap=2,
        op_budget=2,
        family_name="ranker_common_factor",
    )


def make_product_of_sums_instance() -> ProblemInstance:
    a = FastPoly.variable(0, 4, 2, 17)
    b = FastPoly.variable(1, 4, 2, 17)
    c = FastPoly.variable(2, 4, 2, 17)
    d = FastPoly.variable(3, 4, 2, 17)
    return ProblemInstance(
        target=(a + b) * (c + d),
        variables=("a", "b", "c", "d"),
        field_p=17,
        degree_cap=2,
        op_budget=3,
        family_name="ranker_product_of_sums",
    )


def collect_preferences():
    preferences = []
    for instance in (
        make_square_instance(),
        make_common_factor_instance(),
        make_product_of_sums_instance(),
    ):
        history = beam_search(instance, beam_width=8, candidate_k=32)
        assert history.success()
        preferences.extend(extract_preferences(history))
    assert preferences
    return preferences


def test_feature_vector_is_deterministic_and_fixed_length():
    prefs = extract_preferences(beam_search(make_square_instance(), beam_width=4, candidate_k=16))
    pref = prefs[0]
    encoder = CandidateFeatureEncoder()

    first = encoder.encode(pref.instance, pref.state, pref.better)
    second = encoder.encode(pref.instance, pref.state, pref.better)

    assert first == second
    assert len(first) == len(encoder.feature_names)
    assert all(math.isfinite(value) for value in first)


def test_ranker_forward_shape():
    prefs = extract_preferences(beam_search(make_square_instance(), beam_width=4, candidate_k=16))
    encoder = CandidateFeatureEncoder()
    features = torch.tensor(
        [
            encoder.encode(pref.instance, pref.state, pref.better)
            for pref in prefs[:4]
        ],
        dtype=torch.float32,
    )
    ranker = CandidateRanker(input_dim=len(encoder.feature_names), hidden_dim=16, num_layers=2)

    scores = ranker(features)

    assert scores.shape == (features.shape[0],)


def test_pairwise_loss_behavior():
    near_zero = pairwise_ranking_loss(
        torch.tensor([3.0]),
        torch.tensor([0.0]),
        margin=1.0,
    )
    positive = pairwise_ranking_loss(
        torch.tensor([0.0]),
        torch.tensor([3.0]),
        margin=1.0,
    )
    weighted = pairwise_ranking_loss(
        torch.tensor([0.0, 3.0]),
        torch.tensor([3.0, 0.0]),
        weights=torch.tensor([2.0, 0.5]),
        margin=1.0,
    )

    assert near_zero.item() == 0.0
    assert positive.item() > 0.0
    assert weighted.ndim == 0
    assert weighted.item() > 0.0


def test_training_decreases_loss_and_overfits_small_preference_dataset():
    torch.manual_seed(0)
    preferences = collect_preferences()
    encoder = CandidateFeatureEncoder()
    ranker = CandidateRanker(
        input_dim=len(encoder.feature_names),
        hidden_dim=64,
        num_layers=3,
        dropout=0.0,
    )

    history = train_ranker_on_preferences(
        ranker,
        encoder,
        preferences,
        epochs=120,
        lr=1e-3,
        batch_size=16,
        seed=0,
    )

    assert history["loss"][-1] < history["loss"][0]
    assert history["accuracy"][-1] > 0.8


def test_save_load_checkpoint_roundtrip(tmp_path):
    torch.manual_seed(0)
    prefs = extract_preferences(beam_search(make_square_instance(), beam_width=4, candidate_k=16))
    encoder = CandidateFeatureEncoder()
    ranker = CandidateRanker(input_dim=len(encoder.feature_names), hidden_dim=16, num_layers=2)
    ranker.eval()
    features = torch.tensor(
        [
            encoder.encode(pref.instance, pref.state, pref.better)
            for pref in prefs[:4]
        ],
        dtype=torch.float32,
    )
    before = ranker(features)
    path = tmp_path / "ranker.pt"

    save_ranker(path, ranker, encoder)
    loaded_ranker, loaded_encoder = load_ranker(path)
    loaded_ranker.eval()
    after = loaded_ranker(features)

    assert loaded_encoder.feature_names == encoder.feature_names
    assert torch.allclose(before, after)


def test_beam_search_does_not_import_candidate_ranker():
    import lgs.search.beam_search as beam_search_module

    assert "CandidateRanker" not in beam_search_module.__dict__
