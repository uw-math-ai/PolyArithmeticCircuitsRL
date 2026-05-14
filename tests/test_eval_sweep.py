import torch

from lgs.data.benchmark_suite import make_structured_benchmark
from lgs.data.target_generators import make_tiny_train_instances
from lgs.eval.sweep import (
    SweepConfig,
    SweepRow,
    run_search_sweep,
    summarize_failures,
    summarize_sweep,
)
from lgs.models.candidate_ranker import CandidateRanker
from lgs.models.feature_encoder import CandidateFeatureEncoder
from lgs.search.beam_search import beam_search
from lgs.training.preference_dataset import extract_preferences
from lgs.training.train_ranker import train_ranker_on_preferences


def tiny_sweep_instances():
    return make_structured_benchmark(
        field_p=17,
        degree_cap=8,
        max_instances_per_family=1,
    ).instances[:3]


def train_tiny_ranker():
    preferences = []
    for instance in make_tiny_train_instances(field_p=17, degree_cap=8):
        history = beam_search(instance, beam_width=4, candidate_k=16, tier2_m=64)
        preferences.extend(extract_preferences(history))
    encoder = CandidateFeatureEncoder()
    torch.manual_seed(0)
    ranker = CandidateRanker(
        input_dim=len(encoder.feature_names),
        hidden_dim=32,
        num_layers=2,
    )
    train_ranker_on_preferences(
        ranker,
        encoder,
        preferences,
        epochs=5,
        batch_size=16,
        seed=0,
    )
    return ranker, encoder


def test_heuristic_sweep_produces_rows():
    rows = run_search_sweep(
        tiny_sweep_instances(),
        ranker=None,
        encoder=None,
        config=SweepConfig(
            beam_widths=(1,),
            candidate_ks=(4,),
            tier2_ms=(16,),
        ),
    )

    assert rows
    assert {row.method for row in rows} == {"heuristic"}
    for row in rows:
        assert row.instance_id
        assert row.family
        assert isinstance(row.success, bool)
        assert row.expansions >= 0
        assert row.runtime_sec >= 0.0


def test_summarize_sweep_returns_aggregate_metrics():
    rows = run_search_sweep(
        tiny_sweep_instances(),
        ranker=None,
        encoder=None,
        config=SweepConfig(
            beam_widths=(1,),
            candidate_ks=(4,),
            tier2_ms=(16,),
        ),
    )

    summary = summarize_sweep(rows)

    assert summary
    for row in summary:
        assert 0.0 <= row["solve_rate"] <= 1.0
        assert row["avg_expansions"] >= 0.0
        assert row["median_expansions"] >= 0.0
        assert row["avg_runtime_sec"] >= 0.0


def test_summarize_sweep_delta_matches_family_budget_groups():
    rows = [
        SweepRow("heuristic", "b", "a1", "fam_a", 1, 4, 16, True, 2, 10, 0.1, 2),
        SweepRow("heuristic", "b", "a2", "fam_a", 1, 4, 16, False, None, 11, 0.1, 3),
        SweepRow("guided", "b", "a1", "fam_a", 1, 4, 16, True, 2, 10, 0.1, 2),
        SweepRow("guided", "b", "a2", "fam_a", 1, 4, 16, True, 3, 11, 0.1, 3),
        SweepRow("heuristic", "b", "b1", "fam_b", 1, 4, 16, True, 2, 10, 0.1, 2),
        SweepRow("guided", "b", "b1", "fam_b", 2, 4, 16, False, None, 10, 0.1, 2),
    ]

    summary = summarize_sweep(rows)
    fam_a_guided = [
        row
        for row in summary
        if row["family"] == "fam_a" and row["method"] == "guided"
    ][0]
    fam_b_guided = [
        row
        for row in summary
        if row["family"] == "fam_b" and row["method"] == "guided"
    ][0]

    assert fam_a_guided["guided_minus_heuristic_solve_rate"] == 0.5
    assert fam_b_guided["guided_minus_heuristic_solve_rate"] is None


def test_summarize_failures_groups_failed_rows():
    rows = [
        SweepRow("heuristic", "b", "a1", "fam_a", 1, 4, 16, False, None, 10, 0.1, 2),
        SweepRow("heuristic", "b", "a2", "fam_a", 1, 4, 16, False, None, 12, 0.1, 2),
        SweepRow("guided", "b", "a3", "fam_a", 1, 4, 16, True, 2, 8, 0.1, 2),
    ]

    failures = summarize_failures(rows)

    assert failures == [
        {
            "family": "fam_a",
            "intended_complexity": 2,
            "method": "heuristic",
            "beam_width": 1,
            "candidate_k": 4,
            "tier2_m": 16,
            "failure_count": 2,
            "avg_expansions": 11.0,
            "instance_ids": ["a1", "a2"],
        }
    ]


def test_guided_sweep_includes_heuristic_and_guided_rows():
    ranker, encoder = train_tiny_ranker()

    rows = run_search_sweep(
        tiny_sweep_instances()[:2],
        ranker=ranker,
        encoder=encoder,
        config=SweepConfig(
            beam_widths=(1,),
            candidate_ks=(4,),
            tier2_ms=(16,),
            lambda_model=1.0,
        ),
    )

    assert {row.method for row in rows} == {"heuristic", "guided"}
    assert len([row for row in rows if row.method == "heuristic"]) == 2
    assert len([row for row in rows if row.method == "guided"]) == 2
