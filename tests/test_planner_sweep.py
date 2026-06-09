import torch

from lgs.data.target_generators import (
    make_common_factor_instance,
    make_product_of_sums_instance,
    make_square_instance,
)
from lgs.env.verification import verify_trace
from lgs.eval.planner_sweep import (
    PlannerSweepConfig,
    PlannerSweepRow,
    run_planner_sweep,
    summarize_planner_deltas,
    summarize_planner_sweep,
)
from lgs.models.candidate_ranker import CandidateRanker
from lgs.models.feature_encoder import CandidateFeatureEncoder
from lgs.search.beam_search import beam_search
from lgs.search.gumbel_search import gumbel_search


def tiny_instances():
    return [
        make_square_instance(("x", "y"), field_p=17, degree_cap=6, op_budget=2),
        make_common_factor_instance(("a", "b", "c"), field_p=17, degree_cap=6, op_budget=2),
    ]


def tiny_ranker():
    encoder = CandidateFeatureEncoder()
    torch.manual_seed(0)
    ranker = CandidateRanker(
        input_dim=len(encoder.feature_names),
        hidden_dim=16,
        num_layers=2,
    )
    ranker.eval()
    return ranker, encoder


def small_config(**overrides):
    values = {
        "beam_widths": (1,),
        "candidate_ks": (8,),
        "tier2_ms": (16,),
        "expansion_budgets": (32,),
        "gumbel_seeds": (0,),
        "gumbel_initial_width": 4,
        "gumbel_rounds": 2,
        "rollout_depth": 1,
        "include_guided": False,
    }
    values.update(overrides)
    return PlannerSweepConfig(**values)


def test_planner_sweep_heuristic_only_rows():
    rows = run_planner_sweep(
        tiny_instances(),
        config=small_config(include_guided=False),
        benchmark_name="tiny",
    )

    assert {row.method for row in rows} == {"beam_heuristic", "gumbel_heuristic"}
    assert len(rows) == 4
    for row in rows:
        assert row.benchmark_name == "tiny"
        assert row.instance_id
        assert row.family
        assert row.expansion_budget == 32
        assert row.expansions <= 32
        assert row.runtime_sec >= 0.0
        if row.planner == "beam":
            assert row.seed is None
        else:
            assert row.seed == 0


def test_planner_sweep_guided_includes_four_methods():
    ranker, encoder = tiny_ranker()

    rows = run_planner_sweep(
        tiny_instances()[:1],
        ranker=ranker,
        encoder=encoder,
        config=small_config(include_guided=True, lambda_model=1.0),
    )

    assert {row.method for row in rows} == {
        "beam_heuristic",
        "beam_guided",
        "gumbel_heuristic",
        "gumbel_guided",
    }
    guided_rows = [row for row in rows if row.guided]
    assert guided_rows
    assert all(row.lambda_model == 1.0 for row in guided_rows)
    assert any((row.model_forward_calls or 0) > 0 for row in guided_rows)


def test_planner_sweep_requires_ranker_for_guided_rows():
    try:
        run_planner_sweep(tiny_instances()[:1], config=small_config(include_guided=True))
    except ValueError as exc:
        assert "ranker and encoder" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_planner_sweep_preserves_gumbel_seed_rows():
    rows = run_planner_sweep(
        tiny_instances()[:1],
        config=small_config(include_guided=False, gumbel_seeds=(0, 1, 2)),
    )

    gumbel_rows = [row for row in rows if row.planner == "gumbel"]
    beam_rows = [row for row in rows if row.planner == "beam"]
    assert [row.seed for row in gumbel_rows] == [0, 1, 2]
    assert all(row.seed is None for row in beam_rows)


def test_planner_sweep_summary_groups_by_family_complexity_and_budget():
    rows = run_planner_sweep(
        tiny_instances(),
        config=small_config(include_guided=False, gumbel_seeds=(0, 1)),
    )

    summary = summarize_planner_sweep(rows)

    assert summary
    for row in summary:
        assert row["family"]
        assert row["intended_complexity"] is not None
        assert row["expansion_budget"] == 32
        assert 0.0 <= row["success_rate"] <= 1.0
        assert row["avg_expansions"] >= 0.0
        if row["planner"] == "gumbel":
            assert row["num_seeds"] == 2


def test_planner_sweep_deltas_match_only_same_instance_budget_groups():
    rows = [
        _row("beam_heuristic", "beam", False, "a1", budget=32, success=False, lambda_model=0.0),
        _row("beam_guided", "beam", True, "a1", budget=32, success=True, lambda_model=1.0),
        _row(
            "beam_heuristic",
            "beam",
            False,
            "b1",
            budget=32,
            success=True,
            lambda_model=0.0,
            family="other",
        ),
        _row(
            "beam_guided",
            "beam",
            True,
            "b1",
            budget=64,
            success=False,
            lambda_model=1.0,
            family="other",
        ),
        _row("gumbel_heuristic", "gumbel", False, "a1", budget=32, success=False, lambda_model=0.0, seed=0),
        _row("gumbel_guided", "gumbel", True, "a1", budget=32, success=True, lambda_model=1.0, seed=0),
    ]

    deltas = summarize_planner_deltas(rows)
    comparisons = {row["comparison"]: row for row in deltas}

    assert "beam_guided - beam_heuristic" in comparisons
    assert comparisons["beam_guided - beam_heuristic"]["solve_rate_delta"] == 1.0
    assert comparisons["beam_guided - beam_heuristic"]["base_lambda_model"] == 0.0
    assert comparisons["beam_guided - beam_heuristic"]["candidate_lambda_model"] == 1.0
    assert "gumbel_guided - gumbel_heuristic" in comparisons
    assert all(row["instance_set_size"] == 1 for row in deltas)


def test_planner_finished_traces_verify_exactly():
    instances = [
        make_square_instance(("x", "y"), field_p=17, degree_cap=6, op_budget=2),
        make_product_of_sums_instance(
            ("a", "b"),
            ("c", "d"),
            field_p=17,
            degree_cap=6,
            op_budget=3,
        ),
    ]

    for instance in instances:
        beam_history = beam_search(
            instance,
            beam_width=4,
            candidate_k=32,
            tier2_m=64,
            expansion_budget=64,
        )
        gumbel_history = gumbel_search(
            instance,
            candidate_k=32,
            tier2_m=64,
            initial_width=8,
            num_rounds=3,
            rollout_depth=1,
            expansion_budget=64,
            seed=0,
        )
        for history in (beam_history, gumbel_history):
            best = history.best_finished()
            if best is not None:
                assert verify_trace(instance, best.actions)


def _row(
    method,
    planner,
    guided,
    instance_id,
    *,
    budget,
    success,
    lambda_model,
    seed=None,
    family="fam",
):
    return PlannerSweepRow(
        method=method,
        planner=planner,
        guided=guided,
        benchmark_name="bench",
        instance_id=instance_id,
        family=family,
        intended_complexity=2,
        generative_ops=None,
        beam_width=1,
        candidate_k=8,
        tier2_m=16,
        expansion_budget=budget,
        lambda_model=lambda_model,
        seed=seed,
        success=success,
        best_ops=2 if success else None,
        expansions=10,
        runtime_sec=0.1,
    )
