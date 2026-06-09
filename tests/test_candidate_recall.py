from lgs.data.target_generators import (
    make_common_factor_instance,
    make_product_of_sums_instance,
    make_square_instance,
)
from lgs.eval.candidate_recall import (
    compute_candidate_recall_for_trace,
    compute_recall_for_best_finished,
)
from lgs.search.beam_search import beam_search


def test_square_target_best_trace_recall_at_64():
    instance = make_square_instance(
        ("x", "y"),
        field_p=17,
        degree_cap=4,
        op_budget=2,
    )
    history = beam_search(instance, beam_width=4, candidate_k=64, tier2_m=128)
    assert history.success()

    report = compute_recall_for_best_finished(
        history,
        candidate_ks=(64,),
        tier2_m=128,
    )

    assert report is not None
    assert report.recall_at(64) == 1.0
    assert report.mean_rank_at(64) is not None
    assert all(step.rank is not None for step in report.steps if step.k == 64)


def test_common_factor_best_trace_recall_at_64():
    instance = make_common_factor_instance(
        ("a", "b", "c"),
        field_p=17,
        degree_cap=4,
        op_budget=2,
    )
    history = beam_search(instance, beam_width=4, candidate_k=64, tier2_m=128)
    assert history.success()

    report = compute_candidate_recall_for_trace(
        instance,
        tuple(history.best_finished().actions),
        candidate_ks=(64,),
        tier2_m=128,
    )

    assert report.recall_at(64) == 1.0
    assert report.misses_at(64) == []


def test_product_of_sums_best_trace_recall_at_64():
    instance = make_product_of_sums_instance(
        ("a", "b"),
        ("c", "d"),
        field_p=17,
        degree_cap=4,
        op_budget=3,
    )
    history = beam_search(instance, beam_width=8, candidate_k=64, tier2_m=128)
    assert history.success()

    report = compute_recall_for_best_finished(
        history,
        candidate_ks=(64,),
        tier2_m=128,
    )

    assert report is not None
    assert report.recall_at(64) == 1.0
    assert report.mean_rank_at(64) is not None


def test_recall_for_unsolved_history_returns_none():
    instance = make_square_instance(
        ("x", "y"),
        field_p=17,
        degree_cap=4,
        op_budget=1,
    )
    history = beam_search(instance, beam_width=4, candidate_k=64, tier2_m=128)

    assert not history.success()
    assert compute_recall_for_best_finished(history, candidate_ks=(64,)) is None
