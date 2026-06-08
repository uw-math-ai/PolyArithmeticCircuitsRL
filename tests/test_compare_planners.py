from pathlib import Path

from lgs.data.target_generators import (
    make_common_factor_instance,
    make_square_instance,
)
import lgs.eval.compare_planners as compare_planners
from lgs.eval.compare_planners import compare_beam_vs_gumbel
from lgs.search.search_history import SearchHistory


def tiny_instances():
    return [
        make_square_instance(("x", "y"), field_p=17, degree_cap=4, op_budget=2),
        make_common_factor_instance(("a", "b", "c"), field_p=17, degree_cap=4, op_budget=2),
    ]


def test_planner_comparison_returns_beam_and_gumbel_rows():
    rows = compare_beam_vs_gumbel(
        tiny_instances(),
        beam_width=4,
        candidate_k=32,
        tier2_m=128,
        gumbel_initial_width=16,
        gumbel_rounds=3,
        rollout_depth=1,
        seed=0,
    )

    assert {row.planner for row in rows} == {"beam", "gumbel"}
    assert len([row for row in rows if row.planner == "beam"]) == 2
    assert len([row for row in rows if row.planner == "gumbel"]) == 2
    for row in rows:
        assert row.family
        assert row.expansions >= 0
        assert row.runtime_sec >= 0.0
        if row.planner == "beam":
            assert row.seed is None
        else:
            assert row.seed == 0
        if row.success:
            assert isinstance(row.best_ops, int)
        else:
            assert row.best_ops is None


def test_planner_comparison_applies_shared_expansion_budget():
    rows = compare_beam_vs_gumbel(
        tiny_instances(),
        beam_width=4,
        candidate_k=32,
        tier2_m=128,
        gumbel_initial_width=16,
        gumbel_rounds=3,
        rollout_depth=1,
        expansion_budget=2,
        seed=0,
    )

    assert rows
    assert all(row.expansions <= 2 for row in rows)


def test_new_gumbel_modules_do_not_import_sibling_gumbel():
    repo_root = Path(__file__).resolve().parents[1]
    for relative_path in (
        "src/lgs/search/gumbel_search.py",
        "src/lgs/eval/compare_planners.py",
    ):
        text = (repo_root / relative_path).read_text(encoding="utf-8")
        for line in text.splitlines():
            stripped = line.strip()
            assert not stripped.startswith("from gumbel")
            assert not stripped.startswith("import gumbel")
        assert "../gumbel" not in text


def test_planner_comparison_uses_explicit_history_expansion_count(monkeypatch):
    instance = tiny_instances()[0]

    def fake_beam_search(*args, **kwargs):
        del args, kwargs
        return SearchHistory(instance=instance, records=[], finished=[], num_expansions=5)

    def fake_gumbel_search(*args, **kwargs):
        del args, kwargs
        return SearchHistory(instance=instance, records=[], finished=[], num_expansions=7)

    monkeypatch.setattr(compare_planners, "beam_search", fake_beam_search)
    monkeypatch.setattr(compare_planners, "gumbel_search", fake_gumbel_search)

    rows = compare_beam_vs_gumbel([instance])

    assert [row.expansions for row in rows] == [5, 7]
