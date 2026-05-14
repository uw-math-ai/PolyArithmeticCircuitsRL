import pytest

from lgs.env.action import Action
from lgs.env.candidate import Candidate
from lgs.env.circuit_state import CircuitState
from lgs.env.problem_instance import ProblemInstance
from lgs.poly.fast_poly import FastPoly
from lgs.search.beam_search import beam_search
from lgs.search.search_history import ExpandedStateRecord, SearchHistory
from lgs.training.preference_dataset import (
    extract_preferences,
    serialize_preference,
)


def make_square_instance(op_budget: int = 2) -> ProblemInstance:
    x = FastPoly.variable(0, 2, 2, 17)
    y = FastPoly.variable(1, 2, 2, 17)
    return ProblemInstance(
        target=(x + y) * (x + y),
        variables=("x", "y"),
        field_p=17,
        degree_cap=2,
        op_budget=op_budget,
        family_name="test_square_preferences",
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
        family_name="test_common_factor_preferences",
    )


def test_square_target_preferences_include_best_trace_actions():
    instance = make_square_instance()
    history = beam_search(instance, beam_width=4, candidate_k=16)

    prefs = extract_preferences(history, delta=1.0)

    assert prefs
    assert any(
        pref.state.actions == ()
        and pref.better.action == Action.make("add", 0, 1)
        for pref in prefs
    )
    assert any(
        pref.state.actions == (Action.make("add", 0, 1),)
        and pref.better.action == Action.make("mul", 3, 3)
        for pref in prefs
    )
    assert all(pref.return_better > pref.return_worse + 1.0 for pref in prefs)
    assert all(pref.weight > 0 for pref in prefs)


def test_common_factor_preferences_include_best_trace_actions():
    instance = make_common_factor_instance()
    history = beam_search(instance, beam_width=4, candidate_k=16)

    prefs = extract_preferences(history)

    assert prefs
    assert any(pref.better.action == Action.make("add", 1, 2) for pref in prefs)
    assert any(pref.better.action == Action.make("mul", 0, 4) for pref in prefs)


def test_no_success_gives_no_preferences():
    instance = make_square_instance(op_budget=1)
    history = beam_search(instance, beam_width=4, candidate_k=16)

    assert not history.success()
    assert extract_preferences(history) == []


def test_preference_extraction_is_deterministic():
    instance = make_common_factor_instance()
    first = beam_search(instance, beam_width=4, candidate_k=16)
    second = beam_search(instance, beam_width=4, candidate_k=16)

    first_summary = [serialize_preference(pref) for pref in extract_preferences(first)]
    second_summary = [serialize_preference(pref) for pref in extract_preferences(second)]

    assert first_summary == second_summary


def test_preference_candidates_come_from_same_state_candidate_list():
    instance = make_square_instance()
    history = beam_search(instance, beam_width=4, candidate_k=16)
    prefs = extract_preferences(history)

    assert prefs
    for pref in prefs:
        matching_records = [
            record
            for record in history.records
            if record.state.actions == pref.state.actions
        ]
        assert matching_records
        candidate_actions = {
            candidate.action
            for candidate in matching_records[0].candidates
        }
        assert pref.better.action in candidate_actions
        assert pref.worse.action in candidate_actions
        assert pref.better.action != pref.worse.action


def test_square_preferences_keep_better_actions_on_best_trace_and_same_state():
    instance = make_square_instance()
    history = beam_search(instance, beam_width=4, candidate_k=16)
    best = history.best_finished()
    prefs = extract_preferences(history, delta=1.0)

    assert best is not None
    assert prefs
    best_trace = tuple(best.actions)
    for pref in prefs:
        prefix = tuple(pref.state.actions)
        step = len(prefix)
        assert best_trace[:step] == prefix
        assert pref.better.action == best_trace[step]

        matching_records = [
            record
            for record in history.records
            if record.state.actions == pref.state.actions
        ]
        assert matching_records
        candidate_actions = {
            candidate.action
            for candidate in matching_records[0].candidates
        }
        assert pref.better.action in candidate_actions
        assert pref.worse.action in candidate_actions
        assert pref.return_better > pref.return_worse + 1.0


def test_equal_size_alternative_does_not_create_preference_between_them():
    x = FastPoly.variable(0, 1, 2, 17)
    one = FastPoly.one(1, 2, 17)
    target = x + one
    instance = ProblemInstance(
        target=target,
        variables=("x",),
        field_p=17,
        degree_cap=2,
        op_budget=1,
        family_name="test_equal_alternatives",
    )
    source = CircuitState(
        nodes=(x, x, one),
        parents=(None, None, None),
        op_budget=1,
        degree_cap=2,
    )
    better_action = Action.make("add", 0, 2)
    equal_action = Action.make("add", 1, 2)
    better = Candidate(better_action, source.apply(better_action).nodes[-1])
    equal = Candidate(equal_action, source.apply(equal_action).nodes[-1])
    candidates = [better, equal]
    history = SearchHistory(
        instance=instance,
        records=[
            ExpandedStateRecord(instance, source, candidates, better, source.apply(better_action), 0, 1.0),
            ExpandedStateRecord(instance, source, candidates, equal, source.apply(equal_action), 0, 1.0),
        ],
        finished=[source.apply(equal_action), source.apply(better_action)],
    )

    prefs = extract_preferences(history, delta=0.0)

    assert prefs == []


def test_smaller_solution_is_preferred_over_longer_verified_branch():
    x = FastPoly.variable(0, 1, 2, 17)
    one = FastPoly.one(1, 2, 17)
    target = x + one
    instance = ProblemInstance(
        target=target,
        variables=("x",),
        field_p=17,
        degree_cap=2,
        op_budget=2,
        family_name="test_smaller_preferred",
    )
    source = CircuitState.initial(instance)
    better_action = Action.make("add", 0, 1)
    worse_action = Action.make("mul", 0, 1)
    better_next = source.apply(better_action)
    worse_next = source.apply(worse_action)
    longer_finished = worse_next.apply(Action.make("add", 1, 2))
    better = Candidate(better_action, better_next.nodes[-1])
    worse = Candidate(worse_action, worse_next.nodes[-1])
    candidates = [better, worse]
    history = SearchHistory(
        instance=instance,
        records=[
            ExpandedStateRecord(instance, source, candidates, better, better_next, 0, 1.0),
            ExpandedStateRecord(instance, source, candidates, worse, worse_next, 0, 1.0),
        ],
        finished=[longer_finished, better_next],
    )

    prefs = extract_preferences(history, delta=0.0)

    assert len(prefs) == 1
    assert prefs[0].better.action == better_action
    assert prefs[0].worse.action == worse_action
    assert prefs[0].return_better == 99.0
    assert prefs[0].return_worse == 98.0


def test_inconsistent_history_raises_clear_error():
    instance = make_square_instance()
    history = beam_search(instance, beam_width=4, candidate_k=16)
    corrupted = SearchHistory(instance=instance, records=[], finished=history.finished)

    with pytest.raises(ValueError, match="search history is inconsistent"):
        extract_preferences(corrupted)
