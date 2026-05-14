from lgs.env.action import Action
from lgs.env.circuit_state import CircuitState
from lgs.env.problem_instance import ProblemInstance
from lgs.env.verification import verify_trace
from lgs.poly.fast_poly import FastPoly
from lgs.search.beam_search import beam_search, recover_trace
from lgs.search.search_history import SearchHistory


def make_square_instance(op_budget: int = 2, degree_cap: int = 2) -> ProblemInstance:
    x = FastPoly.variable(0, 2, degree_cap, 17)
    y = FastPoly.variable(1, 2, degree_cap, 17)
    return ProblemInstance(
        target=(x + y) * (x + y),
        variables=("x", "y"),
        field_p=17,
        degree_cap=degree_cap,
        op_budget=op_budget,
        family_name="test_square",
    )


def make_common_factor_instance(op_budget: int = 2) -> ProblemInstance:
    a = FastPoly.variable(0, 3, 2, 17)
    b = FastPoly.variable(1, 3, 2, 17)
    c = FastPoly.variable(2, 3, 2, 17)
    return ProblemInstance(
        target=a * b + a * c,
        variables=("a", "b", "c"),
        field_p=17,
        degree_cap=2,
        op_budget=op_budget,
        family_name="test_common_factor",
    )


def make_product_of_sums_instance(op_budget: int = 3) -> ProblemInstance:
    a = FastPoly.variable(0, 4, 2, 17)
    b = FastPoly.variable(1, 4, 2, 17)
    c = FastPoly.variable(2, 4, 2, 17)
    d = FastPoly.variable(3, 4, 2, 17)
    return ProblemInstance(
        target=(a + b) * (c + d),
        variables=("a", "b", "c", "d"),
        field_p=17,
        degree_cap=2,
        op_budget=op_budget,
        family_name="test_product_of_sums",
    )


def test_beam_search_solves_square_target():
    instance = make_square_instance()

    history = beam_search(instance, beam_width=4, candidate_k=16)
    best = history.best_finished()

    assert history.success()
    assert best is not None
    assert best.num_ops() <= 2
    assert verify_trace(instance, recover_trace(best))


def test_beam_search_solves_common_factor_target():
    instance = make_common_factor_instance()

    history = beam_search(instance, beam_width=4, candidate_k=16)
    best = history.best_finished()

    assert history.success()
    assert best is not None
    assert best.num_ops() <= 2
    assert verify_trace(instance, recover_trace(best))


def test_beam_search_solves_product_of_sums_target():
    instance = make_product_of_sums_instance()

    history = beam_search(instance, beam_width=8, candidate_k=32)
    best = history.best_finished()

    assert history.success()
    assert best is not None
    assert best.num_ops() <= 3
    assert verify_trace(instance, recover_trace(best))


def test_beam_search_respects_op_budget():
    instance = make_square_instance(op_budget=1)

    history = beam_search(instance, beam_width=4, candidate_k=16)

    assert not history.success()
    assert all(state.num_ops() <= instance.op_budget for state in history.finished)


def test_beam_search_is_deterministic():
    instance = make_product_of_sums_instance()

    first = beam_search(instance, beam_width=8, candidate_k=32).best_finished()
    second = beam_search(instance, beam_width=8, candidate_k=32).best_finished()

    assert first is not None
    assert second is not None
    assert recover_trace(first) == recover_trace(second)


def test_search_history_records_expansions():
    instance = make_square_instance()

    history = beam_search(instance, beam_width=4, candidate_k=16)

    assert history.records
    for record in history.records:
        assert record.instance == instance
        assert isinstance(record.state, CircuitState)
        assert record.candidates
        assert record.candidate in record.candidates
        assert isinstance(record.next_state, CircuitState)
        assert isinstance(record.depth, int)
        assert isinstance(record.state_score, float)
        assert record.state.apply(record.candidate.action) == record.next_state


def test_search_history_has_one_record_per_expanded_candidate():
    instance = make_square_instance()

    history = beam_search(instance, beam_width=4, candidate_k=16)

    grouped: dict[tuple[int, tuple[Action, ...]], list] = {}
    for record in history.records:
        key = (record.depth, record.state.actions)
        grouped.setdefault(key, []).append(record)

    assert grouped
    for records in grouped.values():
        candidate_actions = tuple(candidate.action for candidate in records[0].candidates)
        recorded_actions = tuple(record.candidate.action for record in records)
        assert len(records) == len(records[0].candidates)
        assert set(recorded_actions) == set(candidate_actions)


def test_best_trace_steps_match_search_history_records():
    instance = make_product_of_sums_instance()

    history = beam_search(instance, beam_width=8, candidate_k=32)
    best = history.best_finished()

    assert best is not None
    prefix_state = CircuitState.initial(instance)
    for depth, action in enumerate(best.actions):
        matching_records = [
            record
            for record in history.records
            if record.depth == depth
            and record.state.actions == prefix_state.actions
            and record.candidate.action == action
        ]
        assert matching_records
        prefix_state = prefix_state.apply(action)
    assert prefix_state == best


def test_record_candidate_lists_are_preserved_after_search():
    instance = make_common_factor_instance()

    history = beam_search(instance, beam_width=4, candidate_k=16)
    snapshots = [
        tuple(candidate.action for candidate in record.candidates)
        for record in history.records
    ]

    assert history.best_finished() is not None
    for record, snapshot in zip(history.records, snapshots):
        assert tuple(candidate.action for candidate in record.candidates) == snapshot
        assert record.candidate in record.candidates


def test_best_finished_returns_smallest_verified_circuit_not_first_finished():
    x = FastPoly.variable(0, 2, 2, 17)
    y = FastPoly.variable(1, 2, 2, 17)
    instance = ProblemInstance(
        target=x + y,
        variables=("x", "y"),
        field_p=17,
        degree_cap=2,
        op_budget=2,
        family_name="test_best_finished",
    )
    short = CircuitState.initial(instance).apply(Action.make("add", 0, 1))
    longer = short.apply(Action.make("add", 2, 2))
    history = SearchHistory(instance=instance, finished=[longer, short])

    best = history.best_finished()

    assert best == short
    assert best is not None
    assert best.num_ops() == 1
    assert verify_trace(instance, recover_trace(best))


def test_candidate_degree_overflow_does_not_crash_beam_search():
    instance = make_square_instance(op_budget=3, degree_cap=2)
    seed_state = CircuitState.initial(instance).apply(Action.make("mul", 0, 0))

    history = beam_search(
        instance,
        beam_width=4,
        candidate_k=16,
        max_depth=2,
    )

    assert history.records
    assert seed_state.num_ops() == 1
    assert all(record.next_state.num_ops() <= instance.op_budget for record in history.records)
