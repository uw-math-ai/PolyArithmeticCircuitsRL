from lgs.data.benchmark_suite import make_structured_benchmark
from lgs.data.target_generators import make_random_circuit_instance
from lgs.env.circuit_state import CircuitState
from lgs.poly.fast_poly import Polynomial
from lgs.search.beam_search import beam_search
from lgs.search.candidate_generator import generate_candidates
from scripts.train_and_eval_benchmark import _split_instances


def test_structured_benchmark_instances_are_valid():
    benchmark = make_structured_benchmark(
        field_p=17,
        degree_cap=8,
        max_instances_per_family=2,
    )

    assert benchmark.instances
    ids = set()
    for instance in benchmark.instances:
        assert isinstance(instance.target, Polynomial)
        assert instance.op_budget > 0
        assert instance.metadata["id"]
        assert instance.metadata["family"]
        assert instance.metadata["benchmark_name"] == benchmark.name
        assert instance.metadata["id"] not in ids
        ids.add(instance.metadata["id"])
        state = CircuitState.initial(instance)
        assert state.num_nodes() == len(instance.variables) + 1


def test_heuristic_search_runs_on_generated_structured_instances():
    benchmark = make_structured_benchmark(
        field_p=17,
        degree_cap=8,
        max_instances_per_family=1,
    )

    for instance in benchmark.instances[:4]:
        beam_search(
            instance,
            beam_width=1,
            candidate_k=4,
            tier2_m=16,
        )


def test_random_circuit_generator_is_deterministic_by_seed():
    first = make_random_circuit_instance(
        4,
        4,
        field_p=17,
        degree_cap=8,
        seed=123,
    )
    second = make_random_circuit_instance(
        4,
        4,
        field_p=17,
        degree_cap=8,
        seed=123,
    )
    different = make_random_circuit_instance(
        4,
        4,
        field_p=17,
        degree_cap=8,
        seed=124,
    )

    assert first.target.key() == second.target.key()
    assert first.metadata["generating_trace"] == second.metadata["generating_trace"]
    assert (
        first.target.key() != different.target.key()
        or first.metadata["generating_trace"] != different.metadata["generating_trace"]
    )


def test_random_circuit_targets_are_not_one_step_from_initial_state():
    instance = make_random_circuit_instance(
        4,
        4,
        field_p=17,
        degree_cap=8,
        seed=123,
    )
    initial = CircuitState.initial(instance)

    assert not instance.target.is_zero()
    assert instance.target.degree() > 0
    assert all(node != instance.target for node in initial.nodes)
    assert all(
        candidate.result_poly != instance.target
        for candidate in generate_candidates(instance, initial, K=128, tier2_m=128)
    )


def test_train_validation_eval_split_ids_are_disjoint():
    benchmark = make_structured_benchmark(
        field_p=17,
        degree_cap=8,
        max_instances_per_family=3,
    )

    train, validation, eval_set = _split_instances(benchmark.instances)
    train_ids = {instance.metadata["id"] for instance in train}
    validation_ids = {instance.metadata["id"] for instance in validation}
    eval_ids = {instance.metadata["id"] for instance in eval_set}

    assert train_ids
    assert validation_ids
    assert eval_ids
    assert train_ids.isdisjoint(validation_ids)
    assert train_ids.isdisjoint(eval_ids)
    assert validation_ids.isdisjoint(eval_ids)
