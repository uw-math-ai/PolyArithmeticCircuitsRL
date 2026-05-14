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
        assert instance.metadata["description"]
        assert type(instance.metadata["intended_complexity"]) is int
        assert instance.metadata["intended_complexity"] > 0
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
    assert first.metadata["generative_ops"] == 4
    assert first.metadata["intended_complexity"] == first.metadata["generative_ops"]
    assert first.metadata["complexity_note"] == (
        "generative construction size, not proven optimal"
    )


def test_structured_benchmark_intended_complexity_matches_known_constructions():
    benchmark = make_structured_benchmark(field_p=17, degree_cap=8)
    by_id = {instance.metadata["id"]: instance for instance in benchmark.instances}

    expected = {
        "power_sum_x_y_pow2": 2,
        "power_sum_x_y_pow3": 3,
        "power_sum_x_y_z_pow2": 3,
        "power_sum_x_y_z_pow3": 4,
        "power_sum_w_x_y_z_pow2": 4,
        "common_factor_a_b_c": 2,
        "common_factor_a_b_c_d": 3,
        "common_factor_a_b_c_d_e": 4,
        "common_factor_symmetric_x_y": 3,
        "common_factor_u_v_w_r_s": 4,
        "product_of_sums_a_b__c_d": 3,
        "product_of_sums_a_b__c_d_e": 4,
        "product_of_sums_a_b_c__d_e": 4,
        "product_of_sums_x_y__x_z": 3,
        "product_of_sums_x_y_z__u_v": 4,
        "nested_reuse_x_y__z": 4,
        "nested_reuse_x_y_z__w": 5,
        "product_of_sums_a_b__a_c": 3,
        "random_circuit_n3_ops3_seed101": 3,
        "random_circuit_n4_ops4_seed102": 4,
        "random_circuit_n5_ops5_seed103": 5,
        "random_circuit_n5_ops6_seed104": 6,
    }

    assert set(expected).issubset(by_id)
    for instance_id, intended_complexity in expected.items():
        instance = by_id[instance_id]
        assert instance.metadata["intended_complexity"] == intended_complexity
        if instance.metadata["family"] == "random_circuit":
            assert instance.metadata["generative_ops"] == intended_complexity
            assert instance.metadata["generating_trace"]


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
