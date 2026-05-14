"""Structured benchmark suite for learned symbolic search."""

from __future__ import annotations

from dataclasses import dataclass

from lgs.env.problem_instance import ProblemInstance
from lgs.data.target_generators import (
    make_common_factor_sum_instance,
    make_power_sum_instance,
    make_product_of_sums_instance,
    make_random_circuit_instance,
    make_sum_square_plus_linear_instance,
    make_xy_symmetric_instance,
)


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    field_p: int
    degree_cap: int
    instances: list[ProblemInstance]


def make_structured_benchmark(
    *,
    field_p: int = 268435399,
    degree_cap: int = 8,
    max_instances_per_family: int | None = None,
) -> BenchmarkSpec:
    if max_instances_per_family is not None and (
        type(max_instances_per_family) is not int
        or max_instances_per_family < 0
    ):
        raise ValueError("max_instances_per_family must be None or a non-negative int")

    by_family: dict[str, list[ProblemInstance]] = {
        "power_sum": [
            make_power_sum_instance(("x", "y"), 2, field_p=field_p, degree_cap=degree_cap, op_budget=2),
            make_power_sum_instance(("x", "y"), 3, field_p=field_p, degree_cap=degree_cap, op_budget=3),
            make_power_sum_instance(("x", "y", "z"), 2, field_p=field_p, degree_cap=degree_cap, op_budget=3),
            make_power_sum_instance(("x", "y", "z"), 3, field_p=field_p, degree_cap=degree_cap, op_budget=4),
            make_power_sum_instance(("w", "x", "y", "z"), 2, field_p=field_p, degree_cap=degree_cap, op_budget=4),
        ],
        "common_factor": [
            make_common_factor_sum_instance("a", ("b", "c"), field_p=field_p, degree_cap=degree_cap, op_budget=2),
            make_common_factor_sum_instance("a", ("b", "c", "d"), field_p=field_p, degree_cap=degree_cap, op_budget=3),
            make_common_factor_sum_instance("a", ("b", "c", "d", "e"), field_p=field_p, degree_cap=degree_cap, op_budget=4),
            make_xy_symmetric_instance(("x", "y"), field_p=field_p, degree_cap=degree_cap, op_budget=4),
            make_common_factor_sum_instance("u", ("v", "w", "r", "s"), field_p=field_p, degree_cap=degree_cap, op_budget=4),
        ],
        "product_of_sums": [
            make_product_of_sums_instance(("a", "b"), ("c", "d"), field_p=field_p, degree_cap=degree_cap, op_budget=3),
            make_product_of_sums_instance(("a", "b"), ("c", "d", "e"), field_p=field_p, degree_cap=degree_cap, op_budget=4),
            make_product_of_sums_instance(("a", "b", "c"), ("d", "e"), field_p=field_p, degree_cap=degree_cap, op_budget=4),
            make_product_of_sums_instance(("x", "y"), ("x", "z"), field_p=field_p, degree_cap=degree_cap, op_budget=3),
            make_product_of_sums_instance(("x", "y", "z"), ("u", "v"), field_p=field_p, degree_cap=degree_cap, op_budget=4),
        ],
        "nested_reuse": [
            make_sum_square_plus_linear_instance(("x", "y"), "z", field_p=field_p, degree_cap=degree_cap, op_budget=4),
            make_sum_square_plus_linear_instance(("x", "y", "z"), "w", field_p=field_p, degree_cap=degree_cap, op_budget=5),
            make_product_of_sums_instance(("a", "b"), ("a", "c"), field_p=field_p, degree_cap=degree_cap, op_budget=3),
        ],
        "random_circuit": [
            make_random_circuit_instance(3, 3, field_p=field_p, degree_cap=degree_cap, seed=101),
            make_random_circuit_instance(4, 4, field_p=field_p, degree_cap=degree_cap, seed=102),
            make_random_circuit_instance(5, 5, field_p=field_p, degree_cap=degree_cap, seed=103),
            make_random_circuit_instance(5, 6, field_p=field_p, degree_cap=degree_cap, seed=104),
        ],
    }

    instances: list[ProblemInstance] = []
    for family_instances in by_family.values():
        selected = (
            family_instances
            if max_instances_per_family is None
            else family_instances[:max_instances_per_family]
        )
        instances.extend(selected)
    benchmark_name = "structured_v1"
    instances = [
        _with_benchmark_metadata(instance, benchmark_name)
        for instance in instances
    ]
    return BenchmarkSpec(
        name=benchmark_name,
        field_p=field_p,
        degree_cap=degree_cap,
        instances=instances,
    )


def _with_benchmark_metadata(
    instance: ProblemInstance,
    benchmark_name: str,
) -> ProblemInstance:
    metadata = dict(instance.metadata)
    metadata["benchmark_name"] = benchmark_name
    return ProblemInstance(
        target=instance.target,
        variables=instance.variables,
        field_p=instance.field_p,
        degree_cap=instance.degree_cap,
        op_budget=instance.op_budget,
        family_name=instance.family_name,
        metadata=metadata,
    )
