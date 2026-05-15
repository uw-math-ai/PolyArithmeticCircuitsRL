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
        "random_circuit": _make_random_circuit_instances(
            count=4 if max_instances_per_family is None else max(4, max_instances_per_family),
            field_p=field_p,
            degree_cap=degree_cap,
        ),
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


_BASE_RANDOM_CIRCUITS: tuple[tuple[int, int, int], ...] = (
    (3, 3, 101),
    (4, 4, 102),
    (5, 5, 103),
    (5, 6, 104),
)


def _make_random_circuit_instances(
    count: int,
    *,
    field_p: int,
    degree_cap: int,
) -> list[ProblemInstance]:
    # The first 4 entries are fixed to preserve stable instance IDs across runs.
    # Extra instances beyond the base 4 use incrementing seeds.
    specs: list[tuple[int, int, int]] = list(_BASE_RANDOM_CIRCUITS[:count])
    next_seed = _BASE_RANDOM_CIRCUITS[-1][2] + 1
    for i in range(len(specs), count):
        num_vars = 3 + (i % 3)
        num_ops = 3 + (i % 4)
        specs.append((num_vars, num_ops, next_seed))
        next_seed += 1
    return [
        make_random_circuit_instance(
            num_vars,
            num_ops,
            field_p=field_p,
            degree_cap=degree_cap,
            seed=seed,
        )
        for num_vars, num_ops, seed in specs
    ]


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
