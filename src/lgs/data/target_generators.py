"""Deterministic structured target families for search benchmarks."""

from __future__ import annotations

import random
from collections.abc import Sequence

from lgs.env.action import Action
from lgs.env.problem_instance import ProblemInstance
from lgs.poly.fast_poly import FastPoly, Polynomial, PolynomialDegreeError


def make_square_instance(
    var_names: Sequence[str] = ("x", "y"),
    *,
    field_p: int,
    degree_cap: int,
    op_budget: int,
) -> ProblemInstance:
    return make_power_sum_instance(
        tuple(var_names),
        2,
        field_p=field_p,
        degree_cap=degree_cap,
        op_budget=op_budget,
    )


def make_common_factor_instance(
    var_names: Sequence[str] = ("a", "b", "c"),
    *,
    field_p: int,
    degree_cap: int,
    op_budget: int,
) -> ProblemInstance:
    names = _validate_var_names(var_names, expected=3)
    return make_common_factor_sum_instance(
        names[0],
        names[1:],
        field_p=field_p,
        degree_cap=degree_cap,
        op_budget=op_budget,
    )


def make_power_sum_instance(
    var_names: tuple[str, ...],
    power: int,
    *,
    field_p: int,
    degree_cap: int,
    op_budget: int,
) -> ProblemInstance:
    names = _validate_var_names(var_names)
    if type(power) is not int or power < 1:
        raise ValueError("power must be a positive int")
    variables = _variables(len(names), field_p=field_p, degree_cap=degree_cap)
    base_sum = _sum_polynomials(variables)
    target = base_sum ** power
    intended_ops = max(0, len(names) - 1) + max(0, power - 1)
    description = f"({' + '.join(names)})^{power}"
    return _make_instance(
        target=target,
        variables=names,
        field_p=field_p,
        degree_cap=degree_cap,
        op_budget=op_budget,
        family="power_sum",
        target_id=f"power_sum_{'_'.join(names)}_pow{power}",
        description=description,
        intended_complexity=intended_ops,
    )


def make_common_factor_sum_instance(
    common_var: str,
    other_vars: tuple[str, ...],
    *,
    field_p: int,
    degree_cap: int,
    op_budget: int,
) -> ProblemInstance:
    names = _validate_var_names((common_var, *other_vars))
    if len(other_vars) < 1:
        raise ValueError("other_vars must be non-empty")
    variables = _variables(len(names), field_p=field_p, degree_cap=degree_cap)
    common = variables[0]
    rest_sum = _sum_polynomials(variables[1:])
    target = common * rest_sum
    intended_ops = max(0, len(other_vars) - 1) + 1
    description = f"{common_var}({' + '.join(other_vars)})"
    return _make_instance(
        target=target,
        variables=names,
        field_p=field_p,
        degree_cap=degree_cap,
        op_budget=op_budget,
        family="common_factor",
        target_id=f"common_factor_{common_var}_{'_'.join(other_vars)}",
        description=description,
        intended_complexity=intended_ops,
    )


def make_product_of_sums_instance(
    left_vars: Sequence[str] = ("a", "b", "c", "d"),
    right_vars: Sequence[str] | None = None,
    *,
    field_p: int,
    degree_cap: int,
    op_budget: int,
) -> ProblemInstance:
    left, right = _resolve_product_var_groups(left_vars, right_vars)
    _validate_group_var_names(left)
    _validate_group_var_names(right)
    names = _unique_in_order((*left, *right))
    variables = _variables(len(names), field_p=field_p, degree_cap=degree_cap)
    by_name = dict(zip(names, variables))
    left_sum = _sum_polynomials([by_name[name] for name in left])
    right_sum = _sum_polynomials([by_name[name] for name in right])
    target = left_sum * right_sum
    intended_ops = max(0, len(left) - 1) + max(0, len(right) - 1) + 1
    description = f"({' + '.join(left)})({' + '.join(right)})"
    return _make_instance(
        target=target,
        variables=names,
        field_p=field_p,
        degree_cap=degree_cap,
        op_budget=op_budget,
        family="product_of_sums",
        target_id=f"product_of_sums_{'_'.join(left)}__{'_'.join(right)}",
        description=description,
        intended_complexity=intended_ops,
    )


def make_xy_symmetric_instance(
    var_names: tuple[str, str] = ("x", "y"),
    *,
    field_p: int,
    degree_cap: int,
    op_budget: int,
) -> ProblemInstance:
    names = _validate_var_names(var_names, expected=2)
    x, y = _variables(2, field_p=field_p, degree_cap=degree_cap)
    target = (x * x) * y + x * (y * y)
    return _make_instance(
        target=target,
        variables=names,
        field_p=field_p,
        degree_cap=degree_cap,
        op_budget=op_budget,
        family="common_factor",
        target_id=f"common_factor_symmetric_{'_'.join(names)}",
        description=f"{names[0]}^2{names[1]} + {names[0]}{names[1]}^2",
        intended_complexity=4,
    )


def make_sum_square_plus_linear_instance(
    sum_vars: tuple[str, ...],
    linear_var: str,
    *,
    field_p: int,
    degree_cap: int,
    op_budget: int,
) -> ProblemInstance:
    names = _validate_var_names((*sum_vars, linear_var))
    if len(sum_vars) < 2:
        raise ValueError("sum_vars must contain at least two variables")
    variables = _variables(len(names), field_p=field_p, degree_cap=degree_cap)
    sum_poly = _sum_polynomials(variables[: len(sum_vars)])
    linear = variables[-1]
    target = sum_poly * sum_poly + linear * sum_poly
    intended_ops = max(0, len(sum_vars) - 1) + 3
    description = f"({' + '.join(sum_vars)})^2 + {linear_var}({' + '.join(sum_vars)})"
    return _make_instance(
        target=target,
        variables=names,
        field_p=field_p,
        degree_cap=degree_cap,
        op_budget=op_budget,
        family="nested_reuse",
        target_id=f"nested_reuse_{'_'.join(sum_vars)}__{linear_var}",
        description=description,
        intended_complexity=intended_ops,
    )


def make_random_circuit_instance(
    n_vars: int,
    num_ops: int,
    *,
    field_p: int,
    degree_cap: int,
    seed: int,
) -> ProblemInstance:
    if type(n_vars) is not int or n_vars <= 0:
        raise ValueError("n_vars must be a positive int")
    if type(num_ops) is not int or num_ops <= 0:
        raise ValueError("num_ops must be a positive int")
    if type(seed) is not int:
        raise ValueError("seed must be an int")

    names = tuple(f"x{i}" for i in range(n_vars))
    for restart in range(50):
        rng = random.Random(seed + restart * 1_000_003)
        nodes = [
            FastPoly.variable(index, n_vars, degree_cap, field_p)
            for index in range(n_vars)
        ]
        nodes.append(FastPoly.one(n_vars, degree_cap, field_p))
        trace: list[tuple[str, int, int]] = []

        attempts = 0
        max_attempts = max(100, num_ops * 100)
        while len(trace) < num_ops and attempts < max_attempts:
            attempts += 1
            op = "mul" if rng.random() < 0.45 else "add"
            i = rng.randrange(len(nodes))
            j = rng.randrange(len(nodes))
            action = Action.make(op, i, j)
            try:
                if action.op == "add":
                    result = nodes[action.i] + nodes[action.j]
                else:
                    result = nodes[action.i] * nodes[action.j]
            except PolynomialDegreeError:
                continue
            if result.is_zero():
                continue
            nodes.append(result)
            trace.append((action.op, action.i, action.j))

        if len(trace) != num_ops:
            continue

        target = nodes[-1]
        if not _is_nontrivial_random_target(
            target,
            n_vars=n_vars,
            field_p=field_p,
            degree_cap=degree_cap,
        ):
            continue

        return _make_instance(
            target=target,
            variables=names,
            field_p=field_p,
            degree_cap=degree_cap,
            op_budget=num_ops,
            family="random_circuit",
            target_id=f"random_circuit_n{n_vars}_ops{num_ops}_seed{seed}",
            description=f"random structured circuit with {n_vars} variables and {num_ops} ops",
            intended_complexity=num_ops,
            extra_metadata={
                "generative_ops": num_ops,
                "generating_trace": tuple(trace),
                "seed": seed,
                "generation_restart": restart,
            },
        )

    raise ValueError(
        f"could not generate a nontrivial {num_ops}-op random circuit "
        f"under degree_cap={degree_cap}"
    )


def make_tiny_train_instances(field_p: int, degree_cap: int) -> list[ProblemInstance]:
    return [
        make_square_instance(("x", "y"), field_p=field_p, degree_cap=degree_cap, op_budget=2),
        make_common_factor_instance(
            ("a", "b", "c"),
            field_p=field_p,
            degree_cap=degree_cap,
            op_budget=2,
        ),
        make_product_of_sums_instance(
            ("a", "b", "c", "d"),
            field_p=field_p,
            degree_cap=degree_cap,
            op_budget=3,
        ),
    ]


def make_tiny_validation_instances(field_p: int, degree_cap: int) -> list[ProblemInstance]:
    return [
        make_square_instance(("u", "v"), field_p=field_p, degree_cap=degree_cap, op_budget=2),
        make_common_factor_instance(
            ("u", "v", "w"),
            field_p=field_p,
            degree_cap=degree_cap,
            op_budget=2,
        ),
        make_product_of_sums_instance(
            ("p", "q", "r", "s"),
            field_p=field_p,
            degree_cap=degree_cap,
            op_budget=3,
        ),
    ]


def _make_instance(
    *,
    target: Polynomial,
    variables: tuple[str, ...],
    field_p: int,
    degree_cap: int,
    op_budget: int,
    family: str,
    target_id: str,
    description: str,
    intended_complexity: int,
    extra_metadata: dict[str, object] | None = None,
) -> ProblemInstance:
    metadata: dict[str, object] = {
        "id": target_id,
        "target_id": target_id,
        "family": family,
        "intended_complexity": intended_complexity,
        "description": description,
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    return ProblemInstance(
        target=target,
        variables=variables,
        field_p=field_p,
        degree_cap=degree_cap,
        op_budget=op_budget,
        family_name=family,
        metadata=metadata,
    )


def _sum_polynomials(polys: Sequence[Polynomial]) -> Polynomial:
    if not polys:
        raise ValueError("cannot sum an empty polynomial sequence")
    total = polys[0]
    for poly in polys[1:]:
        total = total + poly
    return total


def _variables(
    n_vars: int,
    *,
    field_p: int,
    degree_cap: int,
) -> tuple[Polynomial, ...]:
    return tuple(
        FastPoly.variable(index, n_vars, degree_cap, field_p)
        for index in range(n_vars)
    )


def _resolve_product_var_groups(
    left_vars: Sequence[str],
    right_vars: Sequence[str] | None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if right_vars is not None:
        left = tuple(left_vars)
        right = tuple(right_vars)
    else:
        names = tuple(left_vars)
        if len(names) == 4:
            left = names[:2]
            right = names[2:]
        elif len(names) >= 2 and len(names) % 2 == 0:
            midpoint = len(names) // 2
            left = names[:midpoint]
            right = names[midpoint:]
        else:
            raise ValueError(
                "right_vars is required unless left_vars contains an even "
                "combined variable list"
            )
    if not left or not right:
        raise ValueError("product-of-sums groups must be non-empty")
    return left, right


def _is_nontrivial_random_target(
    target: Polynomial,
    *,
    n_vars: int,
    field_p: int,
    degree_cap: int,
) -> bool:
    if target.is_zero() or target.degree() == 0:
        return False
    base_nodes = [
        FastPoly.variable(index, n_vars, degree_cap, field_p)
        for index in range(n_vars)
    ]
    base_nodes.append(FastPoly.one(n_vars, degree_cap, field_p))
    if any(target == node for node in base_nodes):
        return False

    for i, left in enumerate(base_nodes):
        for j, right in enumerate(base_nodes[i:], start=i):
            if left + right == target:
                return False
            try:
                if left * right == target:
                    return False
            except PolynomialDegreeError:
                continue
    return True


def _validate_var_names(
    var_names: Sequence[str],
    *,
    expected: int | None = None,
) -> tuple[str, ...]:
    names = tuple(var_names)
    if expected is not None and len(names) != expected:
        raise ValueError(f"expected {expected} variable names, got {len(names)}")
    if not names:
        raise ValueError("at least one variable name is required")
    if len(set(names)) != len(names):
        raise ValueError("variable names must be unique")
    for name in names:
        if not isinstance(name, str) or not name:
            raise ValueError("variable names must be non-empty strings")
    return names


def _validate_group_var_names(var_names: Sequence[str]) -> tuple[str, ...]:
    names = tuple(var_names)
    if not names:
        raise ValueError("variable group must be non-empty")
    for name in names:
        if not isinstance(name, str) or not name:
            raise ValueError("variable names must be non-empty strings")
    return names


def _unique_in_order(var_names: Sequence[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for name in var_names:
        if name in seen:
            continue
        seen.add(name)
        ordered.append(name)
    return tuple(ordered)
