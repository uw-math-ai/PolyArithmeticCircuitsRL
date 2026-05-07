"""Synthetic generators for structured decomposition traces."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from random import Random

from .baseline_cost import BaselineCostModel
from .cost_model import rebuild_cost, unresolved_children
from .factor_fp import FiniteFieldFactorizer
from .polynomial import SparsePolynomial
from .split_proposals import SplitAction, propose_splits


@dataclass(frozen=True)
class SupervisedExample:
    family: str
    target: SparsePolynomial
    candidates: tuple[SplitAction, ...]
    preferred_action: SplitAction
    value_target: float
    total_cost_target: float


def _validate_supervised_example(
    example: SupervisedExample,
    baseline_model: BaselineCostModel,
    require_positive_value: bool = False,
) -> SupervisedExample:
    if example.target.is_zero:
        raise ValueError("Supervised examples must have non-zero targets")
    if example.preferred_action.g.is_zero or example.preferred_action.h.is_zero:
        raise ValueError("Preferred split must be non-trivial")
    if example.preferred_action.g + example.preferred_action.h != example.target:
        raise ValueError("Preferred split does not reconstruct the target")
    if any(candidate.g + candidate.h != example.target for candidate in example.candidates):
        raise ValueError("At least one candidate split does not reconstruct the target")
    if not any(candidate.key() == example.preferred_action.key() for candidate in example.candidates):
        raise ValueError("Preferred split is missing from the candidate set")

    baseline = float(baseline_model.direct_construction_cost(example.target))
    if require_positive_value and not (example.total_cost_target < baseline and example.value_target > 0):
        raise ValueError("Example must provide a real improvement over the baseline")
    return example


def random_sparse_polynomial(
    rng: Random,
    prime: int,
    variables: tuple[str, ...],
    support_size: int,
    max_degree: int,
) -> SparsePolynomial:
    terms: dict[tuple[int, ...], int] = {}
    while len(terms) < support_size:
        exponent = tuple(rng.randint(0, max_degree) for _ in variables)
        coeff = rng.randint(1, prime - 1)
        terms[exponent] = coeff
    return SparsePolynomial.from_dict(terms, prime, variables)


def _lift_polynomial(
    poly: SparsePolynomial,
    variables: tuple[str, ...],
) -> SparsePolynomial:
    if poly.variables == variables:
        return poly
    index_lookup = [variables.index(name) for name in poly.variables]
    lifted_terms = []
    for coeff, exponent in poly.terms:
        full_exponent = [0] * len(variables)
        for local_index, value in enumerate(exponent):
            full_exponent[index_lookup[local_index]] = value
        lifted_terms.append((coeff, tuple(full_exponent)))
    return SparsePolynomial.from_terms(lifted_terms, poly.p, variables)


def planted_factorable_example(
    rng: Random,
    prime: int,
    variables: tuple[str, ...],
    support_size: int = 3,
    max_degree: int = 2,
    k_candidates: int = 16,
    max_attempts: int = 128,
) -> SupervisedExample:
    baseline_model = BaselineCostModel()
    factorizer = FiniteFieldFactorizer()
    try:
        for _ in range(max_attempts):
            g_left = random_sparse_polynomial(rng, prime, variables, support_size, max_degree)
            g_right = SparsePolynomial.variable(rng.choice(variables), prime, variables) + SparsePolynomial.from_monomial(
                rng.randint(1, prime - 1),
                (0,) * len(variables),
                prime,
                variables,
            )
            h_left = random_sparse_polynomial(rng, prime, variables, support_size, max_degree)
            h_right = SparsePolynomial.variable(rng.choice(variables), prime, variables) + SparsePolynomial.from_monomial(
                rng.randint(1, prime - 1),
                (0,) * len(variables),
                prime,
                variables,
            )
            g = g_left * g_right
            h = h_left * h_right
            target = g + h
            preferred = SplitAction(g=g, h=h, source="planted")
            candidates = tuple(propose_splits(target, k_candidates, baseline_model=baseline_model))
            if preferred.key() not in {candidate.key() for candidate in candidates}:
                candidates = (preferred.ordered(),) + candidates[: max(0, k_candidates - 1)]

            g_factorization = factorizer.factor(g)
            h_factorization = factorizer.factor(h)
            g_rebuild = rebuild_cost(g_factorization)
            h_rebuild = rebuild_cost(h_factorization)
            child_map: dict[str, SparsePolynomial] = {}
            resolved_base_cost = 0.0
            for child in unresolved_children(g_factorization) + unresolved_children(h_factorization):
                if baseline_model.is_base_case(child):
                    resolved_base_cost += float(baseline_model.exact_base_cost(child))
                    continue
                child_map.setdefault(child.to_key(), child)
            total_cost = (
                1
                + g_rebuild
                + h_rebuild
                + resolved_base_cost
                + sum(float(baseline_model.direct_construction_cost(child)) for child in child_map.values())
            )
            value_target = (
                baseline_model.direct_construction_cost(target) - total_cost
            ) / max(1, baseline_model.direct_construction_cost(target))
            if value_target <= 0:
                continue
            return _validate_supervised_example(
                SupervisedExample(
                    family="planted_factorable",
                    target=target,
                    candidates=tuple(candidates),
                    preferred_action=preferred.ordered(),
                    value_target=value_target,
                    total_cost_target=float(total_cost),
                ),
                baseline_model,
                require_positive_value=True,
            )
    finally:
        factorizer.close()
    raise RuntimeError("Failed to generate a useful planted factorable example")


def horner_polynomial(
    coefficients: list[int],
    prime: int,
    variable: str = "x",
) -> SparsePolynomial:
    variables = (variable,)
    poly = SparsePolynomial.zero(prime, variables)
    x = SparsePolynomial.variable(variable, prime, variables)
    for coeff in reversed(coefficients):
        poly = SparsePolynomial.from_monomial(coeff, (0,), prime, variables) + (x * poly)
    return poly


def horner_example(
    coefficients: list[int],
    prime: int,
    k_candidates: int = 8,
) -> SupervisedExample:
    baseline_model = BaselineCostModel()
    target = horner_polynomial(coefficients, prime)
    remainder, quotient = target.split_by_variable(0)
    preferred = SplitAction(
        g=remainder,
        h=target.variable_factor(0) * quotient,
        source="horner_trace",
    ).ordered()
    candidates = tuple(propose_splits(target, k_candidates, baseline_model=baseline_model))
    if preferred.key() not in {candidate.key() for candidate in candidates}:
        candidates = (preferred,) + candidates[: max(0, k_candidates - 1)]
    total_cost = 1 + baseline_model.direct_construction_cost(quotient)
    value_target = (
        baseline_model.direct_construction_cost(target) - total_cost
    ) / max(1, baseline_model.direct_construction_cost(target))
    return _validate_supervised_example(
        SupervisedExample(
            family="horner",
            target=target,
            candidates=tuple(candidates),
            preferred_action=preferred,
            value_target=value_target,
            total_cost_target=float(total_cost),
        ),
        baseline_model,
    )


def multivariate_horner_polynomial(
    coefficient_polynomials: list[SparsePolynomial],
    prime: int,
    variables: tuple[str, ...],
) -> SparsePolynomial:
    if len(variables) < 2:
        raise ValueError("Multivariate Horner form requires at least two variables")
    leading_variable = SparsePolynomial.variable(variables[0], prime, variables)
    poly = SparsePolynomial.zero(prime, variables)
    for coefficient_poly in reversed(coefficient_polynomials):
        poly = _lift_polynomial(coefficient_poly, variables) + (leading_variable * poly)
    return poly


def multivariate_horner_example(
    rng: Random,
    prime: int,
    variables: tuple[str, ...],
    outer_degree: int = 3,
    inner_support_size: int = 2,
    inner_max_degree: int = 2,
    k_candidates: int = 16,
    max_attempts: int = 32,
) -> SupervisedExample:
    if len(variables) < 2:
        raise ValueError("Multivariate Horner example requires at least two variables")
    baseline_model = BaselineCostModel()
    coefficient_variables = variables[1:]
    for _ in range(max_attempts):
        coefficient_polynomials = [
            random_sparse_polynomial(
                rng,
                prime,
                coefficient_variables,
                support_size=max(1, inner_support_size + rng.randint(0, 1)),
                max_degree=inner_max_degree,
            )
            for _ in range(outer_degree + 1)
        ]
        target = multivariate_horner_polynomial(coefficient_polynomials, prime, variables)
        remainder, quotient = target.split_by_variable(0)
        preferred = SplitAction(
            g=remainder,
            h=target.variable_factor(0) * quotient,
            source="multivariate_horner_trace",
        ).ordered()
        candidates = tuple(propose_splits(target, k_candidates, baseline_model=baseline_model))
        if preferred.key() not in {candidate.key() for candidate in candidates}:
            candidates = (preferred,) + candidates[: max(0, k_candidates - 1)]
        total_cost = 1 + baseline_model.direct_construction_cost(quotient)
        value_target = (
            baseline_model.direct_construction_cost(target) - total_cost
        ) / max(1, baseline_model.direct_construction_cost(target))
        if value_target <= 0:
            continue
        return _validate_supervised_example(
            SupervisedExample(
                family="multivariate_horner",
                target=target,
                candidates=tuple(candidates),
                preferred_action=preferred,
                value_target=value_target,
                total_cost_target=float(total_cost),
            ),
            baseline_model,
            require_positive_value=True,
        )
    raise RuntimeError("Failed to generate a useful multivariate Horner example")


def elementary_symmetric_polynomial(
    variables: tuple[str, ...],
    degree: int,
    prime: int,
) -> SparsePolynomial:
    if degree < 0 or degree > len(variables):
        return SparsePolynomial.zero(prime, variables)
    if degree == 0:
        return SparsePolynomial.one(prime, variables)

    result = SparsePolynomial.zero(prime, variables)
    from itertools import combinations

    for combo in combinations(range(len(variables)), degree):
        exponent = [0] * len(variables)
        for index in combo:
            exponent[index] = 1
        result = result + SparsePolynomial.from_monomial(1, tuple(exponent), prime, variables)
    return result


def elementary_symmetric_example(
    variable_count: int,
    degree: int,
    prime: int,
    k_candidates: int = 16,
) -> SupervisedExample:
    baseline_model = BaselineCostModel()
    variables = tuple(f"x{i+1}" for i in range(variable_count))
    target = elementary_symmetric_polynomial(variables, degree, prime)
    reduced_vars = variables[:-1]
    g = SparsePolynomial.zero(prime, variables)
    h = SparsePolynomial.zero(prime, variables)
    if degree <= len(reduced_vars):
        g = SparsePolynomial.from_sympy_expr(
            elementary_symmetric_polynomial(reduced_vars, degree, prime).to_sympy_expr(),
            prime,
            variables,
        )
    if degree - 1 >= 0:
        reduced = elementary_symmetric_polynomial(reduced_vars, degree - 1, prime)
        h = SparsePolynomial.variable(variables[-1], prime, variables) * SparsePolynomial.from_sympy_expr(
            reduced.to_sympy_expr(),
            prime,
            variables,
        )
    preferred = SplitAction(g=g, h=h, source="elementary_symmetric").ordered()
    candidates = tuple(propose_splits(target, k_candidates, baseline_model=baseline_model))
    if preferred.key() not in {candidate.key() for candidate in candidates}:
        candidates = (preferred,) + candidates[: max(0, k_candidates - 1)]
    total_cost = 1 + baseline_model.direct_construction_cost(g) + baseline_model.direct_construction_cost(h)
    value_target = (
        baseline_model.direct_construction_cost(target) - total_cost
    ) / max(1, baseline_model.direct_construction_cost(target))
    return _validate_supervised_example(
        SupervisedExample(
            family="elementary_symmetric",
            target=target,
            candidates=tuple(candidates),
            preferred_action=preferred,
            value_target=value_target,
            total_cost_target=float(total_cost),
        ),
        baseline_model,
    )


def exact_small_example(
    rng: Random,
    prime: int,
    variables: tuple[str, ...] = ("x", "y"),
    support_size: int = 3,
    max_degree: int = 2,
    k_candidates: int = 16,
    max_attempts: int = 64,
) -> SupervisedExample:
    baseline_model = BaselineCostModel()
    factorizer = FiniteFieldFactorizer()
    try:
        for _ in range(max_attempts):
            target = random_sparse_polynomial(rng, prime, variables, support_size, max_degree)
            baseline = baseline_model.direct_construction_cost(target)
            best_cost, best_action = _exact_best_decomposition(
                target,
                factorizer=factorizer,
                baseline_model=baseline_model,
                k_candidates=k_candidates,
                memo={},
            )
            if best_action is None or best_cost >= baseline:
                continue
            candidates = tuple(propose_splits(target, k_candidates, baseline_model=baseline_model))
            if best_action.key() not in {candidate.key() for candidate in candidates}:
                candidates = (best_action.ordered(),) + candidates[: max(0, k_candidates - 1)]
            value_target = (baseline - best_cost) / max(1, baseline)
            return _validate_supervised_example(
                SupervisedExample(
                    family="exact_small",
                    target=target,
                    candidates=tuple(candidates),
                    preferred_action=best_action.ordered(),
                    value_target=value_target,
                    total_cost_target=float(best_cost),
                ),
                baseline_model,
                require_positive_value=True,
            )
    finally:
        factorizer.close()
    raise RuntimeError("Failed to generate an exact-small example with a useful split")


def pretraining_mixture(
    rng: Random,
    prime: int,
    count: int,
    variables: tuple[str, ...] = ("x", "y"),
) -> list[SupervisedExample]:
    if count <= 0:
        return []
    plan = [
        ("planted", int(ceil(count * 0.40))),
        ("horner", int(ceil(count * 0.25))),
        ("elementary", int(ceil(count * 0.20))),
        ("exact_small", int(ceil(count * 0.15))),
    ]
    examples: list[SupervisedExample] = []
    for family, family_count in plan:
        for _ in range(family_count):
            if len(examples) >= count:
                break
            if family == "planted":
                examples.append(planted_factorable_example(rng, prime, variables))
            elif family == "horner":
                if len(variables) > 1:
                    examples.append(multivariate_horner_example(rng, prime, variables))
                else:
                    degree = rng.randint(3, 5)
                    coefficients = [rng.randint(0, prime - 1) for _ in range(degree + 1)]
                    if all(coeff == 0 for coeff in coefficients):
                        coefficients[0] = 1
                    if coefficients[0] == 0:
                        coefficients[0] = 1
                    examples.append(horner_example(coefficients, prime))
            elif family == "elementary":
                examples.append(elementary_symmetric_example(variable_count=4, degree=2, prime=prime))
            else:
                examples.append(exact_small_example(rng, prime, variables=variables))
    return examples[:count]


def _exact_best_decomposition(
    poly: SparsePolynomial,
    factorizer: FiniteFieldFactorizer,
    baseline_model: BaselineCostModel,
    k_candidates: int,
    memo: dict[str, tuple[float, SplitAction | None]],
) -> tuple[float, SplitAction | None]:
    key = poly.to_key()
    if key in memo:
        return memo[key]
    if baseline_model.is_base_case(poly):
        result = (float(baseline_model.exact_base_cost(poly)), None)
        memo[key] = result
        return result

    best_cost = float(baseline_model.direct_construction_cost(poly))
    best_action: SplitAction | None = None
    for action in propose_splits(poly, k_candidates, baseline_model=baseline_model):
        g_factors = factorizer.factor(action.g)
        h_factors = factorizer.factor(action.h)
        total = 1 + rebuild_cost(g_factors) + rebuild_cost(h_factors)

        child_map: dict[str, SparsePolynomial] = {}
        for child in unresolved_children(g_factors) + unresolved_children(h_factors):
            if child.to_key() == key:
                total = float("inf")
                break
            child_map.setdefault(child.to_key(), child)
        if total == float("inf"):
            continue

        for child in child_map.values():
            child_cost, _ = _exact_best_decomposition(
                child,
                factorizer=factorizer,
                baseline_model=baseline_model,
                k_candidates=k_candidates,
                memo=memo,
            )
            total += child_cost
        if total < best_cost:
            best_cost = total
            best_action = action

    memo[key] = (best_cost, best_action)
    return memo[key]
