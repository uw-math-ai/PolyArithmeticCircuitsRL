"""Candidate split generation for decomposition search."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from math import comb
from random import Random

from .baseline_cost import BaselineCostModel
from .canonical import order_pair_by_key
from .config import ProposalConfig
from .polynomial import SparsePolynomial


@dataclass(frozen=True)
class SplitAction:
    g: SparsePolynomial
    h: SparsePolynomial
    source: str
    score_hint: float = 0.0
    metadata: tuple[tuple[str, float], ...] = field(default_factory=tuple)

    def ordered(self) -> "SplitAction":
        if order_pair_by_key(self.g.to_key(), self.h.to_key()):
            return self
        return SplitAction(
            g=self.h,
            h=self.g,
            source=self.source,
            score_hint=self.score_hint,
            metadata=self.metadata,
        )

    def key(self) -> tuple[str, str]:
        ordered = self.ordered()
        return (ordered.g.to_key(), ordered.h.to_key())


def _candidate_hint(
    target: SparsePolynomial,
    action: SplitAction,
    baseline_model: BaselineCostModel,
) -> float:
    size_balance = abs(action.g.support_size - action.h.support_size)
    degree_balance = abs(action.g.total_degree - action.h.total_degree)
    horner_bonus = 0.5 if action.source == "horner" else 0.0
    naive_split_cost = 1 + baseline_model.direct_construction_cost(action.g) + baseline_model.direct_construction_cost(action.h)
    return (
        baseline_model.direct_construction_cost(target)
        - naive_split_cost
        - 0.1 * size_balance
        - 0.1 * degree_balance
        + horner_bonus
    )


def _build_action(
    target: SparsePolynomial,
    g: SparsePolynomial,
    h: SparsePolynomial,
    source: str,
    baseline_model: BaselineCostModel,
) -> SplitAction | None:
    if g.is_zero or h.is_zero:
        return None
    if g + h != target:
        return None
    action = SplitAction(g=g, h=h, source=source)
    hint = _candidate_hint(target, action, baseline_model)
    return SplitAction(
        g=action.ordered().g,
        h=action.ordered().h,
        source=source,
        score_hint=hint,
        metadata=(
            ("g_support", float(g.support_size)),
            ("h_support", float(h.support_size)),
        ),
    )


def _source_priority(source: str) -> int:
    ordering = {
        "family_template": 5,
        "common_factor": 4,
        "horner": 3,
        "support_partition": 2,
        "random_mask": 1,
    }
    return ordering.get(source, 0)


def _support_partition_candidates(
    target: SparsePolynomial,
    budget: int,
    baseline_model: BaselineCostModel,
) -> list[SplitAction]:
    if target.support_size <= 1 or budget <= 0:
        return []
    actions: list[SplitAction] = []
    terms = target.terms
    for pivot in range(1, min(len(terms), budget + 1)):
        left = SparsePolynomial(target.p, target.variables, terms[:pivot])
        right = SparsePolynomial(target.p, target.variables, terms[pivot:])
        action = _build_action(target, left, right, "support_partition", baseline_model)
        if action is not None:
            actions.append(action)
    return actions


def _random_mask_candidates(
    target: SparsePolynomial,
    budget: int,
    baseline_model: BaselineCostModel,
    rng: Random,
) -> list[SplitAction]:
    if target.support_size <= 2 or budget <= 0:
        return []
    terms = list(target.terms)
    actions: list[SplitAction] = []
    for _ in range(budget):
        left_terms: list[tuple[int, tuple[int, ...]]] = []
        right_terms: list[tuple[int, tuple[int, ...]]] = []
        for term in terms:
            (left_terms if rng.random() < 0.5 else right_terms).append(term)
        if not left_terms or not right_terms:
            continue
        action = _build_action(
            target,
            SparsePolynomial(target.p, target.variables, tuple(left_terms)),
            SparsePolynomial(target.p, target.variables, tuple(right_terms)),
            "random_mask",
            baseline_model,
        )
        if action is not None:
            actions.append(action)
    return actions


def _horner_candidates(
    target: SparsePolynomial,
    budget: int,
    baseline_model: BaselineCostModel,
) -> list[SplitAction]:
    if budget <= 0:
        return []
    actions: list[SplitAction] = []
    for index, max_degree in enumerate(target.max_degrees):
        if max_degree == 0:
            continue
        remainder, quotient = target.split_by_variable(index)
        if remainder.is_zero or quotient.is_zero:
            continue
        variable_factor = target.variable_factor(index)
        action = _build_action(
            target,
            remainder,
            variable_factor * quotient,
            "horner",
            baseline_model,
        )
        if action is not None:
            actions.append(action)
        if len(actions) >= budget:
            break
    return actions


def _common_factor_candidates(
    target: SparsePolynomial,
    budget: int,
    baseline_model: BaselineCostModel,
) -> list[SplitAction]:
    if budget <= 0 or target.support_size <= 2:
        return []
    actions: list[SplitAction] = []
    seen_clusters: set[tuple[tuple[int, ...], ...]] = set()
    for index, max_degree in enumerate(target.max_degrees):
        for power in range(1, max_degree + 1):
            cluster_terms = [
                term for term in target.terms if term[1][index] >= power
            ]
            remainder_terms = [
                term for term in target.terms if term[1][index] < power
            ]
            if not cluster_terms or not remainder_terms:
                continue
            cluster_signature = tuple(sorted(exponent for _, exponent in cluster_terms))
            if cluster_signature in seen_clusters:
                continue
            seen_clusters.add(cluster_signature)
            cluster = SparsePolynomial(target.p, target.variables, tuple(cluster_terms))
            remainder = SparsePolynomial(target.p, target.variables, tuple(remainder_terms))
            action = _build_action(target, remainder, cluster, "common_factor", baseline_model)
            if action is not None:
                actions.append(action)
            if len(actions) >= budget:
                return actions
    return actions


def _family_template_candidates(
    target: SparsePolynomial,
    budget: int,
    baseline_model: BaselineCostModel,
) -> list[SplitAction]:
    if budget <= 0:
        return []
    action = _elementary_symmetric_template(target, baseline_model)
    return [action] if action is not None else []


def _elementary_symmetric_template(
    target: SparsePolynomial,
    baseline_model: BaselineCostModel,
) -> SplitAction | None:
    variable_count = len(target.variables)
    degree = target.total_degree
    if variable_count < 2 or degree <= 0 or degree > variable_count:
        return None
    if target.support_size != comb(variable_count, degree):
        return None
    if any(coeff != 1 for coeff, _ in target.terms):
        return None
    support = {exponent for _, exponent in target.terms}
    expected_support: set[tuple[int, ...]] = set()
    for combo in combinations(range(variable_count), degree):
        exponent = [0] * variable_count
        for index in combo:
            exponent[index] = 1
        expected_support.add(tuple(exponent))
    if support != expected_support:
        return None

    g = _elementary_symmetric_poly(target.variables[:-1], degree, target.p, target.variables)
    h = SparsePolynomial.variable(target.variables[-1], target.p, target.variables) * _elementary_symmetric_poly(
        target.variables[:-1],
        degree - 1,
        target.p,
        target.variables,
    )
    return _build_action(target, g, h, "family_template", baseline_model)


def _elementary_symmetric_poly(
    active_variables: tuple[str, ...],
    degree: int,
    prime: int,
    ambient_variables: tuple[str, ...],
) -> SparsePolynomial:
    if degree < 0 or degree > len(active_variables):
        return SparsePolynomial.zero(prime, ambient_variables)
    if degree == 0:
        return SparsePolynomial.one(prime, ambient_variables)
    active_indices = [ambient_variables.index(name) for name in active_variables]
    result = SparsePolynomial.zero(prime, ambient_variables)
    for combo in combinations(active_indices, degree):
        exponent = [0] * len(ambient_variables)
        for index in combo:
            exponent[index] = 1
        result = result + SparsePolynomial.from_monomial(1, tuple(exponent), prime, ambient_variables)
    return result


def propose_splits(
    target: SparsePolynomial,
    k: int,
    config: ProposalConfig | None = None,
    baseline_model: BaselineCostModel | None = None,
) -> list[SplitAction]:
    config = config or ProposalConfig()
    baseline_model = baseline_model or BaselineCostModel()
    rng = Random(config.random_seed)

    support_budget = max(1, int(round(k * config.support_fraction)))
    horner_budget = max(1, int(round(k * config.horner_fraction)))
    common_budget = max(0, int(round(k * config.common_factor_fraction)))
    family_budget = max(0, int(round(k * config.family_fraction)))
    remaining = max(0, k - support_budget - horner_budget - common_budget - family_budget)
    random_budget = min(config.max_random_masks, max(1, remaining or int(round(k * config.random_fraction))))

    candidates = []
    candidates.extend(_support_partition_candidates(target, support_budget, baseline_model))
    candidates.extend(_horner_candidates(target, horner_budget, baseline_model))
    candidates.extend(_common_factor_candidates(target, common_budget, baseline_model))
    candidates.extend(_family_template_candidates(target, family_budget, baseline_model))
    candidates.extend(_random_mask_candidates(target, random_budget, baseline_model, rng))

    deduped: dict[tuple[str, str], SplitAction] = {}
    for candidate in candidates:
        key = candidate.key()
        current = deduped.get(key)
        if current is None or (
            candidate.score_hint,
            _source_priority(candidate.source),
        ) > (
            current.score_hint,
            _source_priority(current.source),
        ):
            deduped[key] = candidate

    ordered = sorted(deduped.values(), key=lambda action: action.score_hint, reverse=True)
    return ordered[:k]
