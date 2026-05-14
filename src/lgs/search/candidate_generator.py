"""Fresh target-conditioned two-tier candidate generation."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Hashable

from lgs.env.action import Action
from lgs.env.candidate import Candidate
from lgs.env.circuit_state import CircuitState
from lgs.env.problem_instance import ProblemInstance
from lgs.poly.fast_poly import Polynomial, PolynomialDegreeError
from lgs.poly.poly_utils import exact_divides, require_same_domain
from lgs.search.heuristic_score import score_tier1, score_tier2


def generate_candidates(
    instance: ProblemInstance,
    state: CircuitState,
    K: int = 64,
    tier2_m: int = 128,
) -> list[Candidate]:
    _validate_inputs(instance, state, K, tier2_m)
    if state.remaining_budget() <= 0:
        return []

    tier1_candidates: list[Candidate] = []
    for candidate in enumerate_basic_pair_candidates(instance, state):
        if _is_safely_filtered(instance, state, candidate.result_poly):
            continue
        compute_tier1_features(instance, state, candidate)
        candidate.tier1_score = score_tier1(candidate.features)
        candidate.heuristic_score = candidate.tier1_score
        candidate.total_score = candidate.heuristic_score
        tier1_candidates.append(candidate)

    unique_candidates = unique_by_result_polynomial(tier1_candidates)
    tier2_candidates = sorted(
        unique_candidates,
        key=_tier1_sort_key,
    )[:tier2_m]

    for candidate in tier2_candidates:
        compute_tier2_features(instance, state, candidate)
        candidate.tier2_score = score_tier2(candidate.features)
        candidate.heuristic_score = candidate.tier1_score + candidate.tier2_score
        candidate.total_score = candidate.heuristic_score

    return sorted(unique_candidates, key=_final_sort_key)[:K]


def enumerate_basic_pair_candidates(
    instance: ProblemInstance,
    state: CircuitState,
) -> list[Candidate]:
    _validate_inputs(instance, state, K=1, tier2_m=1)
    if state.remaining_budget() <= 0:
        return []

    candidates: list[Candidate] = []
    for i in range(len(state.nodes)):
        for j in range(i, len(state.nodes)):
            for op in ("add", "mul"):
                action = Action.make(op, i, j)
                try:
                    next_state = state.apply(action)
                except PolynomialDegreeError:
                    continue
                candidates.append(
                    Candidate(
                        action=action,
                        result_poly=next_state.nodes[-1],
                        source_tags={"basic_pair"},
                        features={},
                    )
                )
    return candidates


def compute_tier1_features(
    instance: ProblemInstance,
    state: CircuitState,
    candidate: Candidate,
) -> dict[str, float]:
    require_same_domain(instance.target, candidate.result_poly)

    result = candidate.result_poly
    target = instance.target
    supp_g = result.support()
    supp_f = target.support()
    overlap = supp_g & supp_f
    residual = target - result
    residual_support = residual.support()

    features = candidate.features
    features["equals_target"] = float(result == target)
    features["degree_result"] = float(result.degree())
    features["degree_target"] = float(target.degree())
    features["degree_gap"] = float(target.degree() - result.degree())
    features["support_size_result"] = float(len(supp_g))
    features["support_size_target"] = float(len(supp_f))
    features["support_overlap_count"] = float(len(overlap))
    features["support_overlap_frac"] = float(len(overlap) / max(1, len(supp_g)))
    features["target_coverage_frac"] = float(len(overlap) / max(1, len(supp_f)))
    features["outside_support_count"] = float(len(supp_g - supp_f))
    features["residual_exists"] = float(residual.key() in state.node_keys)
    features["residual_support_size"] = float(len(residual_support))
    features["residual_target_overlap"] = float(len(residual_support & supp_f))

    if features["equals_target"]:
        candidate.source_tags.add("exact_target")
    if features["support_overlap_count"] > 0:
        candidate.source_tags.add("support_overlap")
    if features["residual_exists"]:
        candidate.source_tags.add("residual_exists")
    return features


def compute_tier2_features(
    instance: ProblemInstance,
    state: CircuitState,
    candidate: Candidate,
) -> dict[str, float]:
    require_same_domain(instance.target, candidate.result_poly)

    result = candidate.result_poly
    features = candidate.features
    features["divides_target"] = 0.0
    features["quotient_exists"] = 0.0
    features["quotient_degree"] = 0.0
    features["quotient_support_size"] = 0.0
    features["one_step_completion_add"] = 0.0
    features["one_step_completion_mul"] = 0.0

    if not result.is_zero():
        quotient = exact_divides(instance.target, result)
        if quotient is not None:
            features["divides_target"] = 1.0
            features["quotient_exists"] = float(quotient.key() in state.node_keys)
            features["quotient_degree"] = float(quotient.degree())
            features["quotient_support_size"] = float(len(quotient.support()))
            candidate.source_tags.add("divides_target")
            if features["quotient_exists"]:
                candidate.source_tags.add("quotient_exists")

    for node in (*state.nodes, result):
        if result + node == instance.target:
            features["one_step_completion_add"] = 1.0
            candidate.source_tags.add("one_step_completion_add")
        try:
            product_completes = result * node == instance.target
        except PolynomialDegreeError:
            product_completes = False
        if product_completes:
            features["one_step_completion_mul"] = 1.0
            candidate.source_tags.add("one_step_completion_mul")

    return features


def unique_by_result_polynomial(candidates: Iterable[Candidate]) -> list[Candidate]:
    by_key: dict[Hashable, Candidate] = {}
    for candidate in candidates:
        key = candidate.result_poly.key()
        existing = by_key.get(key)
        if existing is None or _dedupe_sort_key(candidate) < _dedupe_sort_key(existing):
            by_key[key] = candidate
    return list(by_key.values())


def _is_safely_filtered(
    instance: ProblemInstance,
    state: CircuitState,
    result: Polynomial,
) -> bool:
    require_same_domain(instance.target, result)
    if result.is_zero() and result != instance.target:
        return True
    if result.key() in state.node_keys and result != instance.target:
        return True
    return False


def _validate_inputs(
    instance: ProblemInstance,
    state: CircuitState,
    K: int,
    tier2_m: int,
) -> None:
    if not isinstance(instance, ProblemInstance):
        raise TypeError("instance must be a ProblemInstance")
    if not isinstance(state, CircuitState):
        raise TypeError("state must be a CircuitState")
    if type(K) is not int or K < 0:
        raise ValueError("K must be a non-negative int")
    if type(tier2_m) is not int or tier2_m < 0:
        raise ValueError("tier2_m must be a non-negative int")
    require_same_domain(instance.target, state.nodes[0])


def _action_tie_key(candidate: Candidate) -> tuple[str, int, int]:
    return (candidate.action.op, candidate.action.i, candidate.action.j)


def _dedupe_sort_key(candidate: Candidate) -> tuple[float, tuple[str, int, int]]:
    return (-candidate.tier1_score, _action_tie_key(candidate))


def _tier1_sort_key(candidate: Candidate) -> tuple[float, tuple[str, int, int], Hashable]:
    return (-candidate.tier1_score, _action_tie_key(candidate), candidate.result_poly.key())


def _final_sort_key(candidate: Candidate) -> tuple[float, tuple[str, int, int], Hashable]:
    return (-candidate.heuristic_score, _action_tie_key(candidate), candidate.result_poly.key())


# TODO: Add target factorization over F_p in a future milestone, without relying
# on old factor-library state or non-exact symbolic approximations.
