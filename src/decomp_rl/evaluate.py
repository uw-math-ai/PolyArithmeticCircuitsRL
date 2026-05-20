"""Evaluation helpers for search and supervision metrics."""

from __future__ import annotations

from dataclasses import dataclass
from random import Random

from .andor_search import SearchResult
from .baseline_cost import BaselineCostModel
from .decomp_env import DecompEnv
from .family_generators import SupervisedExample
from .polynomial import SparsePolynomial
from .train_supervised import SupervisedMetrics


@dataclass(frozen=True)
class SearchMetrics:
    average_best_cost: float
    average_search_gain: float
    solved_fraction: float
    average_transposition_hit_rate: float = 0.0
    average_factor_cache_hit_rate: float = 0.0
    average_branch_factor: float = 0.0


@dataclass(frozen=True)
class RandomRolloutMetrics:
    average_cost: float
    average_gain: float
    solved_fraction: float
    average_steps: float
    rollout_count: int


def summarize_search_results(results: list[SearchResult], baseline_costs: list[float]) -> SearchMetrics:
    if not results:
        return SearchMetrics(0.0, 0.0, 0.0)
    total_best = 0.0
    total_gain = 0.0
    solved = 0
    total_transposition_hit_rate = 0.0
    total_factor_cache_hit_rate = 0.0
    total_branch_factor = 0.0
    for result, baseline in zip(results, baseline_costs):
        total_best += result.best_cost
        total_gain += baseline - result.best_cost
        if result.best_cost <= baseline:
            solved += 1
        total_transposition_hit_rate += result.stats.transposition_hit_rate
        total_factor_cache_hit_rate += result.stats.factor_cache_hit_rate
        total_branch_factor += result.stats.average_branch_factor
    count = len(results)
    return SearchMetrics(
        average_best_cost=total_best / count,
        average_search_gain=total_gain / count,
        solved_fraction=solved / count,
        average_transposition_hit_rate=total_transposition_hit_rate / count,
        average_factor_cache_hit_rate=total_factor_cache_hit_rate / count,
        average_branch_factor=total_branch_factor / count,
    )


def random_rollout_cost(
    target: SparsePolynomial,
    env: DecompEnv,
    rng: Random,
    k_candidates: int = 16,
    max_steps: int = 32,
) -> tuple[int, int]:
    """Roll out a uniformly random split policy and return (cost, split_steps).

    If the random policy runs out of steps or finds no split candidates, any
    remaining frontier items are solved directly so every rollout produces a
    complete circuit cost. This is intentionally policy-free: it measures how
    much the learned/search-guided policy improves over random split choices.
    """
    state = env.reset(target)
    split_steps = 0
    for _ in range(max_steps):
        if not state.frontier:
            break
        candidates = env.get_candidate_splits(state, 0, k_candidates)
        if not candidates:
            state, _, _, _ = env.solve_direct(state, 0)
            continue
        action = rng.choice(candidates)
        state, _, _, _ = env.step(state, 0, action)
        split_steps += 1

    while state.frontier:
        state, _, _, _ = env.solve_direct(state, 0)
    return state.acc_cost, split_steps


def evaluate_random_rollouts(
    targets: list[SparsePolynomial],
    baseline_model: BaselineCostModel | None = None,
    rollouts_per_target: int = 1,
    seed: int = 0,
    k_candidates: int = 16,
    max_steps: int = 32,
) -> RandomRolloutMetrics:
    """Evaluate a uniformly random split policy on a target set."""
    if not targets or rollouts_per_target <= 0:
        return RandomRolloutMetrics(0.0, 0.0, 0.0, 0.0, 0)

    baseline_model = baseline_model or BaselineCostModel()
    env = DecompEnv(baseline_model=baseline_model)
    rng = Random(seed)
    total_cost = 0.0
    total_gain = 0.0
    total_steps = 0.0
    solved = 0
    rollout_count = 0

    for target in targets:
        baseline = float(baseline_model.direct_construction_cost(target))
        for _ in range(rollouts_per_target):
            cost, steps = random_rollout_cost(
                target,
                env,
                rng,
                k_candidates=k_candidates,
                max_steps=max_steps,
            )
            total_cost += float(cost)
            total_gain += baseline - float(cost)
            total_steps += float(steps)
            solved += int(float(cost) <= baseline)
            rollout_count += 1

    return RandomRolloutMetrics(
        average_cost=total_cost / rollout_count,
        average_gain=total_gain / rollout_count,
        solved_fraction=solved / rollout_count,
        average_steps=total_steps / rollout_count,
        rollout_count=rollout_count,
    )


def summarize_supervised(metrics: SupervisedMetrics) -> dict[str, float]:
    return {
        "policy_loss": metrics.average_policy_loss,
        "value_loss": metrics.average_value_loss,
        "top1_accuracy": metrics.top1_accuracy,
        "example_count": float(metrics.example_count),
    }
