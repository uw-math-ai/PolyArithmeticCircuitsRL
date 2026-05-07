"""Evaluation helpers for search and supervision metrics."""

from __future__ import annotations

from dataclasses import dataclass

from .andor_search import SearchResult
from .family_generators import SupervisedExample
from .train_supervised import SupervisedMetrics


@dataclass(frozen=True)
class SearchMetrics:
    average_best_cost: float
    average_search_gain: float
    solved_fraction: float
    average_transposition_hit_rate: float = 0.0
    average_factor_cache_hit_rate: float = 0.0
    average_branch_factor: float = 0.0


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


def summarize_supervised(metrics: SupervisedMetrics) -> dict[str, float]:
    return {
        "policy_loss": metrics.average_policy_loss,
        "value_loss": metrics.average_value_loss,
        "top1_accuracy": metrics.top1_accuracy,
        "example_count": float(metrics.example_count),
    }
