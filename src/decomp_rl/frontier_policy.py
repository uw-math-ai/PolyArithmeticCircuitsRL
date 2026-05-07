"""Frontier-selection heuristics for unresolved subproblems."""

from __future__ import annotations

from .baseline_cost import BaselineCostModel
from .polynomial import SparsePolynomial


def choose_frontier_index(
    frontier: list[SparsePolynomial],
    baseline_model: BaselineCostModel,
    strategy: str = "largest_baseline",
) -> int:
    if not frontier:
        raise ValueError("Cannot choose from an empty frontier")

    if strategy == "largest_support":
        return max(range(len(frontier)), key=lambda index: frontier[index].support_size)
    if strategy == "largest_degree":
        return max(range(len(frontier)), key=lambda index: frontier[index].total_degree)
    return max(
        range(len(frontier)),
        key=lambda index: baseline_model.direct_construction_cost(frontier[index]),
    )

