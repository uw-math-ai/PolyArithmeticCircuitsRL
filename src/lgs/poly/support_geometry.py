"""Lightweight geometry diagnostics for polynomial supports."""

from __future__ import annotations

from math import prod

Exponent = tuple[int, ...]


def support_minkowski_sum(
    a: set[Exponent],
    b: set[Exponent],
) -> set[Exponent]:
    """Return pairwise exponent sums for two supports."""

    _require_same_dimension(a, b)
    if not a or not b:
        return set()
    return {
        tuple(left_i + right_i for left_i, right_i in zip(left, right))
        for left in a
        for right in b
    }


def support_minkowski_coverage(
    candidate_support: set[Exponent],
    node_support: set[Exponent],
    target_support: set[Exponent],
) -> tuple[float, float]:
    """Return target coverage and outside fraction for product-compatible support."""

    _require_same_dimension(candidate_support, node_support, target_support)
    minkowski = support_minkowski_sum(candidate_support, node_support)
    coverage = len(minkowski & target_support) / max(1, len(target_support))
    outside_fraction = len(minkowski - target_support) / max(1, len(minkowski))
    return float(coverage), float(outside_fraction)


def support_union_coverage(
    candidate_support: set[Exponent],
    node_support: set[Exponent],
    target_support: set[Exponent],
) -> tuple[float, float]:
    """Return target coverage and outside fraction for add-compatible support."""

    _require_same_dimension(candidate_support, node_support, target_support)
    union = candidate_support | node_support
    coverage = len(union & target_support) / max(1, len(target_support))
    outside_fraction = len(union - target_support) / max(1, len(union))
    return float(coverage), float(outside_fraction)


def support_affine_dimension(support: set[Exponent]) -> int:
    """Return the float-rank affine dimension of the exponent support."""

    _require_consistent_dimension(support)
    if len(support) <= 1:
        return 0

    points = sorted(support)
    origin = points[0]
    rows = [
        [float(value - base) for value, base in zip(point, origin)]
        for point in points[1:]
    ]
    return _matrix_rank(rows)


def support_bounding_box_volume(support: set[Exponent]) -> float:
    """Return the integer lattice volume of the support bounding box."""

    _require_consistent_dimension(support)
    if not support:
        return 0.0
    dimension = len(next(iter(support)))
    mins = [min(exponent[index] for exponent in support) for index in range(dimension)]
    maxs = [max(exponent[index] for exponent in support) for index in range(dimension)]
    return float(prod(max_value - min_value + 1 for min_value, max_value in zip(mins, maxs)))


def _matrix_rank(rows: list[list[float]]) -> int:
    if not rows:
        return 0
    matrix = [row[:] for row in rows]
    n_rows = len(matrix)
    n_cols = len(matrix[0])
    rank = 0
    tolerance = 1e-12

    for col in range(n_cols):
        pivot = None
        for row in range(rank, n_rows):
            if abs(matrix[row][col]) > tolerance:
                pivot = row
                break
        if pivot is None:
            continue
        matrix[rank], matrix[pivot] = matrix[pivot], matrix[rank]
        pivot_value = matrix[rank][col]
        matrix[rank] = [value / pivot_value for value in matrix[rank]]
        for row in range(n_rows):
            if row == rank:
                continue
            factor = matrix[row][col]
            if abs(factor) <= tolerance:
                continue
            matrix[row] = [
                value - factor * pivot_value
                for value, pivot_value in zip(matrix[row], matrix[rank])
            ]
        rank += 1
        if rank == n_rows:
            break
    return rank


def _require_same_dimension(*supports: set[Exponent]) -> None:
    dimensions = {
        len(exponent)
        for support in supports
        for exponent in support
    }
    if len(dimensions) > 1:
        raise ValueError("all support exponent tuples must have the same dimension")


def _require_consistent_dimension(support: set[Exponent]) -> None:
    _require_same_dimension(support)
