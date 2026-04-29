"""Bitset-based minimal-route enumeration for the on-path cache builder.

Routes are represented as Python ints used as bitsets: bit ``i`` is set iff
node id ``i`` is in the route. ``int`` operations are constant-time-ish for
the sizes that show up here (boards have a few thousand nodes, so bitsets
fit in a few hundred machine words). Compared with ``frozenset[int]``, this
gives a single bitwise OR per union, ``int.bit_count()`` for size, and free
hashing inside ``set[int]`` — the dominant cost in the legacy implementation
was ``frozenset.__hash__``.

The legacy ordering used for working-cap truncation was
``(len(route), tuple(sorted(int(node_id) for node_id in route)))`` —
``route_sort_key`` reproduces that bit-identically by peeling LSBs in ascending
order, so output caches match the frozenset version under truncation.
"""

from __future__ import annotations

from typing import Iterator, List, Optional, Sequence


def route_size(route: int) -> int:
    """Number of nodes in a route bitset."""
    return route.bit_count()


def route_contains(route: int, node_id: int) -> bool:
    """O(1) membership test."""
    return ((route >> node_id) & 1) == 1


def route_iter(route: int) -> Iterator[int]:
    """Yield set bit positions in ascending order via LSB peel."""
    while route:
        lsb = route & -route
        yield lsb.bit_length() - 1
        route ^= lsb


def route_sort_key(route: int) -> tuple[int, tuple[int, ...]]:
    """Sort key matching the legacy frozenset ordering exactly.

    Legacy ordering was ``(len(route), tuple(sorted(node_ids)))``. Bit-walking
    via :func:`route_iter` already yields IDs in ascending order, so the
    secondary key is identical.
    """
    return (route.bit_count(), tuple(route_iter(route)))


def route_minimal(routes: Sequence[int]) -> tuple[int, ...]:
    """Filter to routes of the minimum size."""
    if not routes:
        return ()
    min_cost = min(r.bit_count() for r in routes)
    return tuple(r for r in routes if r.bit_count() == min_cost)


def compute_sequential_route_sets_bitset(
    parents_by_id: Sequence[Sequence[tuple[int, int]]],
    base_ids: set[int],
    max_cost: int,
    working_route_cap: Optional[int] = None,
) -> tuple[tuple[tuple[int, ...], ...], set[int]]:
    """Enumerate route bitsets with at most ``max_cost`` non-base nodes.

    Mirror of the legacy frozenset version but with Python int bitsets. A route
    is a coherent set of non-base board nodes that can be constructed in a
    sequential episode; combining two parents unions their route sets (one
    bitwise OR) and adds the child bit. Shared subcomputations count once
    because OR is idempotent on overlapping bits.

    Args:
        parents_by_id: ``parents_by_id[child]`` is a list of ``(left, right)``
            parent-id tuples, one per board edge.
        base_ids: ids of base nodes (variables / constant).
        max_cost: maximum route size (in non-base nodes) to retain.
        working_route_cap: per-node cap on retained routes during enumeration.
            Routes are pruned by ``route_sort_key`` so the smallest survive.
            Capped nodes are recorded in the second return value.

    Returns:
        ``(routes_by_id, capped_nodes)``.

        ``routes_by_id[node_id]`` is a tuple of bitset ints, sorted by
        ``route_sort_key``, giving every route of size ≤ ``max_cost`` that
        produces ``node_id``.

        ``capped_nodes`` is the set of node ids whose working set hit the cap.
    """
    if max_cost < 0:
        raise ValueError("max_cost must be non-negative")

    n_nodes = len(parents_by_id)
    if working_route_cap is None:
        working_route_cap = 128

    routes_by_id: List[set[int]] = [set() for _ in range(n_nodes)]
    for base_id in base_ids:
        bid = int(base_id)
        if 0 <= bid < n_nodes:
            routes_by_id[bid].add(0)  # empty route bitset

    capped_nodes: set[int] = set()
    for _ in range(max_cost):
        changed = False
        for child in range(n_nodes):
            if child in base_ids:
                continue
            child_bit = 1 << child
            before = len(routes_by_id[child])
            for left, right in parents_by_id[child]:
                left_routes = tuple(routes_by_id[int(left)])
                right_routes = tuple(routes_by_id[int(right)])
                if not left_routes or not right_routes:
                    continue
                for left_route in left_routes:
                    if (left_route >> child) & 1:
                        continue
                    for right_route in right_routes:
                        if (right_route >> child) & 1:
                            continue
                        route = child_bit | left_route | right_route
                        if route.bit_count() <= max_cost:
                            routes_by_id[child].add(route)

            if len(routes_by_id[child]) > working_route_cap:
                capped_nodes.add(child)
                kept = sorted(routes_by_id[child], key=route_sort_key)[:working_route_cap]
                routes_by_id[child] = set(kept)

            if len(routes_by_id[child]) != before:
                changed = True
        if not changed:
            break

    return (
        tuple(tuple(sorted(routes, key=route_sort_key)) for routes in routes_by_id),
        capped_nodes,
    )
