"""Cached sequential-route OnPath oracle sets.

The board generator is used as an overcomplete polynomial source, but clean
OnPath cache complexity is the episode action count: the number of non-base
nodes that must be constructed along one coherent route. Shared subcomputations
count once, so ``(x + y)^2`` has route cost 2 while
``(x + y) * (x + 1)`` has route cost 3.

Each cached node also stores a coherent-route bitmask. Runtime intersects these
masks after hits so reward credit stays on at least one compatible optimal route
instead of collecting unrelated nodes from the union of all routes.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

from ..config import Config
from ..environment.fast_polynomial import FastPoly
from ..environment.poly_batch_ops import PolyBatchOps
from .generator import build_game_board
from .parallel_build import map_complexities
from .route_enum import (
    compute_sequential_route_sets_bitset,
    route_iter,
)


CACHE_VERSION = 5
STEP_METRIC = "sequential_route_size_nonbase_nodes"
CANONICALIZER_VERSION = "FastPoly.coeffs.int64.tobytes.v1"
OP_SET = ("add", "mul")
SPLIT_METHOD = "deterministic_shuffle"
SPLIT_RATIO = (0.8, 0.1, 0.1)
ROUTE_MASK_MODE = "coherent_optimal_route_masks_v1"
ROUTE_ENUMERATION_MODE = "bitset_minimal_routes_v1"
ROUTE_WORKING_CAP_MULTIPLIER = 4
ROUTE_WORKING_CAP_MIN = 128


def hash_weights_np(target_size: int) -> np.ndarray:
    """Deterministic uint32 weights mirrored by the JAX environment."""
    idx = np.arange(target_size, dtype=np.uint32)
    return (idx * np.uint32(2654435761) + np.uint32(2246822519)).astype(np.uint32)


def hash_coeff_vector(coeffs: np.ndarray) -> np.uint32:
    """Hash a flat coefficient vector for fast JAX-side prefiltering.

    Hash hits are never trusted by themselves; JAX verifies coefficient equality
    before marking an on-path node as hit.
    """
    vals = coeffs.astype(np.uint32, copy=False) + np.uint32(1)
    return np.bitwise_xor.reduce(vals * hash_weights_np(vals.shape[0]))


def hash_coeff_matrix(coeffs: np.ndarray) -> np.ndarray:
    weights = hash_weights_np(coeffs.shape[1])
    vals = coeffs.astype(np.uint32, copy=False) + np.uint32(1)
    return np.bitwise_xor.reduce(vals * weights[None, :], axis=1).astype(np.uint32)


@dataclass(frozen=True)
class OnPathTargetContext:
    """Per-target on-path data used by environments at reset time."""

    target_id: int
    target_poly: FastPoly
    # Kept as target_board_step for wire compatibility. In v4 caches this is
    # the minimal sequential route size, not board depth.
    target_board_step: int
    on_path_ids: np.ndarray
    on_path_coeffs: np.ndarray
    on_path_hashes: np.ndarray
    on_path_steps: np.ndarray
    on_path_route_masks: np.ndarray

    @property
    def on_path_keys(self) -> Dict[bytes, int]:
        """Canonical-key to sequential-step mapping for PyTorch O(1) lookups."""
        return {
            self.on_path_coeffs[i].astype(np.int64, copy=False).tobytes(): int(step)
            for i, step in enumerate(self.on_path_steps)
        }

    @property
    def on_path_route_keys(self) -> Dict[bytes, int]:
        """Canonical-key to coherent-route bitmask mapping for PyTorch."""
        return {
            self.on_path_coeffs[i].astype(np.int64, copy=False).tobytes(): int(mask)
            for i, mask in enumerate(self.on_path_route_masks)
        }


@dataclass(frozen=True)
class OnPathComplexityCache:
    complexity: int
    metadata: dict
    node_coeffs: np.ndarray
    node_steps: np.ndarray
    node_hashes: np.ndarray
    target_ids: np.ndarray
    train_target_ids: np.ndarray
    val_target_ids: np.ndarray
    test_target_ids: np.ndarray
    on_path_offsets: np.ndarray
    on_path_flat_ids: np.ndarray
    on_path_route_masks: np.ndarray
    route_cap_hit: np.ndarray
    n_variables: int
    mod: int
    max_degree: int

    def target_context(self, target_id: int) -> OnPathTargetContext:
        target_id = int(target_id)
        matches = np.nonzero(self.target_ids == target_id)[0]
        if matches.size == 0:
            raise KeyError(f"target_id {target_id} is not in C{self.complexity} cache")

        idx = int(matches[0])
        start = int(self.on_path_offsets[idx])
        end = int(self.on_path_offsets[idx + 1])
        on_path_ids = self.on_path_flat_ids[start:end].astype(np.int32, copy=False)
        on_path_route_masks = self.on_path_route_masks[start:end].astype(
            np.uint32, copy=False
        )
        if on_path_ids.size == 0:
            raise ValueError(f"target_id {target_id} has an empty non-base on-path set")

        shape = (self.max_degree + 1,) * self.n_variables
        target_poly = FastPoly(
            self.node_coeffs[target_id].reshape(shape).copy(),
            self.mod,
        )
        return OnPathTargetContext(
            target_id=target_id,
            target_poly=target_poly,
            target_board_step=int(self.node_steps[target_id]),
            on_path_ids=on_path_ids,
            on_path_coeffs=self.node_coeffs[on_path_ids].copy(),
            on_path_hashes=self.node_hashes[on_path_ids].copy(),
            on_path_steps=self.node_steps[on_path_ids].copy(),
            on_path_route_masks=on_path_route_masks.copy(),
        )

    def sample_train_context(self, rng: Optional[np.random.Generator] = None) -> OnPathTargetContext:
        if self.train_target_ids.size == 0:
            raise ValueError(f"C{self.complexity} cache has no train target IDs")
        rng = rng or np.random.default_rng()
        target_id = int(rng.choice(self.train_target_ids))
        return self.target_context(target_id)


class OnPathCache:
    """Loaded collection of per-complexity on-path cache files."""

    def __init__(self, by_complexity: Dict[int, OnPathComplexityCache]):
        self.by_complexity = dict(by_complexity)

    @classmethod
    def load(
        cls,
        cache_dir: str | Path,
        config: Config,
        complexities: Sequence[int],
    ) -> "OnPathCache":
        cache_dir = Path(cache_dir)
        if not cache_dir.exists():
            raise FileNotFoundError(f"on-path cache directory not found: {cache_dir}")

        by_complexity = {}
        for complexity in sorted(set(int(c) for c in complexities)):
            path = cache_file_path(cache_dir, complexity)
            if not path.exists():
                raise FileNotFoundError(
                    f"missing on-path cache for C{complexity}: {path}"
                )
            comp_cache = load_complexity_cache(path)
            validate_cache_metadata(comp_cache.metadata, config, requested_complexity=complexity)
            max_size = int(config.on_path_max_size)
            if max_size > 0:
                lengths = np.diff(comp_cache.on_path_offsets)
                too_large = lengths > max_size
                if np.any(too_large):
                    raise ValueError(
                        f"C{complexity} cache has on-path sets larger than "
                        f"on_path_max_size={max_size}"
                    )
            by_complexity[complexity] = comp_cache

        return cls(by_complexity)

    def sample_train_context(
        self,
        complexity: int,
        rng: Optional[np.random.Generator] = None,
    ) -> OnPathTargetContext:
        return self.by_complexity[int(complexity)].sample_train_context(rng)

    def pack_jax_contexts(
        self,
        contexts: Sequence[Optional[OnPathTargetContext]],
        max_size: int,
        target_size: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Pack variable-length contexts into fixed arrays for JAX reset."""
        batch = len(contexts)
        coeffs = np.zeros((batch, max_size, target_size), dtype=np.int32)
        hashes = np.zeros((batch, max_size), dtype=np.uint32)
        steps = np.zeros((batch, max_size), dtype=np.int32)
        route_masks = np.zeros((batch, max_size), dtype=np.uint32)
        active = np.zeros((batch, max_size), dtype=bool)
        target_steps = np.zeros((batch,), dtype=np.int32)

        for i, ctx in enumerate(contexts):
            if ctx is None:
                continue
            n = len(ctx.on_path_ids)
            if n == 0:
                raise ValueError(f"target_id {ctx.target_id} has no on-path nodes")
            if n > max_size:
                raise ValueError(
                    f"target_id {ctx.target_id} has {n} on-path nodes; "
                    f"max_size={max_size}"
                )
            coeffs[i, :n] = ctx.on_path_coeffs.astype(np.int32, copy=False)
            hashes[i, :n] = ctx.on_path_hashes
            steps[i, :n] = ctx.on_path_steps
            route_masks[i, :n] = ctx.on_path_route_masks
            active[i, :n] = True
            target_steps[i] = ctx.target_board_step

        return coeffs, hashes, steps, route_masks, active, target_steps


def cache_file_path(cache_dir: str | Path, complexity: int) -> Path:
    return Path(cache_dir) / f"on_path_C{int(complexity)}.npz"


def compute_sequential_route_sets(
    parents_by_id: Sequence[Sequence[tuple[int, int]]],
    base_ids: set[int],
    max_cost: int,
    working_route_cap: Optional[int] = None,
) -> tuple[tuple[tuple[int, ...], ...], set[int]]:
    """Enumerate minimal route bitsets per node.

    Thin wrapper around :func:`route_enum.compute_sequential_route_sets_bitset`.
    Routes are Python ``int`` bitsets; bit ``i`` is set iff node ``i`` is in
    the route. See ``route_enum.py`` for ordering and truncation guarantees.
    """
    if working_route_cap is None:
        working_route_cap = max(ROUTE_WORKING_CAP_MIN, ROUTE_WORKING_CAP_MULTIPLIER)
    return compute_sequential_route_sets_bitset(
        parents_by_id,
        base_ids,
        max_cost=max_cost,
        working_route_cap=working_route_cap,
    )


def _minimal_routes(routes: Sequence[int]) -> tuple[int, ...]:
    if not routes:
        return tuple()
    min_cost = min(r.bit_count() for r in routes)
    return tuple(r for r in routes if r.bit_count() == min_cost)


def _route_masks_from_minimal_routes(
    minimal_routes: Sequence[int],
    max_routes: int,
    capped_nodes: Optional[set[int]] = None,
    target_id: Optional[int] = None,
) -> tuple[Dict[int, np.uint32], bool]:
    """Build per-node route bitmasks from a list of minimal coherent routes.

    Routes are Python ``int`` bitsets. Returns ``(masks, route_cap_hit)``.
    ``route_cap_hit`` is conservative — it fires if any of the following hold:

      (a) more minimal routes existed than ``max_routes`` (overflow at selection),
      (b) the target's own route enumeration was working-cap truncated,
      (c) any node appearing in a selected route was working-cap truncated
          during enumeration. Ancestor truncation can silently drop minimal
          routes from the target, so we surface it as a cap hit too.
    """
    selected = tuple(minimal_routes[:max_routes])
    masks: Dict[int, np.uint32] = {}
    for route_idx, route in enumerate(selected):
        bit = np.uint32(1 << route_idx)
        for node_id in route_iter(route):
            masks[node_id] = np.uint32(masks.get(node_id, np.uint32(0)) | bit)

    cap_overflow = len(minimal_routes) > max_routes
    target_capped = (
        capped_nodes is not None
        and target_id is not None
        and int(target_id) in capped_nodes
    )
    ancestor_capped = (
        capped_nodes is not None
        and any(
            node_id in capped_nodes
            for route in selected
            for node_id in route_iter(route)
        )
    )
    return masks, bool(cap_overflow or target_capped or ancestor_capped)


def compute_on_path_ids(
    target_id: int,
    parents_by_id: Sequence[Sequence[tuple[int, int]]],
    node_steps: np.ndarray,
) -> set[int]:
    """Return non-base nodes on minimal sequential routes for a target ID.

    NOTE: this re-enumerates routes for ALL nodes (cost O(global)) rather than
    walking only the target's subtree. Cheap for one-off inspection; do NOT
    call in tight loops. The cache builder inlines route enumeration once via
    ``compute_sequential_route_sets`` and reuses results across all targets.
    """
    node_steps_arr = np.asarray(node_steps)
    base_ids = set(np.nonzero(node_steps_arr == 0)[0].tolist())
    positive_steps = node_steps_arr[node_steps_arr > 0]
    max_cost = int(positive_steps.max()) if positive_steps.size else 0
    routes_by_id, _ = compute_sequential_route_sets(
        parents_by_id,
        base_ids,
        max_cost=max_cost,
    )
    routes = _minimal_routes(routes_by_id[int(target_id)])
    on_path: set[int] = set()
    for route in routes:
        on_path.update(route_iter(route))
    return on_path


def compute_on_path_route_masks(
    target_id: int,
    parents_by_id: Sequence[Sequence[tuple[int, int]]],
    node_steps: np.ndarray,
    base_ids: set[int],
    max_routes: int,
) -> tuple[Dict[int, np.uint32], bool]:
    """Map nodes to coherent minimal sequential-route bitmasks.

    Each bit represents one sampled/enumerated coherent optimal derivation tree
    for the target. Runtime intersects these masks after each hit, so once the
    agent takes credit for a node from one optimal branch it cannot also collect
    reward from nodes that only appear in disjoint optimal branches.
    """
    if max_routes <= 0:
        raise ValueError("max_routes must be positive")
    if max_routes > 32:
        raise ValueError("route masks use uint32, so max_routes must be <= 32")

    node_steps_arr = np.asarray(node_steps)
    positive_steps = node_steps_arr[node_steps_arr > 0]
    max_cost = int(positive_steps.max()) if positive_steps.size else 0
    routes_by_id, capped_nodes = compute_sequential_route_sets(
        parents_by_id,
        base_ids,
        max_cost=max_cost,
    )
    minimal_routes = _minimal_routes(routes_by_id[int(target_id)])
    return _route_masks_from_minimal_routes(
        minimal_routes,
        max_routes=max_routes,
        capped_nodes=capped_nodes,
        target_id=int(target_id),
    )


def build_complexity_cache(
    config: Config,
    complexity: int,
    split_seed: int,
    max_on_path_size: int,
    max_routes: int = 32,
    board_backend: str = "numpy",
) -> dict:
    """Build raw cache arrays for one complexity."""
    poly_ops = PolyBatchOps(
        config.n_variables,
        config.effective_max_degree,
        config.mod,
        backend=board_backend,
    )
    board = build_game_board(config, complexity, poly_ops=poly_ops)
    ordered_keys = sorted(board.keys(), key=lambda k: (board[k]["step"], k))
    key_to_id = {key: idx for idx, key in enumerate(ordered_keys)}

    board_steps = np.array([board[key]["step"] for key in ordered_keys], dtype=np.int32)
    node_coeffs = np.stack(
        [board[key]["poly"].coeffs.flatten().astype(np.int64) for key in ordered_keys],
        axis=0,
    )
    node_hashes = hash_coeff_matrix(node_coeffs)

    parents_by_id: List[List[tuple[int, int]]] = [[] for _ in ordered_keys]
    for key, entry in board.items():
        child_id = key_to_id[key]
        for parent in entry["parents"]:
            left = key_to_id[parent["left"]]
            right = key_to_id[parent["right"]]
            parents_by_id[child_id].append((left, right))

    base_ids = set(np.nonzero(board_steps == 0)[0].tolist())
    working_cap = max(ROUTE_WORKING_CAP_MIN, max_routes * ROUTE_WORKING_CAP_MULTIPLIER)
    routes_by_id, capped_nodes = compute_sequential_route_sets(
        parents_by_id,
        base_ids,
        max_cost=int(complexity),
        working_route_cap=working_cap,
    )

    node_steps = np.full(len(ordered_keys), -1, dtype=np.int32)
    for base_id in base_ids:
        node_steps[int(base_id)] = 0
    for idx, routes in enumerate(routes_by_id):
        if routes:
            node_steps[idx] = min(route.bit_count() for route in routes)

    target_ids = np.array(
        [
            idx
            for idx, step in enumerate(node_steps)
            if int(step) == int(complexity) and idx not in base_ids
        ],
        dtype=np.int32,
    )
    if target_ids.size == 0:
        raise ValueError(f"No exact sequential C{complexity} targets found in game board")

    offsets = [0]
    flat_ids: List[int] = []
    flat_route_masks: List[np.uint32] = []
    kept_target_ids: List[int] = []
    route_cap_hits: List[bool] = []
    for target_id in target_ids:
        minimal_routes = tuple(
            route for route in routes_by_id[int(target_id)]
            if route.bit_count() == int(complexity)
        )
        if not minimal_routes:
            continue
        route_masks, route_cap_hit = _route_masks_from_minimal_routes(
            minimal_routes,
            max_routes=max_routes,
            capped_nodes=capped_nodes,
            target_id=int(target_id),
        )
        on_path = set(route_masks)
        nonbase = sorted(
            (idx for idx in on_path - base_ids if int(route_masks.get(int(idx), 0)) != 0),
            key=lambda idx: (node_steps[idx], idx),
        )
        if not nonbase:
            raise ValueError(f"C{complexity} target {target_id} has empty non-base OnPath")
        if max_on_path_size > 0 and len(nonbase) > max_on_path_size:
            raise ValueError(
                f"C{complexity} target {target_id} OnPath size {len(nonbase)} "
                f"exceeds max_on_path_size={max_on_path_size}"
            )
        kept_target_ids.append(int(target_id))
        route_cap_hits.append(bool(route_cap_hit))
        flat_ids.extend(nonbase)
        flat_route_masks.extend(route_masks[int(idx)] for idx in nonbase)
        offsets.append(len(flat_ids))

    target_ids = np.array(kept_target_ids, dtype=np.int32)
    route_cap_hit_array = np.array(route_cap_hits, dtype=bool)
    route_cap_hit_rate = (
        float(route_cap_hit_array.mean()) if route_cap_hit_array.size else 0.0
    )
    route_working_cap_hit_node_rate = (
        float(len(capped_nodes) / max(len(ordered_keys), 1))
    )
    train_ids, val_ids, test_ids = split_target_ids(target_ids, split_seed + complexity)

    metadata = {
        "cache_version": CACHE_VERSION,
        "n_variables": int(config.n_variables),
        "mod": int(config.mod),
        "max_degree": int(config.effective_max_degree),
        "target_size": int(config.target_size),
        "complexity": int(complexity),
        "op_set": list(OP_SET),
        "canonicalizer_version": CANONICALIZER_VERSION,
        "include_constant": True,
        "base_node_count": int(config.n_variables + 1),
        "step_metric": STEP_METRIC,
        "split_seed": int(split_seed),
        "split_method": SPLIT_METHOD,
        "split_ratio": list(SPLIT_RATIO),
        "route_mask_mode": ROUTE_MASK_MODE,
        "route_enumeration_mode": ROUTE_ENUMERATION_MODE,
        "route_count_cap": int(max_routes),
        "route_working_cap": int(working_cap),
        "route_working_cap_hit_node_count": int(len(capped_nodes)),
        "route_working_cap_hit_node_rate": route_working_cap_hit_node_rate,
        "route_cap_hit_count": int(route_cap_hit_array.sum()),
        "route_cap_hit_rate": route_cap_hit_rate,
        "board_backend": poly_ops.backend,
    }

    return {
        "metadata_json": np.array(json.dumps(metadata, sort_keys=True)),
        "node_coeffs": node_coeffs,
        "node_steps": node_steps,
        "node_hashes": node_hashes,
        "target_ids": target_ids,
        "train_target_ids": train_ids,
        "val_target_ids": val_ids,
        "test_target_ids": test_ids,
        "on_path_offsets": np.array(offsets, dtype=np.int64),
        "on_path_flat_ids": np.array(flat_ids, dtype=np.int32),
        "on_path_route_masks": np.array(flat_route_masks, dtype=np.uint32),
        "route_cap_hit": route_cap_hit_array,
    }


def split_target_ids(target_ids: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ids = np.array(target_ids, dtype=np.int32, copy=True)
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n = len(ids)
    if n == 0:
        return ids, ids, ids

    n_train = max(1, int(np.floor(SPLIT_RATIO[0] * n)))
    n_val = int(np.floor(SPLIT_RATIO[1] * n))
    if n_train + n_val > n:
        n_val = max(0, n - n_train)
    train = np.sort(ids[:n_train])
    val = np.sort(ids[n_train:n_train + n_val])
    test = np.sort(ids[n_train + n_val:])
    return train, val, test


def save_complexity_cache(cache_dir: str | Path, arrays: dict, complexity: int) -> Path:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_file_path(cache_dir, complexity)
    np.savez_compressed(path, **arrays)
    return path


def load_complexity_cache(path: str | Path) -> OnPathComplexityCache:
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        if "on_path_route_masks" not in data.files:
            raise ValueError(
                f"{path} was built without coherent-route masks; rebuild the "
                "OnPath cache with the current cache builder"
            )
        if "route_cap_hit" not in data.files:
            raise ValueError(
                f"{path} was built without route-cap diagnostics; rebuild the "
                "OnPath cache with the current cache builder"
            )
        return OnPathComplexityCache(
            complexity=int(metadata["complexity"]),
            metadata=metadata,
            node_coeffs=data["node_coeffs"].copy(),
            node_steps=data["node_steps"].copy(),
            node_hashes=data["node_hashes"].copy(),
            target_ids=data["target_ids"].copy(),
            train_target_ids=data["train_target_ids"].copy(),
            val_target_ids=data["val_target_ids"].copy(),
            test_target_ids=data["test_target_ids"].copy(),
            on_path_offsets=data["on_path_offsets"].copy(),
            on_path_flat_ids=data["on_path_flat_ids"].copy(),
            on_path_route_masks=data["on_path_route_masks"].copy(),
            route_cap_hit=data["route_cap_hit"].copy(),
            n_variables=int(metadata["n_variables"]),
            mod=int(metadata["mod"]),
            max_degree=int(metadata["max_degree"]),
        )


def validate_cache_metadata(metadata: dict, config: Config, requested_complexity: int) -> None:
    expected = {
        "cache_version": CACHE_VERSION,
        "n_variables": int(config.n_variables),
        "mod": int(config.mod),
        "max_degree": int(config.effective_max_degree),
        "target_size": int(config.target_size),
        "op_set": list(OP_SET),
        "canonicalizer_version": CANONICALIZER_VERSION,
        "include_constant": True,
        "base_node_count": int(config.n_variables + 1),
        "step_metric": STEP_METRIC,
        "route_mask_mode": ROUTE_MASK_MODE,
        "route_enumeration_mode": ROUTE_ENUMERATION_MODE,
        "route_count_cap": int(config.on_path_num_routes),
    }
    for key, expected_value in expected.items():
        actual = metadata.get(key)
        if actual != expected_value:
            raise ValueError(
                f"on-path cache metadata mismatch for {key}: "
                f"expected {expected_value!r}, got {actual!r}"
            )
    if "board_backend" not in metadata:
        raise ValueError(
            "on-path cache metadata mismatch for board_backend: missing required key"
        )
    if metadata["board_backend"] not in ("numpy", "jax"):
        raise ValueError(
            "on-path cache metadata mismatch for board_backend: "
            f"expected 'numpy' or 'jax', got {metadata['board_backend']!r}"
        )
    if int(metadata.get("complexity", -1)) != int(requested_complexity):
        raise ValueError(
            f"cache complexity {metadata.get('complexity')} does not match "
            f"requested C{requested_complexity}"
        )


def _config_kwargs(config: Config) -> dict[str, Any]:
    return {field.name: getattr(config, field.name) for field in fields(Config)}


def _build_one_complexity(item: dict[str, Any]) -> tuple[int, dict]:
    config = Config(**item["config_kwargs"])
    complexity = int(item["complexity"])
    arrays = build_complexity_cache(
        config=config,
        complexity=complexity,
        split_seed=int(item["split_seed"]),
        max_on_path_size=int(item["max_on_path_size"]),
        max_routes=int(item["max_routes"]),
        board_backend=str(item["board_backend"]),
    )
    metadata = json.loads(str(arrays["metadata_json"].item()))
    truncation_rate = float(metadata.get("route_cap_hit_rate", 0.0))
    max_route_truncation_rate = float(item["max_route_truncation_rate"])
    if truncation_rate > max_route_truncation_rate:
        raise ValueError(
            f"C{complexity} route cap hit rate {truncation_rate:.2%} "
            f"exceeds max_route_truncation_rate={max_route_truncation_rate:.2%}; "
            "increase --num-routes or narrow the target set"
        )
    return complexity, arrays


def build_caches(
    config: Config,
    complexities: Iterable[int],
    cache_dir: str | Path,
    split_seed: int,
    max_on_path_size: int,
    max_routes: int = 32,
    max_route_truncation_rate: float = 0.25,
    board_backend: str = "numpy",
    force_serial: bool = False,
    num_processes: Optional[int] = None,
) -> List[Path]:
    work_items = [
        {
            "config_kwargs": _config_kwargs(config),
            "complexity": complexity,
            "split_seed": split_seed,
            "max_on_path_size": max_on_path_size,
            "max_routes": max_routes,
            "max_route_truncation_rate": max_route_truncation_rate,
            "board_backend": board_backend,
        }
        for complexity in sorted(set(int(c) for c in complexities))
    ]

    paths = []
    for complexity, arrays in map_complexities(
        _build_one_complexity,
        work_items,
        processes=num_processes,
        force_serial=force_serial,
    ):
        paths.append(save_complexity_cache(cache_dir, arrays, complexity))
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build cached OnPath oracle sets.")
    parser.add_argument("--complexities", type=int, nargs="+", required=True)
    parser.add_argument("--n-variables", type=int, required=True)
    parser.add_argument("--mod", type=int, required=True)
    parser.add_argument("--max-degree", type=int, required=True)
    parser.add_argument("--cache-dir", type=str, required=True)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--max-on-path-size", type=int, default=4096)
    parser.add_argument("--num-routes", type=int, default=32)
    parser.add_argument("--max-route-truncation-rate", type=float, default=0.25)
    parser.add_argument("--backend", choices=("numpy", "jax"), default="numpy")
    parser.add_argument("--no-parallel", action="store_true")
    parser.add_argument("--num-processes", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = Config(
        n_variables=args.n_variables,
        mod=args.mod,
        max_complexity=max(args.complexities),
        max_degree=args.max_degree,
        on_path_split_seed=args.split_seed,
        on_path_max_size=args.max_on_path_size,
        on_path_num_routes=args.num_routes,
    )
    paths = build_caches(
        config=config,
        complexities=args.complexities,
        cache_dir=args.cache_dir,
        split_seed=args.split_seed,
        max_on_path_size=args.max_on_path_size,
        max_routes=args.num_routes,
        max_route_truncation_rate=args.max_route_truncation_rate,
        board_backend=args.backend,
        force_serial=args.no_parallel,
        num_processes=args.num_processes,
    )
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata_json"].item()))
        print(
            f"Wrote {path} "
            f"(route_cap_hit_rate={metadata.get('route_cap_hit_rate', 0.0):.2%}, "
            f"route_cap_hit_count={metadata.get('route_cap_hit_count', 0)}, "
            f"route_working_cap_hit_node_count="
            f"{metadata.get('route_working_cap_hit_node_count', 0)}, "
            f"route_working_cap_hit_node_rate="
            f"{metadata.get('route_working_cap_hit_node_rate', 0.0):.2%})"
        )


if __name__ == "__main__":
    main()
