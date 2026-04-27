"""Cached board-step on-path oracle sets.

The cache stores, for each target sampled from a BFS game board, the set of
non-base board nodes that lie on any optimal backward route to that target under
the board step/depth metric:

    max(step[left], step[right]) + 1 == step[child]

This is a teacher signal for curriculum experiments. It is not a proof of
globally minimal arithmetic circuit size.

Each cached node also stores a coherent-route bitmask. Runtime intersects these
masks after hits so reward credit stays on at least one compatible optimal route
instead of collecting unrelated nodes from the union of all routes.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from ..config import Config
from ..environment.fast_polynomial import FastPoly
from .generator import build_game_board


CACHE_VERSION = 2
STEP_METRIC = "board_step_depth_max_parent_plus_one"
CANONICALIZER_VERSION = "FastPoly.coeffs.int64.tobytes.v1"
OP_SET = ("add", "mul")
SPLIT_METHOD = "deterministic_shuffle"
SPLIT_RATIO = (0.8, 0.1, 0.1)
ROUTE_MASK_MODE = "coherent_optimal_route_masks_v1"


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
    target_board_step: int
    on_path_ids: np.ndarray
    on_path_coeffs: np.ndarray
    on_path_hashes: np.ndarray
    on_path_steps: np.ndarray
    on_path_route_masks: np.ndarray

    @property
    def on_path_keys(self) -> Dict[bytes, int]:
        """Canonical-key to board-step mapping for PyTorch O(1) lookups."""
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


def compute_on_path_ids(
    target_id: int,
    parents_by_id: Sequence[Sequence[tuple[int, int]]],
    node_steps: np.ndarray,
) -> set[int]:
    """Backward traverse optimal parent records for a target ID."""
    on_path = {int(target_id)}
    stack = [int(target_id)]

    while stack:
        child = stack.pop()
        child_step = int(node_steps[child])
        for left, right in parents_by_id[child]:
            if max(int(node_steps[left]), int(node_steps[right])) + 1 != child_step:
                continue
            for parent_id in (left, right):
                if parent_id not in on_path:
                    on_path.add(parent_id)
                    stack.append(parent_id)

    return on_path


def compute_on_path_route_masks(
    target_id: int,
    parents_by_id: Sequence[Sequence[tuple[int, int]]],
    node_steps: np.ndarray,
    base_ids: set[int],
    max_routes: int,
) -> Dict[int, np.uint32]:
    """Map nodes to coherent optimal-route bitmasks.

    Each bit represents one sampled/enumerated coherent optimal derivation tree
    for the target. Runtime intersects these masks after each hit, so once the
    agent takes credit for a node from one optimal branch it cannot also collect
    reward from nodes that only appear in disjoint optimal branches.
    """
    if max_routes <= 0:
        raise ValueError("max_routes must be positive")
    if max_routes > 32:
        raise ValueError("route masks use uint32, so max_routes must be <= 32")

    memo: Dict[int, tuple[frozenset[int], ...]] = {}

    def routes_for(node_id: int) -> tuple[frozenset[int], ...]:
        node_id = int(node_id)
        if node_id in memo:
            return memo[node_id]
        if node_id in base_ids:
            memo[node_id] = (frozenset(),)
            return memo[node_id]

        child_step = int(node_steps[node_id])
        routes: List[frozenset[int]] = []
        seen: set[frozenset[int]] = set()
        for left, right in parents_by_id[node_id]:
            if max(int(node_steps[left]), int(node_steps[right])) + 1 != child_step:
                continue
            for left_route in routes_for(left):
                for right_route in routes_for(right):
                    route = frozenset({node_id}) | left_route | right_route
                    if route in seen:
                        continue
                    seen.add(route)
                    routes.append(route)
                    if len(routes) >= max_routes:
                        memo[node_id] = tuple(routes)
                        return memo[node_id]

        if not routes:
            routes = [frozenset({node_id})]
        memo[node_id] = tuple(routes)
        return memo[node_id]

    masks: Dict[int, np.uint32] = {}
    for route_idx, route in enumerate(routes_for(target_id)[:max_routes]):
        bit = np.uint32(1 << route_idx)
        for node_id in route:
            masks[int(node_id)] = np.uint32(masks.get(int(node_id), np.uint32(0)) | bit)
    return masks


def build_complexity_cache(
    config: Config,
    complexity: int,
    split_seed: int,
    max_on_path_size: int,
    max_routes: int = 32,
) -> dict:
    """Build raw cache arrays for one complexity."""
    board = build_game_board(config, complexity)
    ordered_keys = sorted(board.keys(), key=lambda k: (board[k]["step"], k))
    key_to_id = {key: idx for idx, key in enumerate(ordered_keys)}

    node_steps = np.array([board[key]["step"] for key in ordered_keys], dtype=np.int32)
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

    base_ids = set(np.nonzero(node_steps == 0)[0].tolist())
    target_ids = np.array(
        [idx for idx, step in enumerate(node_steps) if int(step) == int(complexity)],
        dtype=np.int32,
    )
    if target_ids.size == 0:
        raise ValueError(f"No exact C{complexity} targets found in game board")

    offsets = [0]
    flat_ids: List[int] = []
    flat_route_masks: List[np.uint32] = []
    kept_target_ids: List[int] = []
    for target_id in target_ids:
        on_path = compute_on_path_ids(int(target_id), parents_by_id, node_steps)
        route_masks = compute_on_path_route_masks(
            int(target_id),
            parents_by_id,
            node_steps,
            base_ids,
            max_routes=max_routes,
        )
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
        flat_ids.extend(nonbase)
        flat_route_masks.extend(route_masks[int(idx)] for idx in nonbase)
        offsets.append(len(flat_ids))

    target_ids = np.array(kept_target_ids, dtype=np.int32)
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
        "route_count_cap": int(max_routes),
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
        "route_count_cap": int(config.on_path_num_routes),
    }
    for key, expected_value in expected.items():
        actual = metadata.get(key)
        if actual != expected_value:
            raise ValueError(
                f"on-path cache metadata mismatch for {key}: "
                f"expected {expected_value!r}, got {actual!r}"
            )
    if int(metadata.get("complexity", -1)) != int(requested_complexity):
        raise ValueError(
            f"cache complexity {metadata.get('complexity')} does not match "
            f"requested C{requested_complexity}"
        )


def build_caches(
    config: Config,
    complexities: Iterable[int],
    cache_dir: str | Path,
    split_seed: int,
    max_on_path_size: int,
    max_routes: int = 32,
) -> List[Path]:
    paths = []
    for complexity in sorted(set(int(c) for c in complexities)):
        arrays = build_complexity_cache(
            config=config,
            complexity=complexity,
            split_seed=split_seed,
            max_on_path_size=max_on_path_size,
            max_routes=max_routes,
        )
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
    )
    for path in paths:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
