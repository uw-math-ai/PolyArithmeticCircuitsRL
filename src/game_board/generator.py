"""Game board generator: BFS DAG construction and target polynomial sampling.

Uses FastPoly (numpy-based) for all polynomial arithmetic, giving massive
speedup over SymPy for BFS exploration of reachable polynomials.
"""

import os
import random
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

from ..config import Config
from ..environment.fast_polynomial import FastPoly
from ..environment.poly_batch_ops import PolyBatchOps


def _interleave_pair_results(add_rows: np.ndarray, mul_rows: np.ndarray) -> np.ndarray:
    """Return rows in legacy pair-major order: add, mul, add, mul, ..."""
    candidates = np.empty(
        (add_rows.shape[0] + mul_rows.shape[0], add_rows.shape[1]),
        dtype=add_rows.dtype,
    )
    candidates[0::2] = add_rows
    candidates[1::2] = mul_rows
    return candidates


def _row_view(rows: np.ndarray) -> np.ndarray:
    """View a 2D contiguous array as one opaque value per row for np.unique."""
    if rows.ndim != 2:
        raise ValueError("rows must be a 2D array")
    contiguous = np.ascontiguousarray(rows)
    row_dtype = np.dtype((np.void, contiguous.dtype.itemsize * contiguous.shape[1]))
    return contiguous.view(row_dtype).reshape(-1)


def _iter_upper_tri_pair_chunks(
    n: int,
    chunk_size: int,
) -> Iterator[np.ndarray]:
    """Yield upper-triangular pair indices in legacy nested-loop order."""
    if chunk_size <= 0:
        raise ValueError("pair chunk size must be positive")

    pair_idx = np.empty((chunk_size, 2), dtype=np.int64)
    used = 0
    for i in range(n):
        j = i
        while j < n:
            take = min(chunk_size - used, n - j)
            pair_idx[used:used + take, 0] = i
            pair_idx[used:used + take, 1] = np.arange(j, j + take, dtype=np.int64)
            used += take
            j += take
            if used == chunk_size:
                yield pair_idx.copy()
                used = 0
    if used:
        yield pair_idx[:used].copy()


def build_game_board(
    config: Config,
    complexity: int,
    backend: str = "numpy",
    poly_ops: Optional[PolyBatchOps] = None,
    pair_chunk_size: Optional[int] = None,
) -> Dict[bytes, dict]:
    """Build a DAG of all reachable polynomials up to `complexity` operations.

    Uses BFS: layer 0 = initial nodes (variables + constant),
    layer k+1 = all new polynomials obtainable by combining two layer-<=k polys.

    Args:
        config: configuration object
        complexity: maximum number of operations

    Returns:
        Dictionary mapping canonical_key (bytes) -> {poly, step, parents, paths}
    """
    n_vars = config.n_variables
    mod = config.mod
    max_deg = config.effective_max_degree
    coeff_shape = (max_deg + 1,) * n_vars
    ops = poly_ops or PolyBatchOps(n_vars, max_deg, mod, backend=backend)
    if pair_chunk_size is None:
        pair_chunk_size = int(os.environ.get("POLY_BOARD_PAIR_CHUNK_SIZE", "100000"))
    if pair_chunk_size <= 0:
        raise ValueError("POLY_BOARD_PAIR_CHUNK_SIZE must be positive")

    # Initialize with base nodes
    initial_polys = []
    for i in range(n_vars):
        initial_polys.append(FastPoly.variable(i, n_vars, max_deg, mod))
    initial_polys.append(FastPoly.constant(1, n_vars, max_deg, mod))

    board = {}
    all_polys: List[FastPoly] = []
    all_keys: List[bytes] = []
    coeffs = np.empty((0, config.target_size), dtype=np.int64)

    for poly in initial_polys:
        key = poly.canonical_key()
        if key not in board:
            board[key] = {
                "poly": poly,
                "step": 0,
                "parents": [],
                "paths": 1,
            }
            all_polys.append(poly)
            all_keys.append(key)

    if all_polys:
        coeffs = np.stack(
            [poly.coeffs.reshape(-1).astype(np.int64, copy=False) for poly in all_polys],
            axis=0,
        )

    # BFS layers
    for step in range(1, complexity + 1):
        new_polys: List[FastPoly] = []
        new_keys: List[bytes] = []
        new_coeff_rows: List[np.ndarray] = []
        n = len(all_polys)

        for pair_idx in _iter_upper_tri_pair_chunks(n, pair_chunk_size):
            add_rows = ops.add_all_pairs(coeffs, pair_idx)
            mul_rows = ops.mul_all_pairs(coeffs, pair_idx)
            candidate_rows = _interleave_pair_results(add_rows, mul_rows)
            candidate_view = _row_view(candidate_rows)

            _unique_rows, first_indices, inverse = np.unique(
                candidate_view,
                return_index=True,
                return_inverse=True,
            )
            path_counts = np.bincount(inverse, minlength=len(first_indices))
            unique_ids_by_first_seen = np.argsort(first_indices, kind="stable")

            unique_id_to_key: Dict[int, bytes] = {}
            for unique_id in range(len(first_indices)):
                first_idx = int(first_indices[unique_id])
                unique_id_to_key[unique_id] = candidate_rows[first_idx].tobytes()

            for unique_id in unique_ids_by_first_seen:
                key = unique_id_to_key[int(unique_id)]
                if key in board:
                    board[key]["paths"] += int(path_counts[unique_id])
                    continue
                first_idx = int(first_indices[unique_id])
                coeff_row = candidate_rows[first_idx].copy()
                poly = FastPoly(coeff_row.reshape(coeff_shape).copy(), mod)
                board[key] = {
                    "poly": poly,
                    "step": step,
                    "parents": [],
                    "paths": int(path_counts[unique_id]),
                }
                new_polys.append(poly)
                new_keys.append(key)
                new_coeff_rows.append(coeff_row)

            for candidate_idx, unique_id in enumerate(inverse):
                pair_pos = candidate_idx // 2
                op_name = "add" if candidate_idx % 2 == 0 else "mul"
                left_idx = int(pair_idx[pair_pos, 0])
                right_idx = int(pair_idx[pair_pos, 1])
                key = unique_id_to_key[int(unique_id)]
                board[key]["parents"].append({
                    "op": op_name,
                    "left": all_keys[left_idx],
                    "right": all_keys[right_idx],
                })

        all_polys.extend(new_polys)
        all_keys.extend(new_keys)
        if new_coeff_rows:
            coeffs = np.concatenate(
                [coeffs, np.stack(new_coeff_rows, axis=0)],
                axis=0,
            )

    return board


def find_interesting_targets(board: Dict[bytes, dict], min_paths: int = 2) -> List[dict]:
    """Find polynomials reachable via multiple distinct circuits.

    Args:
        board: game board from build_game_board
        min_paths: minimum number of construction paths

    Returns:
        List of board entries sorted by (step, -paths)
    """
    interesting = []
    for key, entry in board.items():
        if entry["paths"] >= min_paths and entry["step"] > 0:
            interesting.append({
                "key": key,
                "poly": entry["poly"],
                "step": entry["step"],
                "paths": entry["paths"],
            })

    interesting.sort(key=lambda e: (e["step"], -e["paths"]))
    return interesting


def sample_target(config: Config, complexity: int,
                  board: Optional[Dict[bytes, dict]] = None) -> Tuple[FastPoly, int]:
    """Sample a target polynomial of given complexity.

    Args:
        config: configuration
        complexity: target complexity (number of operations)
        board: pre-built game board (optional, will build if None)

    Returns:
        (target_polynomial, min_steps_to_build)
    """
    if board is None:
        board = build_game_board(config, complexity)

    candidates = [entry for entry in board.values() if entry["step"] == complexity]

    if not candidates:
        candidates = [entry for entry in board.values() if 0 < entry["step"] <= complexity]

    if not candidates:
        poly, _ = generate_random_circuit(config, complexity)
        return poly, complexity

    entry = random.choice(candidates)
    return entry["poly"], entry["step"]


def generate_random_circuit(config: Config, complexity: int) -> Tuple[FastPoly, List[Tuple[int, int, int]]]:
    """Generate a random circuit of given complexity.

    Args:
        config: configuration
        complexity: number of operations to perform

    Returns:
        (final_polynomial, action_sequence) where actions are (op, i, j) tuples
    """
    n_vars = config.n_variables
    mod = config.mod
    max_deg = config.effective_max_degree

    # Start with base nodes
    nodes: List[FastPoly] = []
    for i in range(n_vars):
        nodes.append(FastPoly.variable(i, n_vars, max_deg, mod))
    nodes.append(FastPoly.constant(1, n_vars, max_deg, mod))

    actions = []

    for _ in range(complexity):
        n = len(nodes)
        i = random.randint(0, n - 1)
        j = random.randint(i, n - 1)
        op = random.randint(0, 1)

        if op == 0:
            result = nodes[i] + nodes[j]
        else:
            result = nodes[i] * nodes[j]

        nodes.append(result)
        actions.append((op, i, j))

    return nodes[-1], actions
