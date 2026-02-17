"""Game board generator: BFS DAG construction and target polynomial sampling.

Uses FastPoly (numpy-based) for all polynomial arithmetic, giving massive
speedup over SymPy for BFS exploration of reachable polynomials.
"""

import random
from typing import Dict, List, Optional, Tuple

from ..config import Config
from ..environment.fast_polynomial import FastPoly


def build_game_board(config: Config, complexity: int) -> Dict[bytes, dict]:
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

    # Initialize with base nodes
    initial_polys = []
    for i in range(n_vars):
        initial_polys.append(FastPoly.variable(i, n_vars, max_deg, mod))
    initial_polys.append(FastPoly.constant(1, n_vars, max_deg, mod))

    board = {}
    all_polys: List[FastPoly] = []

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

    # BFS layers
    for step in range(1, complexity + 1):
        new_polys = []
        n = len(all_polys)

        for i in range(n):
            for j in range(i, n):
                for op in (0, 1):  # 0=add, 1=mul
                    if op == 0:
                        result = all_polys[i] + all_polys[j]
                    else:
                        result = all_polys[i] * all_polys[j]

                    key = result.canonical_key()

                    parent_info = {
                        "op": "add" if op == 0 else "mul",
                        "left": all_polys[i].canonical_key(),
                        "right": all_polys[j].canonical_key(),
                    }

                    if key not in board:
                        board[key] = {
                            "poly": result,
                            "step": step,
                            "parents": [parent_info],
                            "paths": 1,
                        }
                        new_polys.append(result)
                    else:
                        board[key]["parents"].append(parent_info)
                        board[key]["paths"] += 1

        all_polys.extend(new_polys)

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
