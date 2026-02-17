"""Action space encoding/decoding for circuit construction.

Actions encode (operation, node_i, node_j) as a single integer.
Pairs (i, j) with i <= j are mapped via upper-triangular indexing.
Each pair gets 2 slots: add (even index) and multiply (odd index).
"""

import math
from typing import Tuple

import torch


def compute_max_actions(max_nodes: int) -> int:
    """Total number of possible actions.

    Number of pairs with i <= j: max_nodes * (max_nodes + 1) / 2
    Times 2 for add/multiply: max_nodes * (max_nodes + 1)
    """
    return max_nodes * (max_nodes + 1)


def encode_action(op: int, i: int, j: int, max_nodes: int) -> int:
    """Encode (op, node_i, node_j) into a single action index.

    Args:
        op: 0 for add, 1 for multiply
        i: first node index (i <= j will be enforced)
        j: second node index
        max_nodes: maximum number of nodes in the circuit

    Returns:
        Integer action index
    """
    if i > j:
        i, j = j, i

    # Pair index: number of pairs before row i, plus offset within row i
    # Pairs with first element < i: sum_{k=0}^{i-1} (max_nodes - k) = i*max_nodes - i*(i-1)/2
    pair_idx = i * max_nodes - i * (i - 1) // 2 + (j - i)
    return 2 * pair_idx + op


def decode_action(action_idx: int, max_nodes: int) -> Tuple[int, int, int]:
    """Decode action index into (op, node_i, node_j).

    Uses closed-form triangular number inverse for O(1) decoding.

    Args:
        action_idx: integer action index
        max_nodes: maximum number of nodes

    Returns:
        (op, i, j) where op is 0 (add) or 1 (multiply), i <= j
    """
    op = action_idx % 2
    pair_idx = action_idx // 2

    # Closed-form inverse of pair_idx = i * max_nodes - i*(i-1)/2 + (j - i)
    # Pair index for the start of row i: f(i) = i * max_nodes - i*(i-1)/2
    # We need to find i such that f(i) <= pair_idx < f(i+1)
    # f(i) = i * (2*max_nodes - i + 1) / 2
    # Solving: i^2 - (2*max_nodes + 1)*i + 2*pair_idx = 0
    # i = ((2*max_nodes + 1) - sqrt((2*max_nodes + 1)^2 - 8*pair_idx)) / 2

    discriminant = (2 * max_nodes + 1) ** 2 - 8 * pair_idx
    i = int((2 * max_nodes + 1 - math.isqrt(discriminant)) // 2)

    # Clamp i in case of floating point edge cases
    row_start = i * max_nodes - i * (i - 1) // 2
    # Verify and adjust
    while row_start > pair_idx and i > 0:
        i -= 1
        row_start = i * max_nodes - i * (i - 1) // 2
    next_row_start = (i + 1) * max_nodes - (i + 1) * i // 2
    while pair_idx >= next_row_start and i < max_nodes - 1:
        i += 1
        row_start = i * max_nodes - i * (i - 1) // 2
        next_row_start = (i + 1) * max_nodes - (i + 1) * i // 2

    j = i + (pair_idx - row_start)
    return op, i, j


def get_valid_actions_mask(num_current_nodes: int, max_nodes: int) -> torch.BoolTensor:
    """Generate a boolean mask of valid actions given the current number of nodes.

    An action (op, i, j) is valid if both i < num_current_nodes and j < num_current_nodes.

    Args:
        num_current_nodes: number of nodes currently in the circuit
        max_nodes: maximum number of nodes

    Returns:
        Boolean tensor of shape (max_actions,) where True means valid
    """
    max_actions = compute_max_actions(max_nodes)
    mask = torch.zeros(max_actions, dtype=torch.bool)

    for i in range(num_current_nodes):
        for j in range(i, num_current_nodes):
            for op in (0, 1):
                idx = encode_action(op, i, j, max_nodes)
                mask[idx] = True

    return mask
