"""Flat action encoding and decoding for the circuit environment.

The full action space for a circuit with L visible nodes is a flat integer
index in [0, action_space_size(L)).  The layout is:

  [0,        pairs)        — ADD(i, j)        for all 0 <= i <= j < L
  [pairs,    2*pairs)      — MUL(i, j)        for all 0 <= i <= j < L
  [2*pairs,  2*pairs + L)  — SET_OUTPUT(i)    for node i
  [2*pairs + L]            — STOP

where pairs = L*(L+1)//2.

Node pairs (i, j) with i <= j are linearized using the upper-triangular
index: pair_to_index(i, j) = j*(j+1)//2 + i.  This packs all L*(L+1)/2
unordered pairs into a contiguous integer range.

Example (L=3):
  (0,0)->0  (0,1)->1  (1,1)->2  (0,2)->3  (1,2)->4  (2,2)->5
  ADD actions: [0..5], MUL actions: [6..11], SET_OUTPUT: [12,13,14], STOP: 15
"""

from __future__ import annotations

from dataclasses import dataclass
from math import floor, sqrt
from typing import Tuple

ACTION_ADD = "ADD"
ACTION_MUL = "MUL"
ACTION_SET_OUTPUT = "SET_OUTPUT"
ACTION_STOP = "STOP"


@dataclass(frozen=True)
class DecodedAction:
    """A decoded action with its type and optional node indices."""

    kind: str           # One of ACTION_ADD, ACTION_MUL, ACTION_SET_OUTPUT, ACTION_STOP
    i: int | None = None  # Left operand (ADD/MUL) or output node (SET_OUTPUT)
    j: int | None = None  # Right operand (ADD/MUL only); always >= i


def pair_to_index(i: int, j: int) -> int:
    """Map a node pair (i, j) with i <= j to a unique integer index.

    Uses the upper-triangular linearization: index = j*(j+1)//2 + i.
    This is the inverse of index_to_pair.
    """
    if i < 0 or j < 0 or i > j:
        raise ValueError(f"Invalid pair (i={i}, j={j})")
    return (j * (j + 1)) // 2 + i


def index_to_pair(p: int) -> Tuple[int, int]:
    """Map a flat pair index back to (i, j) with i <= j.

    Inverse of pair_to_index.  Uses the closed-form formula for triangular
    numbers to recover j, then i = p - j*(j+1)//2.
    """
    if p < 0:
        raise ValueError("Pair index must be >= 0")
    j = int(floor((sqrt(8 * p + 1) - 1) / 2))
    base = (j * (j + 1)) // 2
    i = p - base
    if i < 0 or i > j:
        raise ValueError("Invalid pair index")
    return i, j


def action_space_size(L: int) -> int:
    """Total number of distinct actions for a circuit with L visible nodes.

    Layout: pairs ADD + pairs MUL + L SET_OUTPUT + 1 STOP
    where pairs = L*(L+1)//2.
    """
    pairs = L * (L + 1) // 2
    return 2 * pairs + L + 1


def encode_action(kind: str, i: int | None, j: int | None, L: int) -> int:
    """Encode an action as a flat integer index in [0, action_space_size(L)).

    Args:
        kind: One of ACTION_ADD, ACTION_MUL, ACTION_SET_OUTPUT, ACTION_STOP.
        i:    Left operand node index (ADD/MUL) or output node (SET_OUTPUT).
        j:    Right operand node index (ADD/MUL, must satisfy j >= i).
        L:    Number of visible nodes (defines action space size).
    """
    pairs = L * (L + 1) // 2
    if kind == ACTION_ADD:
        if i is None or j is None:
            raise ValueError("ADD action requires i and j")
        return pair_to_index(i, j)
    if kind == ACTION_MUL:
        if i is None or j is None:
            raise ValueError("MUL action requires i and j")
        return pairs + pair_to_index(i, j)
    if kind == ACTION_SET_OUTPUT:
        if i is None:
            raise ValueError("SET_OUTPUT action requires i")
        return 2 * pairs + i
    if kind == ACTION_STOP:
        return 2 * pairs + L
    raise ValueError(f"Unknown action kind {kind}")


def decode_action(action_id: int, L: int) -> DecodedAction:
    """Decode a flat action index into a DecodedAction.

    Inverse of encode_action.  Raises ValueError for out-of-range ids.
    """
    pairs = L * (L + 1) // 2
    if action_id < 0 or action_id >= action_space_size(L):
        raise ValueError("Action id out of range")
    if action_id < pairs:
        i, j = index_to_pair(action_id)
        return DecodedAction(ACTION_ADD, i=i, j=j)
    if action_id < 2 * pairs:
        i, j = index_to_pair(action_id - pairs)
        return DecodedAction(ACTION_MUL, i=i, j=j)
    if action_id < 2 * pairs + L:
        i = action_id - 2 * pairs
        return DecodedAction(ACTION_SET_OUTPUT, i=i)
    return DecodedAction(ACTION_STOP)
