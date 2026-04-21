"""Circuit node dataclass.

A Node represents one value in the arithmetic circuit DAG.  Nodes are
created by CircuitBuilder and are never mutated after construction.

Op types and their args fields:
  VAR    args = (var_idx,)         — input variable x_{var_idx}
  CONST  args = (value,)           — integer constant
  ADD    args = (left_id, right_id) — addition of two earlier nodes
  MUL    args = (left_id, right_id) — multiplication of two earlier nodes
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Optional, Tuple

from .poly import Poly

# Operation type constants — used as node.op values.
OP_VAR = "VAR"
OP_CONST = "CONST"
OP_ADD = "ADD"
OP_MUL = "MUL"


@dataclass
class Node:
    """One node in the arithmetic circuit DAG.

    Attributes:
        node_id:   Unique integer index assigned by CircuitBuilder (0-indexed).
        op:        Operation type: one of OP_VAR, OP_CONST, OP_ADD, OP_MUL.
        args:      Operand indices or leaf value (see module docstring for format).
        poly:      Exact polynomial computed at this node (used for solve detection).
        depth:     Circuit depth (0 for leaves, max(parent depths)+1 for ops).
        signature: Canonical tuple used for deduplication (commutative-safe).
        evals:     Optional precomputed evaluation vector for the m fixed points;
                   set by CircuitBuilder when eval_points are provided.
    """

    node_id: int
    op: str
    args: Tuple[int, int] | Tuple[()]  # empty for VAR/CONST
    poly: Poly
    depth: int
    signature: Tuple
    evals: Optional[Tuple[Fraction, ...]] = None
