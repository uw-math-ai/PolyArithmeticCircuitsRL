"""Arithmetic circuit builder with signature-based deduplication.

CircuitBuilder incrementally constructs an arithmetic circuit (DAG) by
adding nodes one at a time.  It deduplicates equivalent nodes using a
canonical signature so that, e.g., ADD(x0, x1) and ADD(x1, x0) produce
the same node (commutativity is handled by sorting operand signatures).

Usage:
    builder = CircuitBuilder(n_vars=2, eval_points=pts)
    r = builder.add_add(0, 1)   # r.node_id is the new/reused node
    builder.set_output(r.node_id)
    output_poly = builder.nodes[builder.output_node].poly
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Dict, List, Optional, Tuple

from .fingerprints import eval_poly_points
from .poly import Poly, add, make_const, make_var, mul
from .node import Node, OP_ADD, OP_CONST, OP_MUL, OP_VAR


@dataclass
class BuildResult:
    """Result of adding a node to the circuit."""

    node_id: int  # ID of the newly created or reused node
    reused: bool  # True if a structurally identical node already existed


class CircuitBuilder:
    """Incrementally builds an arithmetic circuit with structural deduplication.

    Leaf nodes (variables and constants) are pre-populated in __init__.
    Operation nodes (ADD, MUL) are added via add_add / add_mul.

    Deduplication: every new node is assigned a canonical `signature` tuple.
    For ADD/MUL, operand signatures are sorted so that ADD(a,b) == ADD(b,a).
    Before inserting, the builder checks if an identical signature already
    exists and returns that node's ID instead of creating a duplicate.
    """

    def __init__(
        self,
        n_vars: int,
        eval_points: Optional[List[Tuple[int, ...]]] = None,
        include_const_zero: bool = False,
        include_const_one: bool = True,
    ):
        """Initialize the circuit with input variable nodes.

        Args:
            n_vars:             Number of polynomial variables.
            eval_points:        Fixed evaluation points; if provided, each node
                                stores its precomputed eval vector in node.evals.
            include_const_zero: Add a constant-0 leaf node.
            include_const_one:  Add a constant-1 leaf node (default True).
        """
        self.n_vars = n_vars
        self.eval_points = eval_points
        self.nodes: List[Node] = []
        self.signature_to_id: Dict[Tuple, int] = {}
        self.num_add = 0    # count of unique ADD nodes created
        self.num_mul = 0    # count of unique MUL nodes created
        self.depth = 0      # maximum circuit depth seen so far
        self.output_node: Optional[int] = None

        # Pre-populate leaves
        for i in range(n_vars):
            self._add_input(i)
        if include_const_zero:
            self._add_const(0)
        if include_const_one:
            self._add_const(1)

    def _evals_for_poly(self, poly: Poly) -> Optional[Tuple[Fraction, ...]]:
        if self.eval_points is None:
            return None
        evals = eval_poly_points(poly, self.eval_points)
        return tuple(evals)

    def _add_node(self, op: str, args: Tuple[int, int] | Tuple[()], poly: Poly, depth: int) -> BuildResult:
        signature = self._signature(op, args)
        existing = self.signature_to_id.get(signature)
        if existing is not None:
            return BuildResult(node_id=existing, reused=True)

        node_id = len(self.nodes)
        evals = self._evals_for_poly(poly)
        node = Node(
            node_id=node_id,
            op=op,
            args=args,
            poly=poly,
            depth=depth,
            signature=signature,
            evals=evals,
        )
        self.nodes.append(node)
        self.signature_to_id[signature] = node_id
        self.depth = max(self.depth, depth)
        return BuildResult(node_id=node_id, reused=False)

    def _signature(self, op: str, args: Tuple[int, int] | Tuple[()]) -> Tuple:
        """Compute a canonical deduplication key for a node.

        For ADD/MUL, the two operand signatures are sorted so that
        ADD(a,b) and ADD(b,a) produce the same signature (commutativity).
        Leaf signatures are just (op, value).
        """
        if op == OP_VAR:
            return (OP_VAR, args[0])
        if op == OP_CONST:
            return (OP_CONST, args[0])
        if op in (OP_ADD, OP_MUL):
            left_id, right_id = args
            left_sig = self.nodes[left_id].signature
            right_sig = self.nodes[right_id].signature
            # Sort so commutativity is handled automatically
            if left_sig <= right_sig:
                ordered = (left_sig, right_sig)
            else:
                ordered = (right_sig, left_sig)
            return (op, ordered[0], ordered[1])
        raise ValueError(f"Unknown op {op}")

    def _add_input(self, var_idx: int) -> int:
        poly = make_var(self.n_vars, var_idx)
        result = self._add_node(OP_VAR, (var_idx,), poly, depth=0)
        return result.node_id

    def _add_const(self, value: int) -> int:
        poly = make_const(self.n_vars, value)
        result = self._add_node(OP_CONST, (value,), poly, depth=0)
        return result.node_id

    def add_add(self, left_id: int, right_id: int) -> BuildResult:
        """Add a new ADD node computing nodes[left_id] + nodes[right_id].

        Returns a BuildResult with the node_id and whether it was deduplicated.
        """
        self._validate_ids(left_id, right_id)
        left = self.nodes[left_id]
        right = self.nodes[right_id]
        poly = add(left.poly, right.poly)
        depth = max(left.depth, right.depth) + 1
        result = self._add_node(OP_ADD, (left_id, right_id), poly, depth)
        if not result.reused:
            self.num_add += 1
        return result

    def add_mul(self, left_id: int, right_id: int) -> BuildResult:
        """Add a new MUL node computing nodes[left_id] * nodes[right_id].

        Returns a BuildResult with the node_id and whether it was deduplicated.
        """
        self._validate_ids(left_id, right_id)
        left = self.nodes[left_id]
        right = self.nodes[right_id]
        poly = mul(left.poly, right.poly)
        depth = max(left.depth, right.depth) + 1
        result = self._add_node(OP_MUL, (left_id, right_id), poly, depth)
        if not result.reused:
            self.num_mul += 1
        return result

    def set_output(self, node_id: int) -> None:
        """Designate node_id as the circuit's output node."""
        if node_id < 0 or node_id >= len(self.nodes):
            raise ValueError(f"Invalid output node {node_id}")
        self.output_node = node_id

    def _validate_ids(self, left_id: int, right_id: int) -> None:
        max_id = len(self.nodes) - 1
        if left_id < 0 or right_id < 0 or left_id > max_id or right_id > max_id:
            raise ValueError("Operand id out of range")
