"""Oracle action mask for diagnostic testing.

Restricts the agent's action space to only actions that lie on a shortest
path from the current circuit state to the target polynomial.  This is used
to determine whether the architecture / observation encoding is expressive
enough: if the agent still fails with an oracle mask, the bottleneck is
representation, not exploration.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
from sympy import expand

from ..core.poly import Poly, add as poly_add, mul as poly_mul, canonicalize

from .graph_enumeration import shortest_path_intermediates
from .samplers import _sympy_expr_to_poly

log = logging.getLogger(__name__)


def poly_to_hashkey(poly: Poly) -> tuple:
    """Canonical hashable key for an internal ``Poly`` dict."""
    return tuple(sorted(canonicalize(poly).items()))


class OracleMaskHelper:
    """Computes oracle action masks that restrict actions to optimal paths.

    Given the full game-board DAG (from ``build_game_graph``), this helper
    can determine, for any (target, current_circuit) pair, which ADD/MUL
    actions produce a polynomial that lies on a shortest path to the target.

    The helper pre-builds a mapping from internal ``Poly`` hash keys to DAG
    node IDs so that lookups during ``step()`` are fast.
    """

    def __init__(
        self,
        G: nx.DiGraph,
        dist: Dict[str, float],
        roots: Set[str],
        n_vars: int,
        var_names: List[str],
    ):
        self._G = G
        self._dist = dist
        self._roots = roots
        self._n_vars = n_vars
        self._var_names = var_names

        # Mapping: poly hashkey -> DAG node ID (canonical expr string)
        self._poly_to_dag_id: Dict[tuple, str] = {}
        self._build_poly_map()

        # Cache: target poly hashkey -> set of poly hashkeys on shortest paths
        self._target_cache: Dict[tuple, Set[tuple]] = {}

    def _build_poly_map(self) -> None:
        """Convert every DAG node's SymPy expr to internal Poly and store mapping."""
        converted = 0
        for node_id, data in self._G.nodes(data=True):
            expr = data.get("expr")
            if expr is None:
                continue
            try:
                poly = _sympy_expr_to_poly(str(expand(expr)), self._var_names)
                key = poly_to_hashkey(poly)
                self._poly_to_dag_id[key] = node_id
                converted += 1
            except Exception:
                continue
        log.info("OracleMaskHelper: mapped %d / %d DAG nodes to Poly keys",
                 converted, self._G.number_of_nodes())

    def _get_optimal_poly_keys(self, target_poly: Poly) -> Optional[Set[tuple]]:
        """Return set of poly hashkeys on shortest paths to *target_poly*, or None."""
        target_key = poly_to_hashkey(target_poly)

        cached = self._target_cache.get(target_key)
        if cached is not None:
            return cached

        target_dag_id = self._poly_to_dag_id.get(target_key)
        if target_dag_id is None:
            return None  # target not in DAG

        # Get DAG node IDs on shortest paths (excluding roots and target)
        intermediates = shortest_path_intermediates(
            self._G, target_dag_id, self._dist, self._roots,
        )
        # Include the target itself (agent needs to build it as the final step)
        goal_dag_ids = intermediates | {target_dag_id}

        # Also include root nodes — the agent may combine roots directly
        goal_dag_ids |= self._roots

        # Convert DAG node IDs back to poly hashkeys
        optimal_keys: Set[tuple] = set()
        for dag_id in goal_dag_ids:
            expr = self._G.nodes[dag_id].get("expr")
            if expr is None:
                continue
            try:
                poly = _sympy_expr_to_poly(str(expand(expr)), self._var_names)
                optimal_keys.add(poly_to_hashkey(poly))
            except Exception:
                continue

        self._target_cache[target_key] = optimal_keys
        return optimal_keys

    def get_optimal_actions(
        self,
        target_poly: Poly,
        current_node_polys: List[Poly],
    ) -> Optional[Set[Tuple[str, int, int]]]:
        """Return (op, i, j) tuples for actions whose result is on a shortest path.

        Parameters
        ----------
        target_poly : Poly
            The target polynomial for the current episode.
        current_node_polys : list of Poly
            Polynomials of all nodes currently in the circuit builder.

        Returns
        -------
        set of (op, i, j) or None
            Each element is ``("add", i, j)`` or ``("mul", i, j)`` where
            *i* and *j* are node indices.  Returns ``None`` if the target
            is not in the DAG (caller should fall back to the normal mask).
        """
        optimal_keys = self._get_optimal_poly_keys(target_poly)
        if optimal_keys is None:
            return None

        reachable: Set[Tuple[str, int, int]] = set()
        n = len(current_node_polys)
        for i in range(n):
            for j in range(i, n):
                add_result = poly_add(current_node_polys[i], current_node_polys[j])
                add_key = poly_to_hashkey(add_result)
                if add_key in optimal_keys:
                    reachable.add(("add", i, j))

                mul_result = poly_mul(current_node_polys[i], current_node_polys[j])
                mul_key = poly_to_hashkey(mul_result)
                if mul_key in optimal_keys:
                    reachable.add(("mul", i, j))

        return reachable
