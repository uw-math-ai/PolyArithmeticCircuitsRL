"""Target polynomial samplers for training.

RandomCircuitSampler: generates random polynomials by building random circuits.
InterestingPolynomialSampler: loads pre-computed polynomials with multiple
    optimal circuits from Game-Board-Generation analysis JSONL files.
"""

from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..core.builder import CircuitBuilder
from ..core.poly import Poly


class TargetSampler:
    """Base class for target polynomial samplers."""

    def sample(self, rng: random.Random) -> Tuple[Poly, dict]:
        raise NotImplementedError


@dataclass
class RandomCircuitSampler(TargetSampler):
    """Generates random target polynomials by building random circuits."""

    n_vars: int
    max_steps: int

    def sample(self, rng: random.Random) -> Tuple[Poly, dict]:
        builder = CircuitBuilder(self.n_vars, eval_points=None)
        last_id = len(builder.nodes) - 1
        for _ in range(self.max_steps):
            i = rng.randint(0, last_id)
            j = rng.randint(i, last_id)
            if rng.random() < 0.5:
                res = builder.add_add(i, j)
            else:
                res = builder.add_mul(i, j)
            last_id = max(last_id, res.node_id)
        n_initial = self.n_vars + 1  # leaves (vars + const_1)
        if last_id >= n_initial:
            target_id = rng.randint(n_initial, last_id)  # only op nodes
        else:
            target_id = rng.randint(0, last_id)  # fallback
        builder.set_output(target_id)
        target_poly = builder.nodes[target_id].poly
        return target_poly, {"source": "random_circuit", "target_id": target_id}


def _sympy_expr_to_poly(expr_str: str, var_names: List[str]) -> Poly:
    """Convert a SymPy expression string to our internal Poly format."""
    import sympy

    symbols = sympy.symbols(var_names)
    if not isinstance(symbols, tuple):
        symbols = (symbols,)

    expr = sympy.sympify(expr_str)
    sp = sympy.Poly(expr, *symbols)

    poly: Poly = {}
    for monom, coeff in sp.as_dict().items():
        poly[monom] = Fraction(coeff)

    return poly


def _detect_variables(records: List[Dict]) -> List[str]:
    """Detect variable names from expression strings in the dataset."""
    all_vars = set()
    for rec in records[:100]:
        expr = rec["expr_str"]
        found = re.findall(r'\b([a-z]\d*)\b', expr)
        for v in found:
            if v not in ("exp", "log", "sin", "cos", "tan"):
                all_vars.add(v)
    return sorted(all_vars)


class InterestingPolynomialSampler:
    """Samples interesting polynomials from pre-computed analysis data.

    Interesting = has multiple shortest paths (multiple optimal circuits).
    Polynomials are grouped by shortest_length for curriculum filtering.
    """

    def __init__(
        self,
        jsonl_path: str,
        n_vars: int,
        only_multipath: bool = True,
    ):
        self.n_vars = n_vars
        path = Path(jsonl_path)

        records = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

        self.var_names = _detect_variables(records)
        if len(self.var_names) != n_vars:
            pass  # mismatch is tolerated

        if only_multipath:
            records = [
                r for r in records
                if r.get("multiple_shortest_paths", False) or r.get("multiple_paths", False)
            ]

        records = [r for r in records if r.get("shortest_length", 0) > 0]

        self.by_length: Dict[int, List[Tuple[Poly, Dict]]] = defaultdict(list)
        self._all_polys: List[Tuple[Poly, Dict]] = []

        for rec in records:
            try:
                poly = _sympy_expr_to_poly(rec["expr_str"], self.var_names)
                entry = (poly, rec)
                self._all_polys.append(entry)
                self.by_length[rec["shortest_length"]].append(entry)
            except Exception:
                continue

    def sample(self, rng: random.Random, max_ops: int) -> Tuple[Poly, Optional[Dict]]:
        """Sample a polynomial with shortest_length <= max_ops."""
        candidates = []
        for length, entries in self.by_length.items():
            if length <= max_ops:
                candidates.extend(entries)

        if not candidates:
            if self._all_polys:
                poly, meta = rng.choice(self._all_polys)
                return poly, meta
            raise ValueError("No polynomials loaded")

        poly, meta = rng.choice(candidates)
        return poly, meta

    def __len__(self) -> int:
        return len(self._all_polys)


class GenerativeInterestingPolynomialSampler:
    """Auto-generates interesting polynomials via graph enumeration at init time.

    Lazily builds the game-board DAG on first ``sample()`` call (or when
    ``max_ops`` exceeds a previously built level), converts SymPy expressions
    to internal ``Poly`` format, and caches them grouped by shortest-path
    length — same interface as ``InterestingPolynomialSampler``.
    """

    def __init__(
        self,
        n_vars: int,
        max_steps: int = 4,
        only_multipath: bool = True,
        max_graph_nodes: Optional[int] = None,
        max_successors_per_node: Optional[int] = None,
    ):
        self.n_vars = n_vars
        self.max_steps = max_steps
        self.only_multipath = only_multipath
        self.max_graph_nodes = max_graph_nodes
        self.max_successors_per_node = max_successors_per_node

        self.var_names: List[str] = (
            ["x"] if n_vars == 1 else [f"x{i}" for i in range(n_vars)]
        )

        self.by_length: Dict[int, List[Tuple[Poly, Dict]]] = defaultdict(list)
        self._all_polys: List[Tuple[Poly, Dict]] = []
        self._built_up_to: int = 0  # highest step level already built

    def _ensure_built(self, needed_steps: int) -> None:
        """Lazily build the game graph up to *needed_steps* if not already done."""
        if needed_steps <= self._built_up_to:
            return

        from .graph_enumeration import analyze_graph, build_game_graph

        target_steps = min(needed_steps, self.max_steps)
        if target_steps <= self._built_up_to:
            return

        G = build_game_graph(
            steps=target_steps,
            num_vars=self.n_vars,
            max_nodes=self.max_graph_nodes,
            max_successors_per_node=self.max_successors_per_node,
        )
        records = analyze_graph(
            G,
            only_multipath=self.only_multipath,
            max_step=target_steps,
        )

        # Reset caches — a graph at step N subsumes all lower steps
        self.by_length.clear()
        self._all_polys.clear()

        for rec in records:
            sl = rec.get("shortest_length")
            if sl is None or sl <= 0:
                continue
            try:
                poly = _sympy_expr_to_poly(rec["expr_str"], self.var_names)
                entry = (poly, rec)
                self._all_polys.append(entry)
                self.by_length[sl].append(entry)
            except Exception:
                continue

        self._built_up_to = target_steps

    def sample(self, rng: random.Random, max_ops: int) -> Tuple[Poly, Optional[Dict]]:
        """Sample a polynomial with ``shortest_length <= max_ops``."""
        self._ensure_built(max_ops)

        candidates = []
        for length, entries in self.by_length.items():
            if length <= max_ops:
                candidates.extend(entries)

        if not candidates:
            if self._all_polys:
                poly, meta = rng.choice(self._all_polys)
                return poly, meta
            raise ValueError(
                "No interesting polynomials generated. "
                "Try increasing max_steps or disabling only_multipath."
            )

        poly, meta = rng.choice(candidates)
        return poly, meta

    def __len__(self) -> int:
        return len(self._all_polys)
