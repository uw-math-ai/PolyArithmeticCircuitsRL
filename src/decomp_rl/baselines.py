"""Top-down baselines for arithmetic-circuit cost over F_p.

These complement :class:`baseline_cost.BaselineCostModel` with three families
of upper bounds. Each baseline exposes a single ``cost(poly)`` method that
returns an integer op count compatible with ``monomial_build_cost`` /
``repeated_squaring_cost`` (one polynomial-multiplication = 1 op, one
polynomial-addition = 1 op, scalar multiplication and ``x``/``y`` themselves
are free).

The current default setup is 2-variable (``("x", "y")``) over a small prime,
so the analyses below assume two outer variables, but every routine is
``len(variables)``-agnostic.

- :class:`BivariateHornerBaseline` - classical multivariate Horner with
  gap coalescing via repeated-squaring. Tries each variable as the outer one
  and returns the best.
- :class:`CSEBaseline` - DAG-aware cost: each unique ``x^k`` / ``y^k`` is
  built once via a shared power chain, monomials combine those shared powers,
  and the ``support_size - 1`` final additions are amortised across the term
  set.
- :class:`TopDownSearchBaseline` - memoised branch-and-bound over
  power-pivot splits ``poly = lower + x_i^k * upper`` for ``k = 1..max``.
  Generalises the one-step Horner pivot used by ``BaselineCostModel`` and
  terminates because every split strictly shrinks ``support_size``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .baseline_cost import BaselineCostModel
from .cost_model import monomial_build_cost, repeated_squaring_cost
from .polynomial import SparsePolynomial


def _is_atomic(poly: SparsePolynomial) -> bool:
    return (
        poly.is_zero
        or poly.is_constant
        or poly.is_variable()
        or poly.is_monomial
    )


def _atomic_cost(poly: SparsePolynomial) -> int:
    if poly.is_zero or poly.is_constant or poly.is_variable():
        return 0
    return monomial_build_cost(poly)


def _sparse_direct_cost(poly: SparsePolynomial) -> int:
    """Sum the cost of every monomial plus ``support_size - 1`` additions."""
    term_total = 0
    for coeff, exponent in poly.terms:
        mono = SparsePolynomial.from_monomial(coeff, exponent, poly.p, poly.variables)
        term_total += monomial_build_cost(mono)
    return term_total + max(0, poly.support_size - 1)


@dataclass
class BivariateHornerBaseline:
    """Gap-aware nested Horner; tries every variable as outer."""

    _cache: dict[str, int] = field(default_factory=dict)

    def cost(self, poly: SparsePolynomial) -> int:
        key = poly.to_key()
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        if _is_atomic(poly):
            value = _atomic_cost(poly)
            self._cache[key] = value
            return value
        self._cache[key] = _sparse_direct_cost(poly)  # provisional, kills cycles
        best = _sparse_direct_cost(poly)
        for outer_index, max_deg in enumerate(poly.max_degrees):
            if max_deg == 0:
                continue
            candidate = self._gap_horner(poly, outer_index)
            if candidate < best:
                best = candidate
        self._cache[key] = best
        return best

    def _gap_horner(self, poly: SparsePolynomial, outer_index: int) -> int:
        groups: dict[int, list[tuple[int, tuple[int, ...]]]] = {}
        for coeff, exponent in poly.terms:
            outer_e = exponent[outer_index]
            inner_e = tuple(
                value for i, value in enumerate(exponent) if i != outer_index
            )
            groups.setdefault(outer_e, []).append((coeff, inner_e))

        if len(groups) <= 1:
            # Splitting on this variable does not separate terms; bail out
            # to sparse direct so we never inflate the bound.
            return _sparse_direct_cost(poly)

        inner_vars = tuple(
            name for i, name in enumerate(poly.variables) if i != outer_index
        )
        sorted_exps = sorted(groups.keys())

        coeff_polys: dict[int, SparsePolynomial] = {}
        coeff_cost = 0
        for outer_e in sorted_exps:
            coeff_poly = SparsePolynomial(
                poly.p, inner_vars, tuple(groups[outer_e])
            )
            coeff_polys[outer_e] = coeff_poly
            coeff_cost += self.cost(coeff_poly)

        # Leading coefficient sits in the accumulator at the start of the
        # Horner sweep. If it is a scalar, the first `acc * x^gap` step
        # degenerates to a scalar-times-monomial which is free in this cost
        # model (only the rep_sq(gap) power mults are charged).
        leading_is_scalar = coeff_polys[sorted_exps[-1]].is_constant

        horner_cost = 0
        for step_idx, j in enumerate(range(len(sorted_exps) - 1, 0, -1)):
            gap = sorted_exps[j] - sorted_exps[j - 1]
            horner_cost += repeated_squaring_cost(gap)
            if not (step_idx == 0 and leading_is_scalar):
                horner_cost += 1  # acc * x^gap
            horner_cost += 1  # acc + next coefficient
        # If the lowest active exponent is > 0, multiply the running
        # accumulator by x^{lowest} after the sweep finishes.
        if sorted_exps[0] > 0:
            horner_cost += repeated_squaring_cost(sorted_exps[0]) + 1
        return coeff_cost + horner_cost


@dataclass
class CSEBaseline:
    """Per-variable shared power chains + per-term combine + final adds."""

    _cache: dict[str, int] = field(default_factory=dict)

    def cost(self, poly: SparsePolynomial) -> int:
        key = poly.to_key()
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        if _is_atomic(poly):
            value = _atomic_cost(poly)
        else:
            value = self._compute(poly)
        self._cache[key] = value
        return value

    def _compute(self, poly: SparsePolynomial) -> int:
        # Powers >= 2 per variable are the only ones that cost anything;
        # x^0 is the identity, x^1 is the variable itself.
        per_var_powers: list[set[int]] = [set() for _ in poly.variables]
        for _, exponent in poly.terms:
            for i, value in enumerate(exponent):
                if value >= 2:
                    per_var_powers[i].add(value)

        power_cost = sum(self._power_chain_cost(p) for p in per_var_powers)

        combine_cost = 0
        for _, exponent in poly.terms:
            nonzero_vars = sum(1 for value in exponent if value > 0)
            combine_cost += max(0, nonzero_vars - 1)

        add_cost = max(0, poly.support_size - 1)
        return power_cost + combine_cost + add_cost

    @staticmethod
    def _power_chain_cost(powers: set[int]) -> int:
        if not powers:
            return 0
        max_power = max(powers)
        # Incremental chain x^2, x^3, ..., x^max gives every power, costs
        # max - 1 mults total. Wins when the support fills the ladder.
        incremental = max_power - 1
        # Independent repeated-squaring per requested power. Wins when only
        # a few high powers are needed (e.g. {16} -> 4 vs 15).
        independent = sum(repeated_squaring_cost(p) for p in powers)
        return min(incremental, independent)


@dataclass
class TopDownSearchBaseline:
    """Memoised branch-and-bound over power-pivot splits."""

    max_branches_per_var: int = 3
    _cache: dict[str, int] = field(default_factory=dict)

    def cost(self, poly: SparsePolynomial) -> int:
        key = poly.to_key()
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        if _is_atomic(poly):
            value = _atomic_cost(poly)
            self._cache[key] = value
            return value
        # Provisional upper bound prevents recursive blow-up if a child ever
        # canonicalises back to the parent (defensive; the support-size
        # argument below normally rules this out).
        sparse_direct = _sparse_direct_cost(poly)
        self._cache[key] = sparse_direct
        best = sparse_direct

        for var_index, max_deg in enumerate(poly.max_degrees):
            if max_deg == 0:
                continue
            for pivot in range(1, min(max_deg, self.max_branches_per_var) + 1):
                lower_terms: list[tuple[int, tuple[int, ...]]] = []
                upper_terms: list[tuple[int, tuple[int, ...]]] = []
                for coeff, exponent in poly.terms:
                    if exponent[var_index] >= pivot:
                        reduced = list(exponent)
                        reduced[var_index] -= pivot
                        upper_terms.append((coeff, tuple(reduced)))
                    else:
                        lower_terms.append((coeff, exponent))
                if not upper_terms or not lower_terms:
                    continue
                # Both halves strictly shrink: |lower|+|upper| == |poly|
                # and both > 0, so each child has fewer terms than poly.
                lower = SparsePolynomial(poly.p, poly.variables, tuple(lower_terms))
                upper = SparsePolynomial(poly.p, poly.variables, tuple(upper_terms))
                # 1 for the outer add, then build x_i^pivot via repeated
                # squaring (free if pivot == 1) and one multiply against
                # the upper subexpression.
                pivot_power_cost = repeated_squaring_cost(pivot) + 1
                candidate = (
                    1
                    + pivot_power_cost
                    + self.cost(lower)
                    + self.cost(upper)
                )
                if candidate < best:
                    best = candidate

        self._cache[key] = best
        return best


@dataclass
class BaselineBundle:
    """Min upper-bound cost across the five baselines, with per-target memoisation.

    Used by PPO terminal-bonus shaping: a target whose discovered circuit cost
    falls below ``min_cost`` has beaten every closed-form / memoised baseline.
    """

    baseline_model: BaselineCostModel = field(default_factory=BaselineCostModel)
    mv_horner: BivariateHornerBaseline = field(default_factory=BivariateHornerBaseline)
    cse: CSEBaseline = field(default_factory=CSEBaseline)
    top_down_search: TopDownSearchBaseline = field(default_factory=TopDownSearchBaseline)
    _min_cache: dict[str, int] = field(default_factory=dict)

    def min_cost(self, poly: SparsePolynomial) -> int:
        key = poly.to_key()
        cached = self._min_cache.get(key)
        if cached is not None:
            return cached
        candidates = (
            self.baseline_model.sparse_direct_cost(poly),
            self.baseline_model.horner_upper_bound(poly),
            self.mv_horner.cost(poly),
            self.cse.cost(poly),
            self.top_down_search.cost(poly),
        )
        value = int(min(candidates))
        self._min_cache[key] = value
        return value
