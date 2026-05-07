"""Baseline direct-construction costs and base-case handling."""

from __future__ import annotations

from dataclasses import dataclass, field

from .cost_model import monomial_build_cost
from .polynomial import SparsePolynomial


@dataclass
class BaselineCostModel:
    exact_support_limit: int = 3
    _cache: dict[str, int] = field(default_factory=dict)

    def is_base_case(self, poly: SparsePolynomial) -> bool:
        return (
            poly.is_zero
            or poly.is_constant
            or poly.is_variable()
            or poly.is_monomial
        )

    def exact_base_cost(self, poly: SparsePolynomial) -> int:
        if poly.is_zero or poly.is_constant or poly.is_variable():
            return 0
        if poly.is_monomial:
            return monomial_build_cost(poly)
        raise ValueError("Requested exact base cost for a non-base polynomial")

    def sparse_direct_cost(self, poly: SparsePolynomial) -> int:
        if self.is_base_case(poly):
            return self.exact_base_cost(poly)
        term_cost = 0
        for coeff, exponent in poly.terms:
            monomial = SparsePolynomial.from_monomial(coeff, exponent, poly.p, poly.variables)
            term_cost += monomial_build_cost(monomial)
        addition_cost = max(0, poly.support_size - 1)
        return term_cost + addition_cost

    def horner_upper_bound(self, poly: SparsePolynomial) -> int:
        if self.is_base_case(poly):
            return self.exact_base_cost(poly)
        key = f"horner::{poly.to_key()}"
        if key in self._cache:
            return self._cache[key]

        best = self.sparse_direct_cost(poly)
        for index, max_degree in enumerate(poly.max_degrees):
            if max_degree == 0:
                continue
            remainder, quotient = poly.split_by_variable(index)
            if quotient.is_zero:
                continue
            multiply_cost = 0 if quotient.is_constant else 1
            candidate = (
                1
                + multiply_cost
                + self.direct_construction_cost(remainder)
                + self.direct_construction_cost(quotient)
            )
            best = min(best, candidate)
        self._cache[key] = best
        return best

    def direct_construction_cost(self, poly: SparsePolynomial) -> int:
        key = poly.to_key()
        if key in self._cache:
            return self._cache[key]
        if self.is_base_case(poly):
            value = self.exact_base_cost(poly)
        else:
            value = min(self.sparse_direct_cost(poly), self.horner_upper_bound(poly))
        self._cache[key] = value
        return value
