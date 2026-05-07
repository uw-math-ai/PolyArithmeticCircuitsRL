"""Sparse polynomial representation over a prime finite field."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Iterable

from .canonical import canonicalize_terms, modular_inverse, normalize_coeff


@dataclass(frozen=True)
class SparsePolynomial:
    """Immutable sparse polynomial over F_p with canonical term ordering."""

    p: int
    variables: tuple[str, ...]
    terms: tuple[tuple[int, tuple[int, ...]], ...]

    def __post_init__(self) -> None:
        if self.p <= 1:
            raise ValueError("Prime must be greater than 1")
        canonical_terms = canonicalize_terms(self.terms, self.p, len(self.variables))
        object.__setattr__(self, "terms", canonical_terms)

    @classmethod
    def zero(cls, prime: int, variables: tuple[str, ...]) -> "SparsePolynomial":
        return cls(prime, variables, ())

    @classmethod
    def one(cls, prime: int, variables: tuple[str, ...]) -> "SparsePolynomial":
        return cls.from_monomial(1, (0,) * len(variables), prime, variables)

    @classmethod
    def variable(
        cls,
        name: str,
        prime: int,
        variables: tuple[str, ...],
    ) -> "SparsePolynomial":
        try:
            index = variables.index(name)
        except ValueError as exc:
            raise ValueError(f"Unknown variable {name!r}") from exc
        exponent = [0] * len(variables)
        exponent[index] = 1
        return cls.from_monomial(1, tuple(exponent), prime, variables)

    @classmethod
    def from_monomial(
        cls,
        coeff: int,
        exponent: tuple[int, ...],
        prime: int,
        variables: tuple[str, ...],
    ) -> "SparsePolynomial":
        return cls(prime, variables, ((coeff, exponent),))

    @classmethod
    def from_terms(
        cls,
        terms: Iterable[tuple[int, tuple[int, ...]]],
        prime: int,
        variables: tuple[str, ...],
    ) -> "SparsePolynomial":
        return cls(prime, variables, tuple(terms))

    @classmethod
    def from_dict(
        cls,
        terms: dict[tuple[int, ...], int],
        prime: int,
        variables: tuple[str, ...],
    ) -> "SparsePolynomial":
        return cls(prime, variables, tuple((coeff, exp) for exp, coeff in terms.items()))

    @classmethod
    def from_sympy_expr(
        cls,
        expr,
        prime: int,
        variables: tuple[str, ...],
    ) -> "SparsePolynomial":
        try:
            from sympy import Poly, symbols
        except ImportError as exc:
            raise RuntimeError("SymPy is required to convert from SymPy expressions") from exc

        symbols_tuple = symbols(variables)
        poly = Poly(expr, *symbols_tuple, modulus=prime)
        terms = [(int(coeff) % prime, tuple(int(v) for v in exp)) for exp, coeff in poly.terms()]
        return cls(prime, variables, tuple(terms))

    def assert_compatible(self, other: "SparsePolynomial") -> None:
        if self.p != other.p or self.variables != other.variables:
            raise ValueError("Polynomials must share the same field and variables")

    def to_key(self) -> str:
        payload = ",".join(
            f"{coeff}:{'.'.join(str(v) for v in exponent)}"
            for coeff, exponent in self.terms
        )
        return f"p={self.p}|vars={','.join(self.variables)}|{payload}"

    def __hash__(self) -> int:
        return hash(self.to_key())

    @cached_property
    def term_dict(self) -> dict[tuple[int, ...], int]:
        return {exponent: coeff for coeff, exponent in self.terms}

    @cached_property
    def support_size(self) -> int:
        return len(self.terms)

    @cached_property
    def total_degree(self) -> int:
        if not self.terms:
            return 0
        return max(sum(exponent) for _, exponent in self.terms)

    @cached_property
    def max_degrees(self) -> tuple[int, ...]:
        if not self.terms:
            return (0,) * len(self.variables)
        return tuple(
            max(exponent[index] for _, exponent in self.terms)
            for index in range(len(self.variables))
        )

    @cached_property
    def homogeneous(self) -> bool:
        degrees = {sum(exponent) for _, exponent in self.terms}
        return len(degrees) <= 1

    @cached_property
    def monomial_gcd(self) -> tuple[int, ...]:
        if not self.terms:
            return (0,) * len(self.variables)
        gcd = list(self.terms[0][1])
        for _, exponent in self.terms[1:]:
            for index, value in enumerate(exponent):
                gcd[index] = min(gcd[index], value)
        return tuple(gcd)

    @cached_property
    def coefficient_histogram(self) -> dict[int, int]:
        histogram: dict[int, int] = {}
        for coeff, _ in self.terms:
            histogram[coeff] = histogram.get(coeff, 0) + 1
        return histogram

    def coeff(self, exponent: tuple[int, ...]) -> int:
        return self.term_dict.get(exponent, 0)

    @property
    def is_zero(self) -> bool:
        return not self.terms

    @property
    def is_constant(self) -> bool:
        return self.support_size <= 1 and all(
            value == 0 for _, exponent in self.terms for value in exponent
        )

    @property
    def is_monomial(self) -> bool:
        return self.support_size == 1

    def is_variable(self) -> bool:
        if self.support_size != 1:
            return False
        coeff, exponent = self.terms[0]
        return coeff == 1 and sum(exponent) == 1 and max(exponent, default=0) == 1

    def leading_term(self) -> tuple[int, tuple[int, ...]]:
        if self.is_zero:
            raise ValueError("Zero polynomial has no leading term")
        return self.terms[0]

    def make_monic(self) -> tuple["SparsePolynomial", int]:
        coeff, _ = self.leading_term()
        if coeff == 1:
            return self, 1
        inv = modular_inverse(coeff, self.p)
        return self.scale(inv), coeff

    def scale(self, scalar: int) -> "SparsePolynomial":
        scalar = normalize_coeff(scalar, self.p)
        if scalar == 0 or self.is_zero:
            return SparsePolynomial.zero(self.p, self.variables)
        return SparsePolynomial(
            self.p,
            self.variables,
            tuple(((coeff * scalar) % self.p, exponent) for coeff, exponent in self.terms),
        )

    def __neg__(self) -> "SparsePolynomial":
        return self.scale(-1)

    def __add__(self, other: "SparsePolynomial") -> "SparsePolynomial":
        self.assert_compatible(other)
        terms = list(self.terms) + list(other.terms)
        return SparsePolynomial(self.p, self.variables, tuple(terms))

    def __sub__(self, other: "SparsePolynomial") -> "SparsePolynomial":
        return self + (-other)

    def __mul__(self, other: "SparsePolynomial") -> "SparsePolynomial":
        self.assert_compatible(other)
        if self.is_zero or other.is_zero:
            return SparsePolynomial.zero(self.p, self.variables)
        products: list[tuple[int, tuple[int, ...]]] = []
        for left_coeff, left_exp in self.terms:
            for right_coeff, right_exp in other.terms:
                coeff = (left_coeff * right_coeff) % self.p
                exponent = tuple(a + b for a, b in zip(left_exp, right_exp))
                products.append((coeff, exponent))
        return SparsePolynomial(self.p, self.variables, tuple(products))

    def pow(self, exponent: int) -> "SparsePolynomial":
        if exponent < 0:
            raise ValueError("Polynomial powers must be non-negative")
        result = SparsePolynomial.one(self.p, self.variables)
        base = self
        current = exponent
        while current > 0:
            if current & 1:
                result = result * base
            current >>= 1
            if current:
                base = base * base
        return result

    def monomial_divides(self, exponent: tuple[int, ...]) -> bool:
        return all(
            all(term_exp[index] >= exponent[index] for index in range(len(self.variables)))
            for _, term_exp in self.terms
        )

    def divide_by_monomial(self, exponent: tuple[int, ...]) -> "SparsePolynomial":
        if not self.monomial_divides(exponent):
            raise ValueError("Monomial does not divide polynomial")
        return SparsePolynomial(
            self.p,
            self.variables,
            tuple(
                (coeff, tuple(value - exponent[index] for index, value in enumerate(term_exp)))
                for coeff, term_exp in self.terms
            ),
        )

    def split_by_variable(self, variable_index: int) -> tuple["SparsePolynomial", "SparsePolynomial"]:
        remainder_terms: list[tuple[int, tuple[int, ...]]] = []
        quotient_terms: list[tuple[int, tuple[int, ...]]] = []
        for coeff, exponent in self.terms:
            if exponent[variable_index] == 0:
                remainder_terms.append((coeff, exponent))
            else:
                reduced = list(exponent)
                reduced[variable_index] -= 1
                quotient_terms.append((coeff, tuple(reduced)))
        remainder = SparsePolynomial(self.p, self.variables, tuple(remainder_terms))
        quotient = SparsePolynomial(self.p, self.variables, tuple(quotient_terms))
        return remainder, quotient

    def variable_factor(self, variable_index: int) -> "SparsePolynomial":
        exponent = [0] * len(self.variables)
        exponent[variable_index] = 1
        return SparsePolynomial.from_monomial(1, tuple(exponent), self.p, self.variables)

    def to_sympy_expr(self):
        try:
            from sympy import symbols
        except ImportError as exc:
            raise RuntimeError("SymPy is required to convert to SymPy expressions") from exc

        symbols_tuple = symbols(self.variables)
        expr = 0
        for coeff, exponent in self.terms:
            term = coeff
            for symbol, power in zip(symbols_tuple, exponent):
                if power:
                    term *= symbol ** power
            expr += term
        return expr

    def to_feature_vector(self) -> tuple[float, ...]:
        coeff_entropy = 0.0
        if self.support_size:
            for count in self.coefficient_histogram.values():
                prob = count / self.support_size
                coeff_entropy -= prob * (0.0 if prob == 0 else __import__("math").log(prob))
        return (
            float(self.support_size),
            float(self.total_degree),
            float(sum(self.max_degrees)),
            float(sum(self.monomial_gcd)),
            1.0 if self.homogeneous else 0.0,
            coeff_entropy,
            float(len(self.variables)),
            float(self.p),
        )

    def __repr__(self) -> str:
        if self.is_zero:
            return "0"
        pieces = []
        for coeff, exponent in self.terms:
            monomial_bits = []
            for name, power in zip(self.variables, exponent):
                if power == 0:
                    continue
                monomial_bits.append(name if power == 1 else f"{name}^{power}")
            monomial = "*".join(monomial_bits)
            if monomial:
                pieces.append(f"{coeff}*{monomial}" if coeff != 1 else monomial)
            else:
                pieces.append(str(coeff))
        return " + ".join(pieces)
