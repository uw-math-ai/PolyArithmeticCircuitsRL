"""Sparse finite-field polynomial backend.

The implementation intentionally uses Python integers rather than fixed-width
integer arrays. That keeps arithmetic exact and removes overflow as a runtime
failure mode for this backend. The explicit i64 bound helper is provided for
future fixed-width implementations and tests the bound from the design doc.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Hashable, Iterable, Mapping, TypeAlias

Exponent: TypeAlias = tuple[int, ...]
TermItems: TypeAlias = Iterable[tuple[Exponent, int]]


class PolynomialError(ValueError):
    """Base class for polynomial backend errors."""


class PolynomialDomainError(PolynomialError):
    """Raised when polynomial domains or coefficient layouts do not match."""


class PolynomialOverflowError(OverflowError):
    """Raised when a fixed-width accumulation bound would overflow."""


class PolynomialDegreeError(PolynomialError):
    """Raised when a polynomial exceeds an explicit degree cap."""


I64_SIGNED_LIMIT = 2**63


@lru_cache(maxsize=None)
def is_prime(n: int) -> bool:
    """Return whether ``n`` is prime.

    The configured primes in this project are small enough that trial division
    is simple and reliable. This is validation code, not a primality oracle for
    huge cryptographic inputs.
    """

    if type(n) is not int:
        return False
    if n < 2:
        return False
    if n in (2, 3):
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False

    candidate = 5
    step = 2
    while candidate * candidate <= n:
        if n % candidate == 0:
            return False
        candidate += step
        step = 6 - step
    return True


def _validate_field_prime(field_p: int) -> None:
    if type(field_p) is not int:
        raise PolynomialDomainError("field_p must be an int")
    if not is_prime(field_p):
        raise PolynomialDomainError(f"field_p must be prime, got {field_p!r}")


def _validate_num_vars(num_vars: int) -> None:
    if type(num_vars) is not int:
        raise PolynomialDomainError("num_vars must be an int")
    if num_vars < 0:
        raise PolynomialDomainError("num_vars must be non-negative")


def _validate_degree_cap(degree_cap: int | None) -> None:
    if degree_cap is None:
        return
    if type(degree_cap) is not int:
        raise PolynomialDegreeError("degree_cap must be an int or None")
    if degree_cap < 0:
        raise PolynomialDegreeError("degree_cap must be non-negative")


def _validate_exponent(exponent: Exponent, num_vars: int) -> None:
    if not isinstance(exponent, tuple):
        raise PolynomialDomainError("monomial exponents must be tuples")
    if len(exponent) != num_vars:
        raise PolynomialDomainError(
            f"monomial exponent length {len(exponent)} does not match num_vars {num_vars}"
        )
    for power in exponent:
        if type(power) is not int:
            raise PolynomialDomainError("monomial exponents must be ints")
        if power < 0:
            raise PolynomialDomainError("monomial exponents must be non-negative")


def _validate_exponent_degree_cap(exponent: Exponent, degree_cap: int | None) -> None:
    if degree_cap is None:
        return
    if any(power > degree_cap for power in exponent):
        raise PolynomialDegreeError(
            f"monomial exponent {exponent} exceeds degree_cap {degree_cap}"
        )


def _resolve_domain_args(
    positional_num_vars: int | None,
    degree_cap: int | None,
    p: int | None,
    *,
    field_p: int | None,
    num_vars: int | None,
) -> tuple[int, int | None, int]:
    if positional_num_vars is not None and num_vars is not None:
        if positional_num_vars != num_vars:
            raise PolynomialDomainError(
                f"num_vars mismatch: {positional_num_vars} vs {num_vars}"
            )
    if p is not None and field_p is not None:
        if p != field_p:
            raise PolynomialDomainError(f"field mismatch: F_{p} vs F_{field_p}")

    resolved_num_vars = num_vars if num_vars is not None else positional_num_vars
    resolved_field_p = field_p if field_p is not None else p
    if resolved_num_vars is None:
        raise PolynomialDomainError("num_vars is required")
    if resolved_field_p is None:
        raise PolynomialDomainError("field prime p is required")
    return resolved_num_vars, degree_cap, resolved_field_p


def assert_i64_accumulation_safe(field_p: int, accumulation_count: int) -> None:
    """Raise if fixed-width signed i64 accumulation may overflow.

    This helper encodes the project invariant:

    ``accumulation_count * (p - 1)^2 < 2^63``.

    The sparse backend does not need this guard internally because Python ints
    are arbitrary precision, but fixed-width backends must use an equivalent
    check before multiplication.
    """

    _validate_field_prime(field_p)
    if type(accumulation_count) is not int:
        raise PolynomialOverflowError("accumulation_count must be an int")
    if accumulation_count < 0:
        raise PolynomialOverflowError("accumulation_count must be non-negative")

    bound = accumulation_count * (field_p - 1) * (field_p - 1)
    if bound >= I64_SIGNED_LIMIT:
        raise PolynomialOverflowError(
            "fixed-width i64 accumulation would overflow: "
            f"{accumulation_count} * ({field_p} - 1)^2 >= 2^63"
        )


class Polynomial:
    """Sparse polynomial over a prime finite field.

    Terms are stored as canonical ``((exponent_tuple, coefficient), ...)`` with
    coefficients in ``0..field_p-1`` and zero coefficients removed.
    """

    __slots__ = ("field_p", "num_vars", "degree_cap", "_terms")

    field_p: int
    num_vars: int
    degree_cap: int | None
    _terms: tuple[tuple[Exponent, int], ...]

    def __init__(
        self,
        terms: Mapping[Exponent, int] | TermItems | None = None,
        *,
        field_p: int,
        num_vars: int,
        degree_cap: int | None = None,
    ) -> None:
        _validate_field_prime(field_p)
        _validate_num_vars(num_vars)
        _validate_degree_cap(degree_cap)

        normalized: dict[Exponent, int] = {}
        if terms is not None:
            items = terms.items() if isinstance(terms, Mapping) else terms
            for exponent, coefficient in items:
                _validate_exponent(exponent, num_vars)
                _validate_exponent_degree_cap(exponent, degree_cap)
                if type(coefficient) is not int:
                    raise PolynomialDomainError("coefficients must be ints")
                value = coefficient % field_p
                if value == 0:
                    continue
                updated = (normalized.get(exponent, 0) + value) % field_p
                if updated:
                    normalized[exponent] = updated
                else:
                    normalized.pop(exponent, None)

        self.field_p = field_p
        self.num_vars = num_vars
        self.degree_cap = degree_cap
        self._terms = tuple(sorted(normalized.items()))

    @classmethod
    def zero(
        cls,
        n_vars: int | None = None,
        degree_cap: int | None = None,
        p: int | None = None,
        *,
        field_p: int | None = None,
        num_vars: int | None = None,
    ) -> Polynomial:
        resolved_num_vars, resolved_degree_cap, resolved_field_p = _resolve_domain_args(
            n_vars,
            degree_cap,
            p,
            field_p=field_p,
            num_vars=num_vars,
        )
        return cls(
            None,
            field_p=resolved_field_p,
            num_vars=resolved_num_vars,
            degree_cap=resolved_degree_cap,
        )

    @classmethod
    def one(
        cls,
        n_vars: int | None = None,
        degree_cap: int | None = None,
        p: int | None = None,
        *,
        field_p: int | None = None,
        num_vars: int | None = None,
    ) -> Polynomial:
        return cls.constant(
            1,
            n_vars,
            degree_cap,
            p,
            field_p=field_p,
            num_vars=num_vars,
        )

    @classmethod
    def constant(
        cls,
        value: int,
        n_vars: int | None = None,
        degree_cap: int | None = None,
        p: int | None = None,
        *,
        field_p: int | None = None,
        num_vars: int | None = None,
    ) -> Polynomial:
        resolved_num_vars, resolved_degree_cap, resolved_field_p = _resolve_domain_args(
            n_vars,
            degree_cap,
            p,
            field_p=field_p,
            num_vars=num_vars,
        )
        if type(value) is not int:
            raise PolynomialDomainError("constant value must be an int")
        return cls(
            {(0,) * resolved_num_vars: value},
            field_p=resolved_field_p,
            num_vars=resolved_num_vars,
            degree_cap=resolved_degree_cap,
        )

    @classmethod
    def monomial(
        cls,
        first: int | Exponent,
        second: int | Exponent,
        n_vars: int | None = None,
        degree_cap: int | None = None,
        p: int | None = None,
        *,
        field_p: int | None = None,
        num_vars: int | None = None,
    ) -> Polynomial:
        if isinstance(first, tuple):
            exponent = first
            coefficient = second
        else:
            coefficient = first
            exponent = second
        if not isinstance(exponent, tuple):
            raise PolynomialDomainError("monomial exponent must be a tuple")
        if type(coefficient) is not int:
            raise PolynomialDomainError("monomial coefficient must be an int")
        resolved_num_vars, resolved_degree_cap, resolved_field_p = _resolve_domain_args(
            n_vars,
            degree_cap,
            p,
            field_p=field_p,
            num_vars=num_vars,
        )
        return cls(
            {exponent: coefficient},
            field_p=resolved_field_p,
            num_vars=resolved_num_vars,
            degree_cap=resolved_degree_cap,
        )

    @classmethod
    def variable(
        cls,
        index: int,
        n_vars: int | None = None,
        degree_cap: int | None = None,
        p: int | None = None,
        *,
        field_p: int | None = None,
        num_vars: int | None = None,
    ) -> Polynomial:
        resolved_num_vars, resolved_degree_cap, resolved_field_p = _resolve_domain_args(
            n_vars,
            degree_cap,
            p,
            field_p=field_p,
            num_vars=num_vars,
        )
        if type(index) is not int:
            raise PolynomialDomainError("variable index must be an int")
        if index < 0 or index >= resolved_num_vars:
            raise PolynomialDomainError(
                f"variable index {index} out of range for {resolved_num_vars} variables"
            )
        exponent = [0] * resolved_num_vars
        exponent[index] = 1
        return cls.monomial(
            tuple(exponent),
            1,
            resolved_num_vars,
            resolved_degree_cap,
            resolved_field_p,
        )

    @property
    def terms(self) -> dict[Exponent, int]:
        return dict(self._terms)

    def items(self) -> tuple[tuple[Exponent, int], ...]:
        return self._terms

    def is_zero(self) -> bool:
        return not self._terms

    @property
    def p(self) -> int:
        return self.field_p

    @property
    def n_vars(self) -> int:
        return self.num_vars

    def coefficient(self, exponent: Exponent) -> int:
        _validate_exponent(exponent, self.num_vars)
        return self.terms.get(exponent, 0)

    def degree(self) -> int:
        if self.is_zero():
            return -1
        return max(sum(exponent) for exponent, _ in self._terms)

    def support(self) -> set[Exponent]:
        return {exponent for exponent, _ in self._terms}

    def copy(self) -> Polynomial:
        return Polynomial(
            self._terms,
            field_p=self.field_p,
            num_vars=self.num_vars,
            degree_cap=self.degree_cap,
        )

    def key(self) -> Hashable:
        return (self.field_p, self.num_vars, self.degree_cap, self._terms)

    def with_degree_cap(self, degree_cap: int) -> Polynomial:
        _validate_degree_cap(degree_cap)
        if self.degree_cap == degree_cap:
            return self
        if self.degree_cap is not None:
            raise PolynomialDomainError(
                f"cannot retag degree_cap {self.degree_cap} as {degree_cap}"
            )
        return Polynomial(
            self._terms,
            field_p=self.field_p,
            num_vars=self.num_vars,
            degree_cap=degree_cap,
        )

    def _check_same_domain(self, other: Polynomial) -> None:
        if not isinstance(other, Polynomial):
            raise PolynomialDomainError(f"expected Polynomial, got {type(other).__name__}")
        if self.field_p != other.field_p:
            raise PolynomialDomainError(
                f"field mismatch: F_{self.field_p} vs F_{other.field_p}"
            )
        if self.num_vars != other.num_vars:
            raise PolynomialDomainError(
                f"variable-count mismatch: {self.num_vars} vs {other.num_vars}"
            )
        if self.degree_cap != other.degree_cap:
            raise PolynomialDomainError(
                f"degree-cap mismatch: {self.degree_cap} vs {other.degree_cap}"
            )

    def __bool__(self) -> bool:
        return not self.is_zero()

    def __hash__(self) -> int:
        return hash(self.key())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Polynomial):
            return NotImplemented
        self._check_same_domain(other)
        return self._terms == other._terms

    def __neg__(self) -> Polynomial:
        return Polynomial(
            ((exponent, -coefficient) for exponent, coefficient in self._terms),
            field_p=self.field_p,
            num_vars=self.num_vars,
            degree_cap=self.degree_cap,
        )

    def __add__(self, other: Polynomial) -> Polynomial:
        self._check_same_domain(other)
        terms = self.terms
        for exponent, coefficient in other._terms:
            updated = (terms.get(exponent, 0) + coefficient) % self.field_p
            if updated:
                terms[exponent] = updated
            else:
                terms.pop(exponent, None)
        return Polynomial(
            terms,
            field_p=self.field_p,
            num_vars=self.num_vars,
            degree_cap=self.degree_cap,
        )

    def __sub__(self, other: Polynomial) -> Polynomial:
        self._check_same_domain(other)
        return self + (-other)

    def __mul__(self, other: Polynomial) -> Polynomial:
        self._check_same_domain(other)
        if self.is_zero() or other.is_zero():
            return Polynomial.zero(
                field_p=self.field_p,
                num_vars=self.num_vars,
                degree_cap=self.degree_cap,
            )

        terms: dict[Exponent, int] = {}
        for left_exp, left_coeff in self._terms:
            for right_exp, right_coeff in other._terms:
                exponent = tuple(
                    left_power + right_power
                    for left_power, right_power in zip(left_exp, right_exp)
                )
                _validate_exponent_degree_cap(exponent, self.degree_cap)
                updated = (
                    terms.get(exponent, 0) + left_coeff * right_coeff
                ) % self.field_p
                if updated:
                    terms[exponent] = updated
                else:
                    terms.pop(exponent, None)
        return Polynomial(
            terms,
            field_p=self.field_p,
            num_vars=self.num_vars,
            degree_cap=self.degree_cap,
        )

    def __pow__(self, exponent: int) -> Polynomial:
        if type(exponent) is not int:
            raise PolynomialDomainError("power exponent must be an int")
        if exponent < 0:
            raise PolynomialDomainError("power exponent must be non-negative")

        result = Polynomial.one(
            field_p=self.field_p,
            num_vars=self.num_vars,
            degree_cap=self.degree_cap,
        )
        base = self
        power = exponent
        while power:
            if power & 1:
                result = result * base
            power >>= 1
            if power:
                base = base * base
        return result

    def __repr__(self) -> str:
        return (
            f"Polynomial(terms={self.terms!r}, "
            f"field_p={self.field_p!r}, num_vars={self.num_vars!r}, "
            f"degree_cap={self.degree_cap!r})"
        )


FastPoly = Polynomial
