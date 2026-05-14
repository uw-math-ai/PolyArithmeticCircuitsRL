"""Static problem specification for one symbolic synthesis task."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from lgs.poly.fast_poly import Polynomial, PolynomialDomainError, is_prime
from lgs.poly.poly_utils import assert_degree_cap


@dataclass(frozen=True)
class ProblemInstance:
    target: Polynomial
    variables: tuple[str, ...]
    field_p: int
    degree_cap: int
    op_budget: int
    family_name: str = "manual"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        target = self.target
        if not isinstance(target, Polynomial):
            raise PolynomialDomainError("target must be a Polynomial")
        if type(self.field_p) is not int or not is_prime(self.field_p):
            raise PolynomialDomainError(f"field_p must be prime, got {self.field_p!r}")
        if target.p != self.field_p:
            raise PolynomialDomainError(
                f"target field F_{target.p} does not match F_{self.field_p}"
            )

        variables = tuple(self.variables)
        if len(variables) != target.n_vars:
            raise PolynomialDomainError(
                f"{len(variables)} variables do not match target n_vars {target.n_vars}"
            )
        if len(set(variables)) != len(variables):
            raise ValueError("variable names must be unique")
        for name in variables:
            if not isinstance(name, str) or not name:
                raise ValueError("variable names must be non-empty strings")
        object.__setattr__(self, "variables", variables)

        if type(self.degree_cap) is not int or self.degree_cap < 0:
            raise ValueError("degree_cap must be a non-negative int")
        if target.degree_cap is None:
            target = target.with_degree_cap(self.degree_cap)
        elif target.degree_cap != self.degree_cap:
            raise PolynomialDomainError(
                f"target degree_cap {target.degree_cap} does not match {self.degree_cap}"
            )
        assert_degree_cap(target, self.degree_cap)
        object.__setattr__(self, "target", target)

        if type(self.op_budget) is not int or self.op_budget < 0:
            raise ValueError("op_budget must be a non-negative int")
        if not isinstance(self.family_name, str) or not self.family_name:
            raise ValueError("family_name must be a non-empty string")
        if not isinstance(self.metadata, Mapping):
            raise ValueError("metadata must be a mapping")
        object.__setattr__(self, "metadata", dict(self.metadata))

    def base_polynomials(self) -> tuple[Polynomial, ...]:
        """Return base nodes in prompt order: ``x_0, ..., x_{n-1}, 1``."""

        n_vars = len(self.variables)
        return (
            *(
                Polynomial.variable(index, n_vars, self.degree_cap, self.field_p)
                for index in range(n_vars)
            ),
            Polynomial.one(n_vars, self.degree_cap, self.field_p),
        )
