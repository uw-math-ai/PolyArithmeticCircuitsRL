"""Candidate transition data container."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite

from lgs.env.action import Action
from lgs.poly.fast_poly import Polynomial, PolynomialDomainError


@dataclass
class Candidate:
    action: Action
    result_poly: Polynomial
    source_tags: set[str] = field(default_factory=set)
    features: dict[str, float] = field(default_factory=dict)
    tier1_score: float = 0.0
    tier2_score: float = 0.0
    heuristic_score: float = 0.0
    model_score: float = 0.0
    total_score: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.action, Action):
            raise ValueError("candidate action must be an Action")
        if not isinstance(self.result_poly, Polynomial):
            raise PolynomialDomainError("candidate result_poly must be a Polynomial")

        tags = set(self.source_tags)
        for tag in tags:
            if not isinstance(tag, str) or not tag:
                raise ValueError("candidate source tags must be non-empty strings")
        self.source_tags = tags

        features: dict[str, float] = {}
        for name, value in dict(self.features).items():
            if not isinstance(name, str) or not name:
                raise ValueError("feature names must be non-empty strings")
            features[name] = _validate_float(value, f"feature {name!r}")
        self.features = features

        for score_name in (
            "tier1_score",
            "tier2_score",
            "heuristic_score",
            "model_score",
            "total_score",
        ):
            setattr(self, score_name, _validate_float(getattr(self, score_name), score_name))


def _validate_float(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    numeric = float(value)
    if not isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    return numeric
