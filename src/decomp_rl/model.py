"""Policy/value scoring models for candidate splits."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

from .baseline_cost import BaselineCostModel
from .polynomial import SparsePolynomial
from .split_proposals import SplitAction

_DEFAULT_FEATURE_POLY = SparsePolynomial.zero(3, ("x",))
TARGET_FEATURE_DIM = len(_DEFAULT_FEATURE_POLY.to_feature_vector())
CANDIDATE_FEATURE_DIM = TARGET_FEATURE_DIM * 3


class PolicyValueModel(Protocol):
    def score_candidates(
        self,
        target: SparsePolynomial,
        candidates: list[SplitAction],
    ) -> tuple[list[float], float]:
        """Return policy priors over candidates and a normalized value estimate."""


def target_feature_vector(target: SparsePolynomial) -> list[float]:
    return list(target.to_feature_vector())


def candidate_feature_vector(target: SparsePolynomial, action: SplitAction) -> list[float]:
    return list(target.to_feature_vector() + action.g.to_feature_vector() + action.h.to_feature_vector())


def softmax(values: list[float]) -> list[float]:
    if not values:
        return []
    shift = max(values)
    weights = [math.exp(value - shift) for value in values]
    total = sum(weights)
    return [weight / total for weight in weights]


@dataclass
class HeuristicPolicyValueModel:
    baseline_model: BaselineCostModel

    def score_candidates(
        self,
        target: SparsePolynomial,
        candidates: list[SplitAction],
    ) -> tuple[list[float], float]:
        if not candidates:
            return [], 0.0
        target_cost = max(1, self.baseline_model.direct_construction_cost(target))
        scores = []
        for action in candidates:
            naive_cost = (
                1
                + self.baseline_model.direct_construction_cost(action.g)
                + self.baseline_model.direct_construction_cost(action.h)
            )
            source_bonus = 0.5 if action.source == "horner" else 0.0
            balance_penalty = 0.05 * abs(action.g.support_size - action.h.support_size)
            scores.append((target_cost - naive_cost) / target_cost + source_bonus - balance_penalty)
        priors = softmax(scores)
        value = max(scores)
        return priors, max(-1.0, min(1.0, value))


try:
    import torch
    import torch.nn as nn

    class TorchPolicyValueNetwork(nn.Module):
        """Handcrafted-feature MLP baseline for quick experimentation."""

        def __init__(
            self,
            input_dim: int = CANDIDATE_FEATURE_DIM,
            hidden_dim: int = 128,
            target_dim: int = TARGET_FEATURE_DIM,
            shared_layers: int = 3,
            value_hidden_dim: int | None = None,
            value_layers: int = 2,
            activation: str = "relu",
        ) -> None:
            super().__init__()
            activation_cls = nn.GELU if activation.lower() == "gelu" else nn.ReLU
            shared_modules: list[nn.Module] = []
            last_dim = input_dim
            for _ in range(max(1, shared_layers)):
                shared_modules.append(nn.Linear(last_dim, hidden_dim))
                shared_modules.append(activation_cls())
                last_dim = hidden_dim
            self.shared = nn.Sequential(*shared_modules)
            self.policy_head = nn.Linear(hidden_dim, 1)
            value_width = value_hidden_dim or hidden_dim
            value_modules: list[nn.Module] = []
            value_input_dim = target_dim
            for _ in range(max(1, value_layers - 1)):
                value_modules.append(nn.Linear(value_input_dim, value_width))
                value_modules.append(activation_cls())
                value_input_dim = value_width
            value_modules.append(nn.Linear(value_input_dim, 1))
            value_modules.append(nn.Tanh())
            self.value_head = nn.Sequential(*value_modules)

        def forward(self, candidate_features: torch.Tensor, target_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            shared = self.shared(candidate_features)
            logits = self.policy_head(shared).squeeze(-1)
            value = self.value_head(target_features).squeeze(-1)
            return logits, value


    class TorchPolicyValueWrapper:
        def __init__(self, network: TorchPolicyValueNetwork, device: str | None = None) -> None:
            self.network = network
            self.device = device or str(next(network.parameters()).device)
            self.network.eval()

        @staticmethod
        def _candidate_features(target: SparsePolynomial, action: SplitAction) -> list[float]:
            return candidate_feature_vector(target, action)

        def score_candidates(
            self,
            target: SparsePolynomial,
            candidates: list[SplitAction],
        ) -> tuple[list[float], float]:
            if not candidates:
                return [], 0.0
            with torch.no_grad():
                candidate_tensor = torch.tensor(
                    [self._candidate_features(target, action) for action in candidates],
                    dtype=torch.float32,
                    device=self.device,
                ).unsqueeze(0)
                target_tensor = torch.tensor(
                    [target_feature_vector(target)],
                    dtype=torch.float32,
                    device=self.device,
                )
                logits, value = self.network(candidate_tensor, target_tensor)
                priors = torch.softmax(logits.squeeze(0), dim=-1).detach().cpu().tolist()
                return priors, float(value.detach().cpu().item())


except ImportError:  # pragma: no cover - optional dependency
    torch = None
    nn = None
