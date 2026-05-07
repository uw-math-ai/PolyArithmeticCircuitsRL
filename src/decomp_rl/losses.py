"""Training losses for policy/value supervision."""

from __future__ import annotations

import math


def cross_entropy_from_probs(target_probs: list[float], predicted_probs: list[float]) -> float:
    epsilon = 1e-8
    return -sum(
        target * math.log(max(predicted, epsilon))
        for target, predicted in zip(target_probs, predicted_probs)
    )


def mean_squared_error(target: float, prediction: float) -> float:
    return (target - prediction) ** 2


try:
    import torch
    import torch.nn.functional as F

    def torch_policy_value_loss(
        logits,
        target_policy,
        predicted_value,
        target_value,
        value_weight: float = 1.0,
    ):
        policy_loss = -(target_policy * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
        value_loss = F.mse_loss(predicted_value, target_value)
        return policy_loss + value_weight * value_loss

except ImportError:  # pragma: no cover - optional dependency
    torch = None

