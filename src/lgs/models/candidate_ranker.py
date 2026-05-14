"""Simple MLP candidate ranker."""

from __future__ import annotations

import torch


class CandidateRanker(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if type(input_dim) is not int or input_dim <= 0:
            raise ValueError("input_dim must be a positive int")
        if type(hidden_dim) is not int or hidden_dim <= 0:
            raise ValueError("hidden_dim must be a positive int")
        if type(num_layers) is not int or num_layers < 1:
            raise ValueError("num_layers must be a positive int")
        if isinstance(dropout, bool) or not isinstance(dropout, (int, float)):
            raise ValueError("dropout must be numeric")
        if float(dropout) < 0.0 or float(dropout) >= 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = float(dropout)

        layers: list[torch.nn.Module] = []
        in_dim = input_dim
        for _ in range(num_layers - 1):
            layers.append(torch.nn.Linear(in_dim, hidden_dim))
            layers.append(torch.nn.ReLU())
            if self.dropout:
                layers.append(torch.nn.Dropout(self.dropout))
            in_dim = hidden_dim
        layers.append(torch.nn.Linear(in_dim, 1))
        self.network = torch.nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2:
            raise ValueError("features must have shape [batch_size, input_dim]")
        if features.shape[1] != self.input_dim:
            raise ValueError(
                f"expected input_dim {self.input_dim}, got {features.shape[1]}"
            )
        return self.network(features).squeeze(-1)

    def hyperparameters(self) -> dict[str, int | float]:
        return {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
        }
