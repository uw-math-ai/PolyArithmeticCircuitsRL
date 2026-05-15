"""Offline pairwise training helpers for candidate rankers."""

from __future__ import annotations

from pathlib import Path
from collections.abc import Sequence

import torch

from lgs.models.candidate_ranker import CandidateRanker
from lgs.models.feature_encoder import CandidateFeatureEncoder
from lgs.training.preference_dataset import PreferenceExample


def pairwise_ranking_loss(
    better_scores: torch.Tensor,
    worse_scores: torch.Tensor,
    weights: torch.Tensor | None = None,
    margin: float = 1.0,
) -> torch.Tensor:
    if better_scores.shape != worse_scores.shape:
        raise ValueError("better_scores and worse_scores must have the same shape")
    if better_scores.ndim != 1:
        raise ValueError("scores must have shape [batch_size]")
    losses = torch.relu(float(margin) - better_scores + worse_scores)
    if weights is not None:
        if weights.shape != losses.shape:
            raise ValueError("weights must match score shape")
        losses = losses * weights
    return losses.mean()


def train_ranker_on_preferences(
    ranker: CandidateRanker,
    encoder: CandidateFeatureEncoder,
    preferences: Sequence[PreferenceExample],
    *,
    epochs: int = 100,
    lr: float = 1e-3,
    margin: float = 1.0,
    batch_size: int = 32,
    seed: int = 0,
) -> dict[str, list[float]]:
    if not preferences:
        raise ValueError("preferences must be non-empty")
    if type(epochs) is not int or epochs < 0:
        raise ValueError("epochs must be a non-negative int")
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive int")

    torch.manual_seed(seed)
    device = next(ranker.parameters()).device
    better_features, worse_features, weights = _encode_preferences(
        encoder,
        preferences,
        device=device,
    )

    anti_h_flags = [
        p.metadata.get("better_heuristic_score", 0.0)
        < p.metadata.get("worse_heuristic_score", 0.0)
        for p in preferences
    ]
    anti_h_count = sum(anti_h_flags)

    optimizer = torch.optim.Adam(ranker.parameters(), lr=lr)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    history: dict = {"loss": [], "accuracy": [], "anti_heuristic_accuracy": [], "anti_heuristic_count": anti_h_count}
    _append_metrics(history, ranker, better_features, worse_features, weights, margin, anti_h_flags, anti_h_count)

    n = better_features.shape[0]
    for _ in range(epochs):
        permutation = torch.randperm(n, generator=generator)
        ranker.train()
        for start in range(0, n, batch_size):
            indices = permutation[start : start + batch_size].to(device)
            better_scores = ranker(better_features[indices])
            worse_scores = ranker(worse_features[indices])
            loss = pairwise_ranking_loss(
                better_scores,
                worse_scores,
                weights[indices],
                margin=margin,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        _append_metrics(history, ranker, better_features, worse_features, weights, margin, anti_h_flags, anti_h_count)

    return history


def save_ranker(
    path: str | Path,
    ranker: CandidateRanker,
    encoder: CandidateFeatureEncoder,
) -> None:
    checkpoint = {
        "model_state_dict": ranker.state_dict(),
        "input_dim": ranker.input_dim,
        "feature_names": tuple(encoder.feature_names),
        "model_hparams": ranker.hyperparameters(),
    }
    torch.save(checkpoint, Path(path))


def load_ranker(path: str | Path) -> tuple[CandidateRanker, CandidateFeatureEncoder]:
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    feature_names = tuple(checkpoint["feature_names"])
    encoder = CandidateFeatureEncoder(feature_names=feature_names)
    ranker = CandidateRanker(**checkpoint["model_hparams"])
    ranker.load_state_dict(checkpoint["model_state_dict"])
    ranker.eval()
    return ranker, encoder


def _encode_preferences(
    encoder: CandidateFeatureEncoder,
    preferences: Sequence[PreferenceExample],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    better_rows = [
        encoder.encode(pref.instance, pref.state, pref.better)
        for pref in preferences
    ]
    worse_rows = [
        encoder.encode(pref.instance, pref.state, pref.worse)
        for pref in preferences
    ]
    weights = [pref.weight for pref in preferences]
    return (
        torch.tensor(better_rows, dtype=torch.float32, device=device),
        torch.tensor(worse_rows, dtype=torch.float32, device=device),
        torch.tensor(weights, dtype=torch.float32, device=device),
    )


def _append_metrics(
    history: dict,
    ranker: CandidateRanker,
    better_features: torch.Tensor,
    worse_features: torch.Tensor,
    weights: torch.Tensor,
    margin: float,
    anti_h_flags: list[bool] | None = None,
    anti_h_count: int = 0,
) -> None:
    ranker.eval()
    with torch.no_grad():
        better_scores = ranker(better_features)
        worse_scores = ranker(worse_features)
        loss = pairwise_ranking_loss(
            better_scores,
            worse_scores,
            weights,
            margin=margin,
        )
        accuracy = (better_scores > worse_scores).float().mean()
        if anti_h_count > 0 and anti_h_flags is not None:
            mask = torch.tensor(anti_h_flags, dtype=torch.bool, device=better_scores.device)
            anti_acc = float((better_scores[mask] > worse_scores[mask]).float().mean().item())
        else:
            anti_acc = float("nan")
    history["loss"].append(float(loss.item()))
    history["accuracy"].append(float(accuracy.item()))
    if "anti_heuristic_accuracy" in history:
        history["anti_heuristic_accuracy"].append(anti_acc)
