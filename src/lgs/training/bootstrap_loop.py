"""Small deterministic bootstrap loop for candidate-ranker training."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from lgs.data.curriculum import FixedCurriculum
from lgs.eval.evaluate_search import SearchEvalMetrics, evaluate_beam_search
from lgs.models.candidate_ranker import CandidateRanker
from lgs.models.feature_encoder import CandidateFeatureEncoder
from lgs.search.beam_search import beam_search
from lgs.training.preference_dataset import PreferenceExample, extract_preferences
from lgs.training.train_ranker import train_ranker_on_preferences


@dataclass
class BootstrapConfig:
    num_rounds: int = 3
    beam_width: int = 16
    candidate_k: int = 64
    tier2_m: int = 128
    epochs_per_round: int = 50
    lr: float = 1e-3
    margin: float = 1.0
    batch_size: int = 32
    seed: int = 0
    lambda_values: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0)
    preference_delta_start: float = 5.0
    preference_delta_end: float = 1.0

    def __post_init__(self) -> None:
        _require_non_negative_int("num_rounds", self.num_rounds)
        _require_positive_int("beam_width", self.beam_width)
        _require_non_negative_int("candidate_k", self.candidate_k)
        _require_non_negative_int("tier2_m", self.tier2_m)
        _require_non_negative_int("epochs_per_round", self.epochs_per_round)
        _require_positive_int("batch_size", self.batch_size)
        _require_non_negative_int("seed", self.seed)
        _require_non_negative_float("lr", self.lr)
        _require_non_negative_float("margin", self.margin)
        _require_non_negative_float("preference_delta_start", self.preference_delta_start)
        _require_non_negative_float("preference_delta_end", self.preference_delta_end)
        if not self.lambda_values:
            raise ValueError("lambda_values must be non-empty")
        previous = -1.0
        normalized: list[float] = []
        for value in self.lambda_values:
            _require_non_negative_float("lambda_values", value)
            as_float = float(value)
            if as_float < previous:
                raise ValueError("lambda_values must be sorted ascending")
            normalized.append(as_float)
            previous = as_float
        self.lambda_values = tuple(normalized)


@dataclass
class RoundMetrics:
    round_idx: int
    lambda_model: float
    num_preferences_added: int
    total_preferences: int
    train_loss_final: float | None
    train_accuracy_final: float | None
    heuristic_val_success_rate: float
    guided_val_success_rate: float
    heuristic_val_avg_best_ops: float | None
    guided_val_avg_best_ops: float | None
    heuristic_val_avg_expansions: float
    guided_val_avg_expansions: float


@dataclass
class BootstrapResult:
    ranker: CandidateRanker
    encoder: CandidateFeatureEncoder
    metrics: list[RoundMetrics]
    final_lambda_model: float


def preference_delta_for_round(config: BootstrapConfig, round_idx: int) -> float:
    if type(round_idx) is not int or round_idx < 0:
        raise ValueError("round_idx must be a non-negative int")
    if config.num_rounds <= 1:
        return float(config.preference_delta_end)
    clamped_round = min(round_idx, config.num_rounds - 1)
    fraction = clamped_round / (config.num_rounds - 1)
    return (
        float(config.preference_delta_start)
        + fraction * (float(config.preference_delta_end) - float(config.preference_delta_start))
    )


def should_promote_lambda(
    heuristic_metrics: SearchEvalMetrics,
    guided_metrics: SearchEvalMetrics,
) -> bool:
    if guided_metrics.success_rate < heuristic_metrics.success_rate:
        return False
    if guided_metrics.success_rate > heuristic_metrics.success_rate:
        return True
    return guided_metrics.avg_expansions <= heuristic_metrics.avg_expansions


def next_lambda(current: float, lambda_values: tuple[float, ...]) -> float:
    _require_non_negative_float("current", current)
    if not lambda_values:
        raise ValueError("lambda_values must be non-empty")
    for value in sorted(float(item) for item in lambda_values):
        if value > float(current):
            return value
    return float(current)


def run_bootstrap_training(
    curriculum: FixedCurriculum,
    config: BootstrapConfig,
) -> BootstrapResult:
    if not isinstance(curriculum, FixedCurriculum):
        raise TypeError("curriculum must be a FixedCurriculum")
    if not isinstance(config, BootstrapConfig):
        raise TypeError("config must be a BootstrapConfig")

    torch.manual_seed(config.seed)
    encoder = CandidateFeatureEncoder()
    ranker = CandidateRanker(input_dim=len(encoder.feature_names))
    preference_buffer: list[PreferenceExample] = []
    round_metrics: list[RoundMetrics] = []
    lambda_model = 0.0

    for round_idx in range(config.num_rounds):
        delta = preference_delta_for_round(config, round_idx)
        new_preferences = _collect_preferences(
            curriculum=curriculum,
            config=config,
            ranker=ranker,
            encoder=encoder,
            lambda_model=lambda_model,
            round_idx=round_idx,
            delta=delta,
        )
        preference_buffer.extend(new_preferences)

        train_loss_final: float | None = None
        train_accuracy_final: float | None = None
        if preference_buffer:
            train_history = train_ranker_on_preferences(
                ranker,
                encoder,
                preference_buffer,
                epochs=config.epochs_per_round,
                lr=config.lr,
                margin=config.margin,
                batch_size=config.batch_size,
                seed=config.seed + round_idx,
            )
            train_loss_final = train_history["loss"][-1]
            train_accuracy_final = train_history["accuracy"][-1]

        validation_instances = curriculum.validation_set()
        heuristic_metrics = evaluate_beam_search(
            validation_instances,
            beam_width=config.beam_width,
            candidate_k=config.candidate_k,
            tier2_m=config.tier2_m,
        )
        current_guided_metrics = evaluate_beam_search(
            validation_instances,
            ranker=ranker,
            encoder=encoder,
            lambda_model=lambda_model,
            beam_width=config.beam_width,
            candidate_k=config.candidate_k,
            tier2_m=config.tier2_m,
        )

        candidate_lambda = next_lambda(lambda_model, config.lambda_values)
        guided_metrics = current_guided_metrics
        if preference_buffer and candidate_lambda != lambda_model:
            candidate_guided_metrics = evaluate_beam_search(
                validation_instances,
                ranker=ranker,
                encoder=encoder,
                lambda_model=candidate_lambda,
                beam_width=config.beam_width,
                candidate_k=config.candidate_k,
                tier2_m=config.tier2_m,
            )
            if should_promote_lambda(heuristic_metrics, candidate_guided_metrics):
                lambda_model = candidate_lambda
                guided_metrics = candidate_guided_metrics

        round_metrics.append(
            RoundMetrics(
                round_idx=round_idx,
                lambda_model=lambda_model,
                num_preferences_added=len(new_preferences),
                total_preferences=len(preference_buffer),
                train_loss_final=train_loss_final,
                train_accuracy_final=train_accuracy_final,
                heuristic_val_success_rate=heuristic_metrics.success_rate,
                guided_val_success_rate=guided_metrics.success_rate,
                heuristic_val_avg_best_ops=heuristic_metrics.avg_best_ops,
                guided_val_avg_best_ops=guided_metrics.avg_best_ops,
                heuristic_val_avg_expansions=heuristic_metrics.avg_expansions,
                guided_val_avg_expansions=guided_metrics.avg_expansions,
            )
        )

    return BootstrapResult(
        ranker=ranker,
        encoder=encoder,
        metrics=round_metrics,
        final_lambda_model=lambda_model,
    )


def _collect_preferences(
    *,
    curriculum: FixedCurriculum,
    config: BootstrapConfig,
    ranker: CandidateRanker,
    encoder: CandidateFeatureEncoder,
    lambda_model: float,
    round_idx: int,
    delta: float,
) -> list[PreferenceExample]:
    preferences: list[PreferenceExample] = []
    for instance in curriculum.sample_training_instances(round_idx):
        history = beam_search(
            instance,
            ranker=ranker,
            encoder=encoder,
            lambda_model=lambda_model,
            beam_width=config.beam_width,
            candidate_k=config.candidate_k,
            tier2_m=config.tier2_m,
        )
        preferences.extend(extract_preferences(history, delta=delta))
    return preferences


def _require_positive_int(name: str, value: int) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive int")


def _require_non_negative_int(name: str, value: int) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative int")


def _require_non_negative_float(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    if float(value) < 0.0:
        raise ValueError(f"{name} must be non-negative")
