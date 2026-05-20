"""Supervised warm-start helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from random import Random
from pathlib import Path

from .family_generators import SupervisedExample
from .losses import cross_entropy_from_probs, mean_squared_error
from .model import (
    HeuristicPolicyValueModel,
    PolicyValueModel,
    candidate_feature_vector,
    target_feature_vector,
)
from .polynomial import SparsePolynomial
from .split_proposals import SplitAction

try:
    import torch
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - optional dependency
    torch = None
    F = None


@dataclass(frozen=True)
class SupervisedMetrics:
    average_policy_loss: float
    average_value_loss: float
    top1_accuracy: float
    example_count: int


@dataclass(frozen=True)
class PolicyValueTrainingExample:
    target: SparsePolynomial
    candidates: tuple[SplitAction, ...]
    policy_target: tuple[float, ...]
    value_target: float
    source: str


@dataclass(frozen=True)
class PolicyValueBatch:
    candidate_features: "torch.Tensor"
    target_features: "torch.Tensor"
    policy_targets: "torch.Tensor"
    value_targets: "torch.Tensor"
    candidate_mask: "torch.Tensor"
    batch_size: int


@dataclass(frozen=True)
class PackedTrainingExamples:
    candidate_features: "torch.Tensor"
    target_features: "torch.Tensor"
    policy_targets: "torch.Tensor"
    value_targets: "torch.Tensor"
    candidate_mask: "torch.Tensor"
    example_count: int


@dataclass(frozen=True)
class TorchTrainConfig:
    epochs: int = 5
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    value_loss_weight: float = 0.25
    seed: int = 0
    device: str = "auto"
    batch_size: int = 64
    auto_batch_size: bool = True
    max_auto_batch_size: int = 0
    target_gpu_memory_fraction: float = 0.9
    cache_dataset_on_device: bool = True
    use_amp: bool = True
    matmul_precision: str = "high"


@dataclass(frozen=True)
class TorchEpochMetrics:
    epoch: int
    average_loss: float
    average_policy_loss: float
    average_value_loss: float
    top1_accuracy: float


@dataclass(frozen=True)
class TorchTrainResult:
    network: object
    wrapper: PolicyValueModel
    history: tuple[TorchEpochMetrics, ...]
    final_metrics: SupervisedMetrics
    batch_size_stats: "BatchSizeStats"
    optimizer_state_dict: dict | None = None


@dataclass(frozen=True)
class BatchSizeStats:
    requested_batch_size: int
    resolved_batch_size: int
    auto_batch_size: bool
    device: str
    target_gpu_memory_fraction: float
    dataset_limited: bool = False
    peak_memory_mb: float | None = None
    total_memory_mb: float | None = None
    available_memory_mb: float | None = None
    peak_memory_fraction: float | None = None
    dataset_cache_device: str | None = None


def make_training_examples(examples: list[SupervisedExample]) -> list[PolicyValueTrainingExample]:
    training_examples = []
    for example in examples:
        candidates = tuple(example.candidates)
        preferred_index = next(
            index for index, candidate in enumerate(candidates) if candidate.key() == example.preferred_action.key()
        )
        target_policy = [0.0] * len(candidates)
        target_policy[preferred_index] = 1.0
        training_examples.append(
            PolicyValueTrainingExample(
                target=example.target,
                candidates=candidates,
                policy_target=tuple(target_policy),
                value_target=example.value_target,
                source=example.family,
            )
        )
    return training_examples


def evaluate_examples(
    examples: list[SupervisedExample],
    model: PolicyValueModel,
) -> SupervisedMetrics:
    return evaluate_training_examples(make_training_examples(examples), model)


def evaluate_training_examples(
    examples: list[PolicyValueTrainingExample],
    model: PolicyValueModel,
) -> SupervisedMetrics:
    if not examples:
        return SupervisedMetrics(0.0, 0.0, 0.0, 0)

    policy_losses = []
    value_losses = []
    correct = 0
    for example in examples:
        candidates = list(example.candidates)
        priors, value = model.score_candidates(example.target, candidates)
        if not priors:
            continue
        preferred_index = max(range(len(example.policy_target)), key=lambda idx: example.policy_target[idx])
        policy_losses.append(cross_entropy_from_probs(list(example.policy_target), priors))
        value_losses.append(mean_squared_error(example.value_target, value))
        if max(range(len(priors)), key=lambda idx: priors[idx]) == preferred_index:
            correct += 1

    total = max(1, len(policy_losses))
    return SupervisedMetrics(
        average_policy_loss=sum(policy_losses) / total,
        average_value_loss=sum(value_losses) / total,
        top1_accuracy=correct / total,
        example_count=total,
    )


def save_examples_jsonl(examples: list[SupervisedExample], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for example in examples:
            payload = {
                "family": example.family,
                "target": example.target.to_key(),
                "candidate_keys": [candidate.key() for candidate in example.candidates],
                "preferred_key": example.preferred_action.key(),
                "value_target": example.value_target,
                "total_cost_target": example.total_cost_target,
            }
            handle.write(json.dumps(payload) + "\n")


def build_default_model() -> HeuristicPolicyValueModel:
    from .baseline_cost import BaselineCostModel

    return HeuristicPolicyValueModel(BaselineCostModel())


def train_torch_model(
    examples: list[PolicyValueTrainingExample],
    config: TorchTrainConfig | None = None,
    network: object | None = None,
    optimizer_state_dict: dict | None = None,
) -> TorchTrainResult:
    if torch is None or F is None:
        raise RuntimeError("Torch is not installed. Install the training dependencies first.")

    from .model import TorchPolicyValueNetwork, TorchPolicyValueWrapper

    config = config or TorchTrainConfig()
    torch.manual_seed(config.seed)
    device = _resolve_torch_device(config.device)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision(config.matmul_precision)
    if device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    if network is None:
        network = TorchPolicyValueNetwork()
    network = network.to(device)
    optimizer = torch.optim.AdamW(
        network.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    if optimizer_state_dict is not None:
        try:
            optimizer.load_state_dict(optimizer_state_dict)
        except (ValueError, RuntimeError):
            pass  # Incompatible state (e.g. model architecture changed); start fresh
    scaler = torch.amp.GradScaler("cuda", enabled=device.startswith("cuda") and config.use_amp)
    amp_enabled = device.startswith("cuda") and config.use_amp
    amp_dtype = torch.bfloat16 if device.startswith("cuda") and torch.cuda.is_bf16_supported() else torch.float16
    batch_size_stats = resolve_training_batch_size(
        examples=examples,
        config=config,
        network=network,
        device=device,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
    )
    packed_examples = pack_training_examples(
        examples,
        pin_memory=device.startswith("cuda") and torch.cuda.is_available(),
    )
    if (
        config.cache_dataset_on_device
        and device.startswith("cuda")
        and torch.cuda.is_available()
        and packed_examples.example_count > 0
    ):
        packed_examples = move_packed_training_examples(packed_examples, device=device, non_blocking=True)
        batch_size_stats = BatchSizeStats(
            requested_batch_size=batch_size_stats.requested_batch_size,
            resolved_batch_size=batch_size_stats.resolved_batch_size,
            auto_batch_size=batch_size_stats.auto_batch_size,
            device=batch_size_stats.device,
            target_gpu_memory_fraction=batch_size_stats.target_gpu_memory_fraction,
            dataset_limited=batch_size_stats.dataset_limited,
            peak_memory_mb=batch_size_stats.peak_memory_mb,
            total_memory_mb=batch_size_stats.total_memory_mb,
            available_memory_mb=batch_size_stats.available_memory_mb,
            peak_memory_fraction=batch_size_stats.peak_memory_fraction,
            dataset_cache_device=device,
        )

    resolved_batch_size = batch_size_stats.resolved_batch_size
    history: list[TorchEpochMetrics] = []
    for epoch in range(config.epochs):
        while True:
            network.train()
            total_loss = 0.0
            total_policy_loss = 0.0
            total_value_loss = 0.0
            correct = 0
            try:
                for batch in iterate_training_batches(
                    packed_examples,
                    batch_size=resolved_batch_size,
                    device=device,
                    seed=config.seed + epoch,
                    shuffle=True,
                ):
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device_type=device.split(":")[0], dtype=amp_dtype, enabled=amp_enabled):
                        logits, predicted_value = network(batch.candidate_features, batch.target_features)
                        masked_logits = logits.masked_fill(~batch.candidate_mask, torch.finfo(logits.dtype).min)
                        log_probs = torch.log_softmax(masked_logits, dim=-1)
                        policy_loss = -(batch.policy_targets * log_probs).sum(dim=-1).mean()
                        value_loss = F.mse_loss(predicted_value, batch.value_targets)
                        loss = policy_loss + config.value_loss_weight * value_loss

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                    total_loss += float(loss.item()) * batch.batch_size
                    total_policy_loss += float(policy_loss.item()) * batch.batch_size
                    total_value_loss += float(value_loss.item()) * batch.batch_size
                    predictions = torch.argmax(masked_logits, dim=-1)
                    targets = torch.argmax(batch.policy_targets, dim=-1)
                    correct += int((predictions == targets).sum().item())
                break
            except RuntimeError as exc:
                if not _is_cuda_oom(exc) or resolved_batch_size <= 1:
                    raise
                optimizer.zero_grad(set_to_none=True)
                network.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                next_batch_size = max(1, resolved_batch_size // 2)
                print(
                    {
                        "stage": "train_batch_oom_backoff",
                        "epoch": epoch + 1,
                        "old_batch_size": resolved_batch_size,
                        "new_batch_size": next_batch_size,
                    },
                    flush=True,
                )
                resolved_batch_size = next_batch_size

        count = max(1, len(examples))
        history.append(
            TorchEpochMetrics(
                epoch=epoch + 1,
                average_loss=total_loss / count,
                average_policy_loss=total_policy_loss / count,
                average_value_loss=total_value_loss / count,
                top1_accuracy=correct / count,
            )
        )

    if resolved_batch_size != batch_size_stats.resolved_batch_size:
        batch_size_stats = BatchSizeStats(
            requested_batch_size=batch_size_stats.requested_batch_size,
            resolved_batch_size=resolved_batch_size,
            auto_batch_size=batch_size_stats.auto_batch_size,
            device=batch_size_stats.device,
            target_gpu_memory_fraction=batch_size_stats.target_gpu_memory_fraction,
            dataset_limited=resolved_batch_size >= len(examples),
            peak_memory_mb=batch_size_stats.peak_memory_mb,
            total_memory_mb=batch_size_stats.total_memory_mb,
            available_memory_mb=batch_size_stats.available_memory_mb,
            peak_memory_fraction=batch_size_stats.peak_memory_fraction,
            dataset_cache_device=batch_size_stats.dataset_cache_device,
        )

    wrapper = TorchPolicyValueWrapper(network, device=device)
    final_metrics = evaluate_training_examples(examples, wrapper)
    return TorchTrainResult(
        network=network,
        wrapper=wrapper,
        history=tuple(history),
        final_metrics=final_metrics,
        batch_size_stats=batch_size_stats,
        optimizer_state_dict=optimizer.state_dict(),
    )


def _resolve_torch_device(device: str) -> str:
    if device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _is_cuda_oom(exc: RuntimeError) -> bool:
    return "out of memory" in str(exc).lower()


def iterate_training_batches(
    examples: PackedTrainingExamples,
    batch_size: int,
    device: str,
    seed: int,
    shuffle: bool,
) -> "list[PolicyValueBatch]":
    if torch is None:
        raise RuntimeError("Torch is required for batch iteration")
    indices = list(range(examples.example_count))
    if shuffle:
        Random(seed).shuffle(indices)
    for start in range(0, len(indices), max(1, batch_size)):
        selected = indices[start : start + max(1, batch_size)]
        yield slice_training_batch(examples, selected, device)


def resolve_training_batch_size(
    examples: list[PolicyValueTrainingExample],
    config: TorchTrainConfig,
    network,
    device: str,
    amp_enabled: bool,
    amp_dtype,
) -> BatchSizeStats:
    requested = max(1, min(config.batch_size, max(1, len(examples))))
    if not examples:
        return BatchSizeStats(
            requested_batch_size=requested,
            resolved_batch_size=requested,
            auto_batch_size=config.auto_batch_size,
            device=device,
            target_gpu_memory_fraction=config.target_gpu_memory_fraction,
        )
    if not (config.auto_batch_size and device.startswith("cuda") and torch.cuda.is_available()):
        return BatchSizeStats(
            requested_batch_size=requested,
            resolved_batch_size=requested,
            auto_batch_size=config.auto_batch_size,
            device=device,
            target_gpu_memory_fraction=config.target_gpu_memory_fraction,
        )

    upper = len(examples)
    if config.max_auto_batch_size > 0:
        upper = min(upper, config.max_auto_batch_size)
    upper = max(1, upper)

    cuda_device = torch.device(device)
    device_index = cuda_device.index if cuda_device.index is not None else torch.cuda.current_device()
    total_memory = float(torch.cuda.get_device_properties(device_index).total_memory)
    free_memory, _ = torch.cuda.mem_get_info(device)
    current_allocated = float(torch.cuda.memory_allocated(device))
    available_budget = current_allocated + float(free_memory)
    target_peak = min(
        total_memory * max(0.1, min(config.target_gpu_memory_fraction, 0.99)),
        available_budget * max(0.1, min(config.target_gpu_memory_fraction, 0.99)),
    )

    success = 0
    best_peak = 0.0
    failure = upper + 1
    candidate = min(requested, upper)

    probe_success, peak = try_training_batch_size(
        examples=examples,
        batch_size=candidate,
        network=network,
        device=device,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
        value_loss_weight=config.value_loss_weight,
    )
    if probe_success:
        success = candidate
        best_peak = peak
        while success < upper and best_peak < target_peak:
            next_candidate = min(upper, max(success + 1, success * 2))
            probe_success, peak = try_training_batch_size(
                examples=examples,
                batch_size=next_candidate,
                network=network,
                device=device,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
                value_loss_weight=config.value_loss_weight,
            )
            if probe_success:
                success = next_candidate
                best_peak = peak
                if next_candidate == upper:
                    break
            else:
                failure = next_candidate
                break
    else:
        failure = candidate

    search_low = success
    search_high = min(upper, failure - 1) if failure <= upper else upper
    while search_low < search_high:
        mid = (search_low + search_high + 1) // 2
        probe_success, peak = try_training_batch_size(
            examples=examples,
            batch_size=mid,
            network=network,
            device=device,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            value_loss_weight=config.value_loss_weight,
        )
        if probe_success:
            search_low = mid
            best_peak = peak
        else:
            search_high = mid - 1

    resolved = max(1, search_low)
    peak_fraction = (best_peak / total_memory) if total_memory > 0 else None
    return BatchSizeStats(
        requested_batch_size=requested,
        resolved_batch_size=resolved,
        auto_batch_size=True,
        device=device,
        target_gpu_memory_fraction=config.target_gpu_memory_fraction,
        dataset_limited=resolved >= len(examples),
        peak_memory_mb=best_peak / (1024.0 * 1024.0) if best_peak else 0.0,
        total_memory_mb=total_memory / (1024.0 * 1024.0),
        available_memory_mb=available_budget / (1024.0 * 1024.0),
        peak_memory_fraction=peak_fraction,
    )


def try_training_batch_size(
    examples: list[PolicyValueTrainingExample],
    batch_size: int,
    network,
    device: str,
    amp_enabled: bool,
    amp_dtype,
    value_loss_weight: float,
) -> tuple[bool, float]:
    if torch is None or F is None:
        raise RuntimeError("Torch is required for batch-size probing")
    probe_examples = select_batch_probe_examples(examples, batch_size)
    if not probe_examples:
        return False, 0.0

    previous_mode = network.training
    network.train()
    peak = 0.0
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        batch = collate_training_examples(probe_examples, device)
        with torch.autocast(device_type=device.split(":")[0], dtype=amp_dtype, enabled=amp_enabled):
            logits, predicted_value = network(batch.candidate_features, batch.target_features)
            masked_logits = logits.masked_fill(~batch.candidate_mask, torch.finfo(logits.dtype).min)
            log_probs = torch.log_softmax(masked_logits, dim=-1)
            policy_loss = -(batch.policy_targets * log_probs).sum(dim=-1).mean()
            value_loss = F.mse_loss(predicted_value, batch.value_targets)
            loss = policy_loss + value_loss_weight * value_loss
        loss.backward()
        torch.cuda.synchronize(device)
        peak = float(torch.cuda.max_memory_allocated(device))
        del batch, logits, predicted_value, masked_logits, log_probs, policy_loss, value_loss, loss
        network.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        return True, peak
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower():
            raise
        network.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        return False, 0.0
    finally:
        if not previous_mode:
            network.eval()


def select_batch_probe_examples(
    examples: list[PolicyValueTrainingExample],
    batch_size: int,
) -> list[PolicyValueTrainingExample]:
    ranked = sorted(
        examples,
        key=lambda example: (
            len(example.candidates),
            example.target.support_size,
            example.target.total_degree,
            len(example.target.variables),
        ),
        reverse=True,
    )
    return ranked[: max(1, batch_size)]


def collate_training_examples(
    examples: list[PolicyValueTrainingExample],
    device: str,
) -> PolicyValueBatch:
    if torch is None:
        raise RuntimeError("Torch is required for collation")
    if not examples:
        raise ValueError("Cannot collate an empty example batch")

    feature_dim = len(candidate_feature_vector(examples[0].target, examples[0].candidates[0]))
    target_dim = len(target_feature_vector(examples[0].target))
    max_candidates = max(len(example.candidates) for example in examples)
    batch_size = len(examples)

    candidate_feature_rows: list[list[list[float]]] = []
    target_feature_rows: list[list[float]] = []
    policy_target_rows: list[list[float]] = []
    value_target_rows: list[float] = []
    candidate_mask_rows: list[list[bool]] = []
    zero_candidate = [0.0] * feature_dim

    for example in examples:
        target_feature_rows.append(list(target_feature_vector(example.target)))
        value_target_rows.append(float(example.value_target))
        candidate_rows = [
            list(candidate_feature_vector(example.target, action))
            for action in example.candidates
        ]
        candidate_count = len(candidate_rows)
        padding = max_candidates - candidate_count
        candidate_feature_rows.append(candidate_rows + [zero_candidate] * padding)
        policy_target_rows.append(list(example.policy_target) + [0.0] * padding)
        candidate_mask_rows.append(([True] * candidate_count) + ([False] * padding))

    candidate_features = torch.tensor(candidate_feature_rows, dtype=torch.float32)
    target_features = torch.tensor(target_feature_rows, dtype=torch.float32)
    policy_targets = torch.tensor(policy_target_rows, dtype=torch.float32)
    value_targets = torch.tensor(value_target_rows, dtype=torch.float32)
    candidate_mask = torch.tensor(candidate_mask_rows, dtype=torch.bool)

    use_pinned_memory = device.startswith("cuda") and torch.cuda.is_available()
    if use_pinned_memory:
        candidate_features = candidate_features.pin_memory()
        target_features = target_features.pin_memory()
        policy_targets = policy_targets.pin_memory()
        value_targets = value_targets.pin_memory()
        candidate_mask = candidate_mask.pin_memory()

    candidate_features = candidate_features.to(device, non_blocking=use_pinned_memory)
    target_features = target_features.to(device, non_blocking=use_pinned_memory)
    policy_targets = policy_targets.to(device, non_blocking=use_pinned_memory)
    value_targets = value_targets.to(device, non_blocking=use_pinned_memory)
    candidate_mask = candidate_mask.to(device, non_blocking=use_pinned_memory)

    return PolicyValueBatch(
        candidate_features=candidate_features,
        target_features=target_features,
        policy_targets=policy_targets,
        value_targets=value_targets,
        candidate_mask=candidate_mask,
        batch_size=batch_size,
    )


def pack_training_examples(
    examples: list[PolicyValueTrainingExample],
    pin_memory: bool = False,
) -> PackedTrainingExamples:
    if torch is None:
        raise RuntimeError("Torch is required for packing training examples")
    if not examples:
        empty_float = torch.empty((0,), dtype=torch.float32)
        empty_bool = torch.empty((0,), dtype=torch.bool)
        return PackedTrainingExamples(
            candidate_features=empty_float,
            target_features=empty_float,
            policy_targets=empty_float,
            value_targets=empty_float,
            candidate_mask=empty_bool,
            example_count=0,
        )

    feature_dim = len(candidate_feature_vector(examples[0].target, examples[0].candidates[0]))
    target_dim = len(target_feature_vector(examples[0].target))
    max_candidates = max(len(example.candidates) for example in examples)
    batch_size = len(examples)

    candidate_features = torch.zeros((batch_size, max_candidates, feature_dim), dtype=torch.float32)
    target_features = torch.zeros((batch_size, target_dim), dtype=torch.float32)
    policy_targets = torch.zeros((batch_size, max_candidates), dtype=torch.float32)
    value_targets = torch.zeros((batch_size,), dtype=torch.float32)
    candidate_mask = torch.zeros((batch_size, max_candidates), dtype=torch.bool)

    for index, example in enumerate(examples):
        target_features[index] = torch.tensor(target_feature_vector(example.target), dtype=torch.float32)
        value_targets[index] = float(example.value_target)
        candidate_count = len(example.candidates)
        for candidate_index, action in enumerate(example.candidates):
            candidate_features[index, candidate_index] = torch.tensor(
                candidate_feature_vector(example.target, action),
                dtype=torch.float32,
            )
        policy_targets[index, :candidate_count] = torch.tensor(example.policy_target, dtype=torch.float32)
        candidate_mask[index, :candidate_count] = True

    if pin_memory:
        candidate_features = candidate_features.pin_memory()
        target_features = target_features.pin_memory()
        policy_targets = policy_targets.pin_memory()
        value_targets = value_targets.pin_memory()
        candidate_mask = candidate_mask.pin_memory()

    return PackedTrainingExamples(
        candidate_features=candidate_features,
        target_features=target_features,
        policy_targets=policy_targets,
        value_targets=value_targets,
        candidate_mask=candidate_mask,
        example_count=batch_size,
    )


def move_packed_training_examples(
    examples: PackedTrainingExamples,
    device: str,
    non_blocking: bool,
) -> PackedTrainingExamples:
    return PackedTrainingExamples(
        candidate_features=examples.candidate_features.to(device, non_blocking=non_blocking),
        target_features=examples.target_features.to(device, non_blocking=non_blocking),
        policy_targets=examples.policy_targets.to(device, non_blocking=non_blocking),
        value_targets=examples.value_targets.to(device, non_blocking=non_blocking),
        candidate_mask=examples.candidate_mask.to(device, non_blocking=non_blocking),
        example_count=examples.example_count,
    )


def slice_training_batch(
    examples: PackedTrainingExamples,
    indices: list[int],
    device: str,
) -> PolicyValueBatch:
    if torch is None:
        raise RuntimeError("Torch is required for packed batch slicing")
    index_tensor = torch.tensor(indices, dtype=torch.long, device=examples.value_targets.device)
    use_non_blocking = device.startswith("cuda") and torch.cuda.is_available()
    candidate_features = examples.candidate_features.index_select(0, index_tensor)
    target_features = examples.target_features.index_select(0, index_tensor)
    policy_targets = examples.policy_targets.index_select(0, index_tensor)
    value_targets = examples.value_targets.index_select(0, index_tensor)
    candidate_mask = examples.candidate_mask.index_select(0, index_tensor)

    if candidate_features.device.type != device.split(":")[0]:
        candidate_features = candidate_features.to(device, non_blocking=use_non_blocking)
        target_features = target_features.to(device, non_blocking=use_non_blocking)
        policy_targets = policy_targets.to(device, non_blocking=use_non_blocking)
        value_targets = value_targets.to(device, non_blocking=use_non_blocking)
        candidate_mask = candidate_mask.to(device, non_blocking=use_non_blocking)

    return PolicyValueBatch(
        candidate_features=candidate_features,
        target_features=target_features,
        policy_targets=policy_targets,
        value_targets=value_targets,
        candidate_mask=candidate_mask,
        batch_size=len(indices),
    )
