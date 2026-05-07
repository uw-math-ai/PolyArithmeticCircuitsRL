"""Search-distillation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .andor_search import AndOrSearch
from .baseline_cost import BaselineCostModel
from .elite_buffer import EliteBuffer
from .factor_fp import FiniteFieldFactorizer
from .polynomial import SparsePolynomial
from .split_proposals import propose_splits
from .split_proposals import SplitAction
from .train_supervised import PolicyValueTrainingExample


@dataclass(frozen=True)
class DistilledExample:
    target: SparsePolynomial
    candidates: tuple[SplitAction, ...]
    search_policy: tuple[float, ...]
    value_target: float
    best_cost: float


def distill_targets(
    targets,
    search: AndOrSearch,
    elite_buffer: EliteBuffer | None = None,
    fresh_search_per_target: bool = False,
    retry_failures: int = 1,
    progress_callback: Callable[[dict[str, object]], None] | None = None,
) -> list[DistilledExample]:
    distilled = []
    working_search = search
    for index, target in enumerate(targets):
        attempts = 0
        while True:
            if fresh_search_per_target:
                working_search = _fresh_search_like(search)
            if progress_callback is not None:
                progress_callback(
                    {
                        "status": "start",
                        "target_index": index,
                        "attempt": attempts + 1,
                        "support_size": target.support_size,
                        "total_degree": target.total_degree,
                        "variable_count": len(target.variables),
                    }
                )
            try:
                result = working_search.search(target)
                break
            except Exception as exc:
                if progress_callback is not None:
                    progress_callback(
                        {
                            "status": "error",
                            "target_index": index,
                            "attempt": attempts + 1,
                            "error": repr(exc),
                        }
                    )
                working_search.close()
                attempts += 1
                if attempts > retry_failures:
                    raise RuntimeError(
                        f"Failed to distill target {index} after {attempts} attempts"
                    ) from exc
                working_search = _fresh_search_like(search)

        baseline = search.baseline_model.direct_construction_cost(target)
        if elite_buffer is not None:
            elite_buffer.maybe_add(result.best_trace, baseline_cost=baseline)
        distilled.append(
            DistilledExample(
                target=target,
                candidates=result.root_candidates,
                search_policy=result.root_policy,
                value_target=result.root_value,
                best_cost=result.best_cost,
            )
        )
        if progress_callback is not None:
            progress_callback(
                {
                    "status": "done",
                    "target_index": index,
                    "best_cost": result.best_cost,
                    "candidate_count": len(result.root_candidates),
                    "factor_cache_size": working_search.factorizer.cache_size,
                }
            )
        working_search.factorizer.clear()
        if fresh_search_per_target:
            working_search.close()
    return distilled


def _fresh_search_like(search: AndOrSearch) -> AndOrSearch:
    return AndOrSearch(
        factorizer=FiniteFieldFactorizer(search.factorizer.config),
        baseline_model=search.baseline_model,
        model=search.model,
        search_config=search.search_config,
    )


def make_distillation_training_examples(
    examples: list[DistilledExample],
) -> list[PolicyValueTrainingExample]:
    return [
        PolicyValueTrainingExample(
            target=example.target,
            candidates=tuple(example.candidates),
            policy_target=tuple(example.search_policy),
            value_target=example.value_target,
            source="search_distill",
        )
        for example in examples
        if example.candidates
    ]


def make_elite_training_examples(
    elite_entries,
    baseline_model: BaselineCostModel | None = None,
    k_candidates: int = 16,
) -> list[PolicyValueTrainingExample]:
    baseline_model = baseline_model or BaselineCostModel()
    training_examples: list[PolicyValueTrainingExample] = []
    for entry in elite_entries:
        for trace in _iter_trace_nodes(entry.trace):
            if trace.chosen_action is None:
                continue
            candidates = tuple(propose_splits(trace.poly, k_candidates, baseline_model=baseline_model))
            if trace.chosen_action.key() not in {candidate.key() for candidate in candidates}:
                candidates = (trace.chosen_action.ordered(),) + candidates[: max(0, k_candidates - 1)]
            chosen_index = next(
                index
                for index, candidate in enumerate(candidates)
                if candidate.key() == trace.chosen_action.key()
            )
            policy_target = [0.0] * len(candidates)
            policy_target[chosen_index] = 1.0
            baseline = baseline_model.direct_construction_cost(trace.poly)
            value_target = (baseline - trace.total_cost) / max(1.0, float(baseline))
            training_examples.append(
                PolicyValueTrainingExample(
                    target=trace.poly,
                    candidates=tuple(candidates),
                    policy_target=tuple(policy_target),
                    value_target=value_target,
                    source="elite_self_imitation",
                )
            )
    return training_examples


def _iter_trace_nodes(trace):
    yield trace
    for child in trace.children:
        yield from _iter_trace_nodes(child)
