"""Prioritized replay buffers for decomposition targets."""

from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass
class ReplayItem(Generic[T]):
    payload: T
    priority: float


class PrioritizedReplayBuffer(Generic[T]):
    def __init__(self, capacity: int = 1024, alpha: float = 1.0, seed: int = 0) -> None:
        self.capacity = capacity
        self.alpha = alpha
        self._rng = Random(seed)
        self._items: list[ReplayItem[T]] = []

    def __len__(self) -> int:
        return len(self._items)

    def add(self, payload: T, priority: float) -> None:
        item = ReplayItem(payload=payload, priority=max(1e-6, priority))
        if len(self._items) >= self.capacity:
            min_index = min(range(len(self._items)), key=lambda idx: self._items[idx].priority)
            if self._items[min_index].priority >= item.priority:
                return
            self._items[min_index] = item
        else:
            self._items.append(item)

    def sample(self, batch_size: int, uniform_fraction: float = 0.1) -> list[T]:
        if not self._items:
            return []
        requested = min(batch_size, len(self._items))
        uniform_count = min(requested, max(0, int(round(requested * uniform_fraction))))
        prioritized_count = max(0, requested - uniform_count)

        chosen: list[ReplayItem[T]] = []
        if prioritized_count:
            weights = [item.priority ** self.alpha for item in self._items]
            chosen.extend(self._rng.choices(self._items, weights=weights, k=prioritized_count))
        if uniform_count:
            chosen.extend(self._rng.sample(self._items, k=uniform_count))
        self._rng.shuffle(chosen)
        return [item.payload for item in chosen]

    def update_priorities(self, pairs: list[tuple[int, float]]) -> None:
        for index, priority in pairs:
            if 0 <= index < len(self._items):
                self._items[index].priority = max(1e-6, priority)
