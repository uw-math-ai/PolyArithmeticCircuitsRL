"""Elite trace buffer for self-imitation."""

from __future__ import annotations

from dataclasses import dataclass

from .andor_search import DecompositionTrace


@dataclass(frozen=True)
class EliteEntry:
    trace: DecompositionTrace
    root_key: str
    cost: float
    saving: float
    family: str | None = None


class EliteBuffer:
    def __init__(self, capacity: int = 256) -> None:
        self.capacity = capacity
        self._entries: dict[str, EliteEntry] = {}

    def __len__(self) -> int:
        return len(self._entries)

    def maybe_add(
        self,
        trace: DecompositionTrace,
        baseline_cost: float,
        family: str | None = None,
    ) -> bool:
        root_key = trace.poly.to_key()
        saving = baseline_cost - trace.total_cost
        candidate = EliteEntry(
            trace=trace,
            root_key=root_key,
            cost=trace.total_cost,
            saving=saving,
            family=family,
        )
        current = self._entries.get(root_key)
        if current is not None and current.cost <= candidate.cost:
            return False
        self._entries[root_key] = candidate
        if len(self._entries) > self.capacity:
            worst_key = min(self._entries, key=lambda key: self._entries[key].saving)
            self._entries.pop(worst_key)
        return True

    def sample(self, limit: int) -> list[EliteEntry]:
        ordered = sorted(self._entries.values(), key=lambda entry: entry.saving, reverse=True)
        return ordered[:limit]

