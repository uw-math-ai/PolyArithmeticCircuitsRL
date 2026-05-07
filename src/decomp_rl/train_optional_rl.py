"""Optional RL fine-tuning placeholders.

The roadmap positions policy-gradient fine-tuning after the search-distillation
stack is stable. This module intentionally stays thin in the MVP so the project
has a clean extension point without implying that PPO is production-ready here.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OptionalRLConfig:
    algorithm: str = "ppo"
    total_steps: int = 10000


def run_optional_rl(*args, **kwargs):  # pragma: no cover - intentionally deferred
    raise NotImplementedError(
        "Optional RL fine-tuning is intentionally deferred until the symbolic/search stack is validated."
    )

