"""Configuration objects for the MVP research stack."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


def _default_cas_python_path() -> str | None:
    repo_root = Path(__file__).resolve().parents[2]
    candidate = repo_root / ".cas_env" / "bin" / "python"
    if candidate.exists():
        return str(candidate)
    return None


@dataclass(frozen=True)
class FactorizerConfig:
    cache_enabled: bool = True
    backend_name: str = "auto"
    cas_python_path: str | None = field(default_factory=_default_cas_python_path)
    helper_startup_timeout_sec: float = 30.0


@dataclass(frozen=True)
class ProposalConfig:
    max_candidates: int = 32
    support_fraction: float = 0.5
    horner_fraction: float = 0.2
    common_factor_fraction: float = 0.15
    family_fraction: float = 0.1
    random_fraction: float = 0.05
    random_seed: int = 0
    max_random_masks: int = 8


@dataclass(frozen=True)
class DecompEnvConfig:
    proposal: ProposalConfig = field(default_factory=ProposalConfig)
    dedup_frontier: bool = True
    exact_support_limit: int = 3


@dataclass(frozen=True)
class SearchConfig:
    simulations: int = 96
    max_depth: int = 8
    puct_exploration: float = 1.25
    expand_top_k: int = 32


@dataclass(frozen=True)
class ProjectConfig:
    prime: int = 3
    variables: tuple[str, ...] = ("x", "y")
    factorizer: FactorizerConfig = field(default_factory=FactorizerConfig)
    env: DecompEnvConfig = field(default_factory=DecompEnvConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
