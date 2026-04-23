"""Disk cache for ExpertDemoGenerator graph artifacts.

Keyed by (config graph-gen fields) + sha256 of the source files whose logic
produces the cache payload, so any edit to graph construction or Poly hashing
automatically invalidates stale caches.
"""

from __future__ import annotations

import hashlib
import os
import pickle
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

_REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = _REPO_ROOT / "data" / "expert_graph_cache"

SOURCE_FILES = (
    "poly_circuit_rl/env/graph_enumeration.py",
    "poly_circuit_rl/env/expert_demos.py",
    "poly_circuit_rl/core/poly.py",
)


def compute_cache_key(
    *,
    n_vars: int,
    steps: int,
    gen_max_graph_nodes: Optional[int],
    gen_max_successors: Optional[int],
    gen_max_seconds: Optional[float],
) -> str:
    h = hashlib.sha256()
    h.update(
        repr(
            (
                n_vars,
                steps,
                gen_max_graph_nodes,
                gen_max_successors,
                gen_max_seconds,
            )
        ).encode()
    )
    for rel in SOURCE_FILES:
        p = _REPO_ROOT / rel
        try:
            h.update(p.read_bytes())
        except OSError:
            h.update(b"__missing__" + rel.encode())
    return h.hexdigest()[:16]


def cache_path(key: str) -> Path:
    return CACHE_DIR / f"graph_{key}.pkl"


def load(key: str) -> Optional[Dict[str, Any]]:
    p = cache_path(key)
    if not p.exists():
        return None
    try:
        with p.open("rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def save(key: str, payload: Dict[str, Any]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(CACHE_DIR), suffix=".pkl.tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, cache_path(key))
    except Exception:
        Path(tmp).unlink(missing_ok=True)
        raise
