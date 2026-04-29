"""Small multiprocessing helper for per-complexity cache builds."""

from __future__ import annotations

import multiprocessing as mp
import os
from typing import Callable, Iterable, TypeVar


R = TypeVar("R")
T = TypeVar("T")


def _worker_init() -> None:
    # Keep BLAS/OpenMP libraries from oversubscribing CPU cores in each worker.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


def map_complexities(
    fn: Callable[[T], R],
    items: Iterable[T],
    *,
    processes: int | None = None,
    force_serial: bool = False,
) -> list[R]:
    """Map ``fn`` over complexity work items, optionally using spawn workers."""
    item_list = list(items)
    if force_serial or len(item_list) <= 1 or processes == 1:
        return [fn(item) for item in item_list]

    worker_count = processes
    if worker_count is None:
        worker_count = min(len(item_list), os.cpu_count() or 1)
    else:
        worker_count = min(int(worker_count), len(item_list))
    if worker_count <= 1:
        return [fn(item) for item in item_list]

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=worker_count, initializer=_worker_init) as pool:
        return pool.map(fn, item_list)
