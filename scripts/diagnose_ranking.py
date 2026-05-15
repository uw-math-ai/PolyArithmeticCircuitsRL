#!/usr/bin/env python3
"""Diagnose candidate ranking: compare heuristic vs learned score ordering.

Checks:
  1. Top-K candidate table by heuristic rank vs model rank — shows whether
     the learned score changes candidate ordering at all.
  5. Score distributions (min/max/mean/std) and scale ratio
     model_score_std / heuristic_score_std — if this is near zero, the model
     output is too small to affect additive ranking regardless of quality.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from lgs.data.benchmark_suite import make_structured_benchmark
from lgs.search.beam_search import beam_search
from lgs.training.train_ranker import load_ranker


# ---------------------------------------------------------------------------
# Pure-Python stat helpers (no numpy/scipy in venv)
# ---------------------------------------------------------------------------

def _stats(values: list[float]) -> dict[str, float]:
    n = len(values)
    if n == 0:
        return {"min": float("nan"), "max": float("nan"),
                "mean": float("nan"), "std": float("nan")}
    mn, mx = min(values), max(values)
    mean = sum(values) / n
    std = math.sqrt(sum((v - mean) ** 2 for v in values) / n)
    return {"min": mn, "max": mx, "mean": mean, "std": std}


def _spearman_r(xs: list[float], ys: list[float]) -> float:
    """Spearman rank correlation (competition ranking, no tie averaging)."""
    n = len(xs)
    if n < 2:
        return float("nan")

    def _ranks(vals: list[float]) -> list[float]:
        order = sorted(range(n), key=lambda k: vals[k])
        r = [0.0] * n
        for rank, idx in enumerate(order):
            r[idx] = float(rank + 1)
        return r

    rx, ry = _ranks(xs), _ranks(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    dx = math.sqrt(sum((rx[i] - mx) ** 2 for i in range(n)))
    dy = math.sqrt(sum((ry[i] - my) ** 2 for i in range(n)))
    if dx == 0.0 or dy == 0.0:
        return float("nan")
    return num / (dx * dy)


# ---------------------------------------------------------------------------
# Per-instance diagnostic
# ---------------------------------------------------------------------------

def _print_instance_diagnostics(
    instance,
    history,
    *,
    top_k: int,
    max_states: int,
    learned_only: bool,
    beam_width: int = 8,
) -> None:
    instance_id = str(instance.metadata.get("id", instance.metadata.get("target_id", "?")))
    family = str(instance.metadata.get("family", instance.family_name))
    print(f"\n{'=' * 72}")
    print(f"Instance: {instance_id}  family={family}  op_budget={instance.op_budget}")
    print(f"Success: {history.success()}  |  expanded records: {len(history.records)}")

    # Group records by state (identified by its action-tuple prefix)
    state_groups: dict[tuple, list] = defaultdict(list)
    for record in history.records:
        key = tuple((a.op, a.i, a.j) for a in record.state.actions)
        state_groups[key].append(record)

    # Accumulate all candidates across all states for global Spearman
    all_h: list[float] = []
    all_t: list[float] = []
    all_m: list[float] = []

    for state_idx, (state_key, group) in enumerate(state_groups.items()):
        if state_idx >= max_states:
            break

        # All records in a group share the same candidates list
        candidates = group[0].candidates
        n = len(candidates)

        h_scores = [c.heuristic_score for c in candidates]
        m_scores = [c.model_score for c in candidates]
        t_scores = [c.total_score for c in candidates]
        all_h.extend(h_scores)
        all_m.extend(m_scores)
        all_t.extend(t_scores)

        h_sorted = sorted(range(n), key=lambda i: -h_scores[i])
        m_sorted = sorted(range(n), key=lambda i: -m_scores[i])
        h_rank_of = {idx: r for r, idx in enumerate(h_sorted)}
        m_rank_of = {idx: r for r, idx in enumerate(m_sorted)}

        depth = group[0].depth
        print(f"\n  --- State {state_idx} (depth={depth}, {n} candidates) ---")

        # Score distributions
        h_st = _stats(h_scores)
        m_st = _stats(m_scores)
        t_st = _stats(t_scores)
        print(f"  heuristic  min={h_st['min']:+9.4f}  max={h_st['max']:+9.4f}"
              f"  mean={h_st['mean']:+9.4f}  std={h_st['std']:.4f}")
        print(f"  model      min={m_st['min']:+9.4f}  max={m_st['max']:+9.4f}"
              f"  mean={m_st['mean']:+9.4f}  std={m_st['std']:.4f}")
        print(f"  total      min={t_st['min']:+9.4f}  max={t_st['max']:+9.4f}"
              f"  mean={t_st['mean']:+9.4f}  std={t_st['std']:.4f}")

        # Scale ratio: can model scores compete with heuristic scores?
        if h_st["std"] > 0:
            scale_ratio = m_st["std"] / h_st["std"]
            flag = "  <<< near-zero, model cannot affect additive ranking" if scale_ratio < 0.01 else ""
            print(f"  scale ratio  model_std / heuristic_std = {scale_ratio:.4f}{flag}")
        else:
            print("  scale ratio  heuristic_std=0, cannot compute")

        if m_st["std"] < 0.01:
            print(f"  WARNING: model_score std={m_st['std']:.6f} < 0.01 — output is near-constant")

        # Top-K ranking table
        k = min(top_k, n)
        mode_label = "model_score (learned-only)" if learned_only else "heuristic_score + lambda*model"
        print(f"\n  Top-{k} by heuristic rank  [ranking mode: {mode_label}]")
        print(f"  {'h_rank':>6}  {'m_rank':>6}  {'Δrank':>6}  "
              f"{'h_score':>10}  {'m_score':>10}  {'total':>10}  action")
        for h_r in range(k):
            idx = h_sorted[h_r]
            c = candidates[idx]
            m_r = m_rank_of[idx]
            delta = m_r - h_r
            act = f"{c.action.op}({c.action.i},{c.action.j})"
            print(f"  {h_r:>6}  {m_r:>6}  {delta:>+6}  "
                  f"{c.heuristic_score:>10.4f}  {c.model_score:>10.4f}  "
                  f"{c.total_score:>10.4f}  {act}")

        # Beam-survivor Jaccard
        beam_k = min(beam_width, n)
        h_top_k = set(h_sorted[:beam_k])
        t_sorted_idx = sorted(range(n), key=lambda i: -t_scores[i])
        t_top_k = set(t_sorted_idx[:beam_k])
        jaccard = len(h_top_k & t_top_k) / len(h_top_k | t_top_k)
        added = t_top_k - h_top_k
        dropped = h_top_k - t_top_k
        print(f"  Beam-{beam_k} Jaccard(h,total)={jaccard:.4f}  "
              f"guided_adds={len(added)}  guided_drops={len(dropped)}")

    # Per-instance Spearman
    print()
    if len(all_h) >= 2:
        rho = _spearman_r(all_h, all_t)
        n_shown = min(max_states, len(state_groups))
        print(f"  Spearman rho (heuristic vs total, {len(all_h)} candidates "
              f"across {n_shown} states): {rho:.4f}")
        if rho > 0.99:
            print("  WARNING: rho > 0.99 — learned score not changing candidate ordering")
    else:
        print("  Spearman rho: insufficient data")

    # Model score variance check across all states
    if all_m:
        m_global = _stats(all_m)
        print(f"  Global model_score: mean={m_global['mean']:.4f}  std={m_global['std']:.4f}")
        if m_global["std"] < 0.01:
            print("  WARNING: global model_score near-constant across all states/candidates")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    ranker = None
    encoder = None
    lambda_model = 0.0
    learned_only = False

    if args.checkpoint:
        path = Path(args.checkpoint)
        if not path.exists():
            print(f"ERROR: checkpoint not found: {path}", file=sys.stderr)
            sys.exit(1)
        ranker, encoder = load_ranker(path)
        lambda_model = 1.0
        learned_only = args.learned_only
        mode = "learned-only  (total = model_score)" if learned_only else "additive  (total = heuristic + 1.0 * model)"
        print(f"Loaded ranker: {path}")
        print(f"Ranking mode: {mode}")
    else:
        print("No checkpoint — heuristic-only baseline (model_score=0 everywhere)")

    benchmark = make_structured_benchmark(
        max_instances_per_family=args.max_instances_per_family,
    )
    instances = benchmark.instances
    print(f"Benchmark instances: {len(instances)} "
          f"(max_instances_per_family={args.max_instances_per_family})")

    for instance in instances:
        history = beam_search(
            instance,
            ranker=ranker,
            encoder=encoder,
            lambda_model=lambda_model,
            learned_only=learned_only,
            beam_width=args.beam_width,
            candidate_k=args.candidate_k,
            tier2_m=128,
        )
        _print_instance_diagnostics(
            instance,
            history,
            top_k=args.top_k,
            max_states=args.max_states,
            learned_only=learned_only,
            beam_width=args.beam_width,
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="",
                        help="Path to ranker.pt (omit for heuristic-only baseline)")
    parser.add_argument("--learned-only", action="store_true",
                        help="Use model score alone for ranking (ignores heuristic)")
    parser.add_argument("--max-instances-per-family", type=int, default=2)
    parser.add_argument("--beam-width", type=int, default=8)
    parser.add_argument("--candidate-k", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=10,
                        help="Candidates to show in ranking table (default: 10)")
    parser.add_argument("--max-states", type=int, default=3,
                        help="Max expanded states to inspect per instance (default: 3)")
    return parser.parse_args()


if __name__ == "__main__":
    main()
