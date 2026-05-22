#!/usr/bin/env python3
"""Compare multiple runs (e.g. PPO vs SAC) on the held-out Ck metric.

Reads each run's metrics.jsonl (the same data as W&B) and renders two
publication-quality comparison figures:
  * held-out Ck-match rate over iterations (one line per run),
  * final per-Ck held-out match rate as grouped bars (one bar per run + random).

Example:
    python scripts/compare_runs.py \
        --run "PPO+MCTS" artifacts/ck_curriculum/mcts-big/metrics.jsonl \
        --run "SAC"      artifacts/ck_curriculum/sac-big-tuned/metrics.jsonl \
        --outdir paper_plots/comparison_ppo_vs_sac \
        --title "Top-down circuit discovery: PPO+MCTS vs SAC"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 13, "axes.titlesize": 16, "axes.titleweight": "bold",
    "axes.labelsize": 14, "legend.fontsize": 11, "legend.frameon": False,
    "xtick.labelsize": 12, "ytick.labelsize": 12,
    "axes.grid": True, "grid.alpha": 0.30,
    "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 2.2, "figure.constrained_layout.use": True,
})
PALETTE = ["#1f6feb", "#d1495b", "#2a9d8f", "#9b5de5", "#e76f51"]
REFERENCE = "#888888"


def load_evals(path: Path):
    return [json.loads(l) for l in path.open(encoding="utf-8")
            if l.strip() and '"heldout/ck_match_rate"' in l]


def ck_buckets(evals) -> list[int]:
    cks = set()
    for r in evals[-1:]:
        for k in r:
            if k.startswith("heldout/C") and k.endswith("/match_rate"):
                cks.add(int(k[len("heldout/C"):-len("/match_rate")]))
    return sorted(cks)


def _save(fig, outdir, name):
    for ext in ("png", "pdf"):
        fig.savefig(outdir / f"{name}.{ext}")
    plt.close(fig)
    print(f"  wrote {name}.png / .pdf", flush=True)


def plot_heldout_over_iters(runs, outdir, title):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ref = None
    for i, (label, evals) in enumerate(runs):
        it = [r["iteration"] for r in evals]
        y = [r["heldout/ck_match_rate"] for r in evals]
        ax.plot(it, y, color=PALETTE[i % len(PALETTE)], marker="o", markersize=3, label=label)
        ref = ref or evals[0].get("baseline_heldout/random_ck_match_rate")
    if ref is not None:
        ax.axhline(ref, color=REFERENCE, linestyle="--", linewidth=1.8, label=f"random ({ref:.2f})")
    ax.set_xlabel("training iteration"); ax.set_ylabel("held-out Ck-match rate")
    ax.set_ylim(0, 1.02)
    ax.set_title(f"Held-out success over training\n{title}")
    ax.legend(loc="lower right")
    _save(fig, outdir, "cmp_01_heldout_over_iterations")


def plot_final_per_ck(runs, outdir, title):
    cks = ck_buckets(runs[0][1])
    x = np.arange(len(cks))
    n = len(runs) + 1  # + random
    w = 0.8 / n
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (label, evals) in enumerate(runs):
        f = evals[-1]
        vals = [f.get(f"heldout/C{ck}/match_rate", 0.0) for ck in cks]
        ax.bar(x + (i - n / 2 + 0.5) * w, vals, w, label=label, color=PALETTE[i % len(PALETTE)], edgecolor="white")
    rnd = [runs[0][1][-1].get(f"baseline_heldout/C{ck}/random_match_rate", 0.0) for ck in cks]
    ax.bar(x + (n - 1 - n / 2 + 0.5) * w, rnd, w, label="random", color=REFERENCE, edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels([f"C{ck}" for ck in cks])
    ax.set_xlabel("circuit complexity Ck (ops)"); ax.set_ylabel("held-out Ck-match rate (final)")
    ax.set_ylim(0, 1.08)
    ax.set_title(f"Final held-out success per complexity\n{title}")
    ax.legend(loc="upper right", ncol=n)
    _save(fig, outdir, "cmp_02_final_per_ck")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", action="append", nargs=2, metavar=("LABEL", "METRICS"), required=True,
                   help="Repeatable: a run label and its metrics.jsonl path.")
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--title", default="")
    args = p.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    runs = [(label, load_evals(Path(path))) for label, path in args.run]
    for label, evals in runs:
        if not evals:
            raise SystemExit(f"No held-out eval rows for run {label!r}")
        print(f"{label}: {len(evals)} eval points, final held-out={evals[-1]['heldout/ck_match_rate']:.3f}")

    plot_heldout_over_iters(runs, args.outdir, args.title)
    plot_final_per_ck(runs, args.outdir, args.title)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
