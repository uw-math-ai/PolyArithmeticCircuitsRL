#!/usr/bin/env python3
"""Publication-quality plots from a run's metrics.jsonl (same data as W&B).

Reads the per-iteration log written by ``run_ppo_curriculum.py`` and renders a
set of clean, large-font figures (PNG at 300 dpi + vector PDF) into an output
folder. Used to produce the paper figures for the PPO+MCTS Ck run.

Example:
    python scripts/plot_training.py \
        --metrics artifacts/ck_curriculum/mcts-1000/metrics.jsonl \
        --outdir  paper_plots/top-down-ppo-ck-mcts-1000 \
        --title "PPO+MCTS (Ck curriculum, factor action)"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless / file output only
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

# ── Shared paper style: large fonts, light grid, no top/right spines ─────────
plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 13,
    "axes.titlesize": 16,
    "axes.titleweight": "bold",
    "axes.labelsize": 14,
    "legend.fontsize": 11,
    "legend.frameon": False,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "axes.grid": True,
    "grid.alpha": 0.30,
    "grid.linestyle": "-",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "lines.linewidth": 2.2,
    "figure.constrained_layout.use": True,
})
ACCENT = "#1f6feb"     # primary line
REFERENCE = "#888888"  # random/reference line
CMAP = "viridis"       # complexity gradient (perceptually uniform)


def load_metrics(path: Path):
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    evals = [r for r in rows if "eval/ck_match_rate" in r]
    return rows, evals


def _save(fig, outdir: Path, name: str):
    for ext in ("png", "pdf"):
        fig.savefig(outdir / f"{name}.{ext}")
    plt.close(fig)
    print(f"  wrote {name}.png / .pdf", flush=True)


def _ck_buckets(evals) -> list[int]:
    cks = set()
    for r in evals:
        for k in r:
            if k.startswith("eval/C") and k.endswith("/match_rate"):
                cks.add(int(k[len("eval/C"):-len("/match_rate")]))
    return sorted(cks)


def _rolling(values, window: int = 21):
    if window <= 1 or len(values) < 2:
        return values
    out = []
    half = window // 2
    for i in range(len(values)):
        lo, hi = max(0, i - half), min(len(values), i + half + 1)
        chunk = values[lo:hi]
        out.append(sum(chunk) / len(chunk))
    return out


def plot_success_overall(evals, outdir: Path, title: str):
    it = [r["iteration"] for r in evals]
    y = [r["eval/ck_match_rate"] for r in evals]
    ref = evals[0].get("baseline/random_ck_match_rate")
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(it, y, color=ACCENT, marker="o", markersize=4, label="learned policy (greedy)")
    if ref is not None:
        ax.axhline(ref, color=REFERENCE, linestyle="--", linewidth=1.8,
                   label=f"random policy ({ref:.2f})")
    ax.set_xlabel("PPO iteration")
    ax.set_ylabel("Ck-match rate")
    ax.set_ylim(0, 1.02)
    ax.set_title(f"Success rate over training\n{title}")
    ax.legend(loc="lower right")
    _save(fig, outdir, "01_success_rate_over_iterations")


def plot_success_by_complexity(evals, outdir: Path, title: str):
    cks = _ck_buckets(evals)
    it = [r["iteration"] for r in evals]
    norm = Normalize(vmin=min(cks), vmax=max(cks))
    cmap = plt.get_cmap(CMAP)
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    for ck in cks:
        y = [r.get(f"eval/C{ck}/match_rate", float("nan")) for r in evals]
        ax.plot(it, y, color=cmap(norm(ck)), marker="o", markersize=3, label=f"C{ck}")
    ax.set_xlabel("PPO iteration")
    ax.set_ylabel("Ck-match rate")
    ax.set_ylim(-0.02, 1.04)
    ax.set_title(f"Success rate by circuit complexity\n{title}")
    sm = ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("complexity Ck (ops)")
    ax.legend(ncol=2, loc="lower right", title="bucket")
    _save(fig, outdir, "02_success_rate_by_complexity")


def plot_final_bars(evals, outdir: Path, title: str):
    cks = _ck_buckets(evals)
    final = evals[-1]
    rates = [final.get(f"eval/C{ck}/match_rate", 0.0) for ck in cks]
    norm = Normalize(vmin=min(cks), vmax=max(cks))
    colors = [plt.get_cmap(CMAP)(norm(ck)) for ck in cks]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    bars = ax.bar([f"C{ck}" for ck in cks], rates, color=colors, edgecolor="white")
    for b, r in zip(bars, rates):
        ax.text(b.get_x() + b.get_width() / 2, r + 0.02, f"{r:.2f}",
                ha="center", va="bottom", fontsize=11)
    ax.set_xlabel("circuit complexity Ck (ops)")
    ax.set_ylabel("Ck-match rate (final)")
    ax.set_ylim(0, 1.08)
    ax.set_title(f"Final success rate per complexity\n{title}")
    _save(fig, outdir, "03_final_success_by_complexity")


def plot_gap(evals, outdir: Path, title: str):
    it = [r["iteration"] for r in evals]
    y = [r["eval/mean_gap_to_optimal"] for r in evals]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(it, y, color="#d1495b", marker="o", markersize=4)
    ax.axhline(0, color=REFERENCE, linestyle="--", linewidth=1.5)
    ax.set_xlabel("PPO iteration")
    ax.set_ylabel("mean gap to optimal (ops)")
    ax.set_title(f"Distance from the shortest circuit\n{title}")
    ax.set_ylim(bottom=min(0, min(y) - 0.05))
    _save(fig, outdir, "04_gap_to_optimal_over_iterations")


def plot_training_curves(rows, outdir: Path, title: str):
    it = [r["iteration"] for r in rows]
    panels = [
        ("train/mean_episode_reward", "mean episode reward", ACCENT, True),
        ("train/entropy", "policy entropy", "#2a9d8f", False),
        ("train/policy_loss", "policy loss", "#e76f51", True),
        ("train/value_loss", "value loss (Tanh-capped; see caption)", "#9b5de5", False),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, (key, label, color, smooth) in zip(axes.flat, panels):
        y = [r.get(key, float("nan")) for r in rows]
        if smooth:
            ax.plot(it, y, color=color, alpha=0.25, linewidth=1.0)
            ax.plot(it, _rolling(y), color=color, linewidth=2.2)
        else:
            ax.plot(it, y, color=color)
        ax.set_xlabel("PPO iteration")
        ax.set_ylabel(label)
    fig.suptitle(f"Training diagnostics — {title}", fontsize=16, fontweight="bold")
    _save(fig, outdir, "05_training_curves")


def plot_train_vs_heldout(evals, outdir: Path, title: str):
    """Overlay train and held-out Ck-match rate (generalization check)."""
    if not any("heldout/ck_match_rate" in r for r in evals):
        return
    it = [r["iteration"] for r in evals]
    tr = [r.get("eval/ck_match_rate", float("nan")) for r in evals]
    ho = [r.get("heldout/ck_match_rate", float("nan")) for r in evals]
    ref = evals[0].get("baseline_heldout/random_ck_match_rate",
                       evals[0].get("baseline_train/random_ck_match_rate"))
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(it, tr, color=ACCENT, marker="o", markersize=4, label="train")
    ax.plot(it, ho, color="#e76f51", marker="s", markersize=4, label="held-out (unseen)")
    if ref is not None:
        ax.axhline(ref, color=REFERENCE, linestyle="--", linewidth=1.8, label=f"random ({ref:.2f})")
    ax.set_xlabel("PPO iteration"); ax.set_ylabel("Ck-match rate")
    ax.set_ylim(0, 1.02)
    ax.set_title(f"Generalization: train vs held-out\n{title}")
    ax.legend(loc="lower right")
    _save(fig, outdir, "06_train_vs_heldout")


def plot_agent_vs_random_final(evals, outdir: Path, title: str):
    """Grouped bars: final per-Ck agent match rate vs random reference."""
    final = evals[-1]
    cks = _ck_buckets(evals)
    agent = [final.get(f"eval/C{ck}/match_rate", 0.0) for ck in cks]
    rand = [final.get(f"baseline_train/C{ck}/random_match_rate", float("nan")) for ck in cks]
    if all(r != r for r in rand):  # no random per-Ck recorded
        return
    import numpy as np
    x = np.arange(len(cks)); w = 0.4
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(x - w / 2, agent, w, label="learned policy", color=ACCENT, edgecolor="white")
    ax.bar(x + w / 2, rand, w, label="random policy", color=REFERENCE, edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels([f"C{ck}" for ck in cks])
    ax.set_xlabel("circuit complexity Ck (ops)"); ax.set_ylabel("Ck-match rate")
    ax.set_ylim(0, 1.08)
    ax.set_title(f"Learned vs random, per complexity (final)\n{title}")
    ax.legend(loc="upper right")
    _save(fig, outdir, "07_agent_vs_random_by_complexity")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--metrics", type=Path, required=True)
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--title", default="")
    args = p.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    rows, evals = load_metrics(args.metrics)
    if not evals:
        raise SystemExit(f"No eval rows in {args.metrics}")
    print(f"Loaded {len(rows)} iters, {len(evals)} eval points -> {args.outdir}", flush=True)

    plot_success_overall(evals, args.outdir, args.title)
    plot_success_by_complexity(evals, args.outdir, args.title)
    plot_final_bars(evals, args.outdir, args.title)
    plot_gap(evals, args.outdir, args.title)
    plot_training_curves(rows, args.outdir, args.title)
    plot_train_vs_heldout(evals, args.outdir, args.title)
    plot_agent_vs_random_final(evals, args.outdir, args.title)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
