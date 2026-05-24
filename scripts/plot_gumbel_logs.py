#!/usr/bin/env python3
"""Render publication-style diagnostics from the Gumbel run stdout logs.

The Gumbel curriculum run in ``/Users/rohanpandey/Downloads/gumbel outputs``
emits plain-text logs with iteration-level metrics. This script parses those logs
and produces one 2x2 training-diagnostics figure per phase, matching the style
of the reference plot: mean reward, policy entropy, policy loss, and value loss.

By default the figures are written into the ICML Submission folder so they can
be dropped straight into the manuscript assets.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


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

ACCENT = "#1f6feb"
ENTROPY = "#2a9d8f"
POLICY_LOSS = "#e76f51"
VALUE_LOSS = "#9b5de5"
PHASE_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]

ITER_RE = re.compile(
    r"\[PPO\+MCTS-JAX iter (?P<iter>\d+)\].*?"
    r"success=(?P<success>[0-9.]+)%.*?"
    r"reward=(?P<reward>[-0-9.]+)\s+"
    r"pg_loss=(?P<pg_loss>[-0-9.]+)\s+"
    r"vf_loss=(?P<vf_loss>[-0-9.]+)\s+"
    r"entropy=(?P<entropy>[-0-9.]+)"
)
RESULTS_RE = re.compile(
    r"Results will be saved to .*?/phase(?P<phase>\d+)_C(?P<low>\d+)(?:_(?P<high>\d+))?"
)
FIXED_RE = re.compile(r"fixed_complexities=\[(?P<vals>[0-9,\s]+)\]")
CHECKPOINT_RE = re.compile(r"Loaded PPO\+MCTS JAX checkpoint from .*?/phase(?P<phase>\d+)_C")


@dataclass
class PhaseLog:
    path: Path
    phase: str
    complexities: str
    rows: list[dict[str, float]]


def rolling_mean(values: list[float], window: int = 9) -> list[float]:
    if len(values) < 2 or window <= 1:
        return values
    half = window // 2
    out: list[float] = []
    for idx in range(len(values)):
        lo = max(0, idx - half)
        hi = min(len(values), idx + half + 1)
        chunk = values[lo:hi]
        out.append(sum(chunk) / len(chunk))
    return out


def parse_log(path: Path) -> PhaseLog | None:
    phase = None
    complexities = None
    rows: list[dict[str, float]] = []

    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = RESULTS_RE.search(line)
        if match:
            phase = match.group("phase")
            if match.group("high") is None:
                complexities = f"C{match.group('low')}"
            else:
                complexities = f"C{match.group('low')}, C{match.group('high')}"
        elif phase is None:
            match = CHECKPOINT_RE.search(line)
            if match:
                phase = match.group("phase")

        if complexities is None:
            match = FIXED_RE.search(line)
            if match:
                vals = [value.strip() for value in match.group("vals").split(",") if value.strip()]
                complexities = ", ".join(f"C{value}" for value in vals)

        match = ITER_RE.search(line)
        if match:
            rows.append({
                "iteration": float(match.group("iter")),
                "success": float(match.group("success")),
                "reward": float(match.group("reward")),
                "pg_loss": float(match.group("pg_loss")),
                "vf_loss": float(match.group("vf_loss")),
                "entropy": float(match.group("entropy")),
            })

    if not rows:
        return None

    if phase is None:
        phase = path.stem
    if complexities is None:
        complexities = "unknown complexities"

    return PhaseLog(path=path, phase=phase, complexities=complexities, rows=rows)


def discover_logs(log_dir: Path) -> list[Path]:
    return sorted(log_dir.glob("*.log"))


def phase_sort_key(phase_log: PhaseLog) -> tuple[int, str]:
    try:
        return (int(phase_log.phase), phase_log.path.name)
    except ValueError:
        return (10_000, phase_log.path.name)


def plot_phase(phase_log: PhaseLog, outdir: Path) -> Path:
    iterations = [row["iteration"] for row in phase_log.rows]
    metrics = [
        ("reward", "mean episode reward", ACCENT, True),
        ("entropy", "policy entropy", ENTROPY, False),
        ("pg_loss", "policy loss", POLICY_LOSS, True),
        ("vf_loss", "value loss", VALUE_LOSS, False),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.5))
    fig.patch.set_facecolor("white")

    for axis, (key, label, color, smooth) in zip(axes.flat, metrics):
        values = [row[key] for row in phase_log.rows]
        axis.plot(iterations, values, color=color, alpha=0.22, linewidth=1.0)
        axis.plot(iterations, rolling_mean(values), color=color, linewidth=2.4)
        axis.set_xlabel("PPO iteration")
        axis.set_ylabel(label)

    title = f"Training diagnostics — PPO+MCTS JAX Gumbel, phase {phase_log.phase} ({phase_log.complexities})"
    fig.suptitle(title, fontsize=16, fontweight="bold")

    output = outdir / f"gumbel_phase_{phase_log.phase}_diagnostics.png"
    fig.savefig(output, dpi=300)
    plt.close(fig)
    return output


def metric_specs() -> list[tuple[str, str]]:
    return [
        ("reward", "mean episode reward"),
        ("entropy", "policy entropy"),
        ("pg_loss", "policy loss"),
        ("vf_loss", "value loss"),
    ]


def phase_label(phase_log: PhaseLog) -> str:
    return f"P{int(phase_log.phase):02d} ({phase_log.complexities})"


def short_phase_label(phase_log: PhaseLog) -> str:
    return f"P{int(phase_log.phase):02d}\n{phase_log.complexities}"


def plot_staggered_phases(phase_logs: list[PhaseLog], outdir: Path) -> Path:
    phase_gap = 45.0
    phase_padding = 18.0
    phase_width = max(max(row["iteration"] for row in phase_log.rows) for phase_log in phase_logs)
    widths = [phase_width for _ in phase_logs]
    starts: list[float] = []
    cursor = 0.0
    for width in widths:
        starts.append(cursor)
        cursor += width + phase_gap

    fig, axes = plt.subplots(2, 2, figsize=(15.5, 9.0))
    fig.patch.set_facecolor("white")

    for axis, (key, label) in zip(axes.flat, metric_specs()):
        for idx, phase_log in enumerate(phase_logs):
            color = PHASE_COLORS[idx % len(PHASE_COLORS)]
            start = starts[idx]
            width = widths[idx]
            band_start = start - phase_padding
            band_end = start + width + phase_padding

            axis.axvspan(band_start, band_end, color=color, alpha=0.035, linewidth=0)
            if idx > 0:
                axis.axvspan(start - phase_gap + phase_padding, start - phase_padding, color="#333333", alpha=0.035, linewidth=0)
                axis.axvline(start - phase_padding, color="#999999", alpha=0.35, linewidth=0.8)

            iterations = [start + row["iteration"] for row in phase_log.rows]
            values = [row[key] for row in phase_log.rows]
            smooth_values = rolling_mean(values)
            axis.plot(iterations, values, color=color, alpha=0.16, linewidth=1.0)
            axis.plot(iterations, smooth_values, color=color, linewidth=2.7)
            axis.scatter(iterations[-1], smooth_values[-1], color=color, s=22, zorder=4)

        y_top = axis.get_ylim()[1]
        for idx, phase_log in enumerate(phase_logs):
            color = PHASE_COLORS[idx % len(PHASE_COLORS)]
            start = starts[idx]
            width = widths[idx]
            axis.text(
                start + width / 2,
                y_top,
                short_phase_label(phase_log),
                ha="center",
                va="bottom",
                color=color,
                fontsize=9,
                fontweight="bold",
                clip_on=False,
            )

        tick_positions = [start + width / 2 for start, width in zip(starts, widths)]
        tick_labels = [f"P{int(phase_log.phase):02d}" for phase_log in phase_logs]
        axis.set_xticks(tick_positions, tick_labels)
        axis.set_xlabel("curriculum phase (each band spans 0-500 PPO iterations)")
        axis.set_ylabel(label)
        axis.margins(x=0.015)

    fig.suptitle("Gumbel PPO+MCTS JAX diagnostics across curriculum phases", fontsize=18, fontweight="bold")
    output = outdir / "gumbel_all_phases_staggered_diagnostics.png"
    fig.savefig(output, dpi=300)
    plt.close(fig)
    return output


def plot_overlaid_phases(phase_logs: list[PhaseLog], outdir: Path) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.5))
    fig.set_constrained_layout(False)
    fig.patch.set_facecolor("white")

    legend_handles = []
    legend_labels = []

    for axis, (key, label) in zip(axes.flat, metric_specs()):
        for idx, phase_log in enumerate(phase_logs):
            color = PHASE_COLORS[idx % len(PHASE_COLORS)]
            iterations = [row["iteration"] for row in phase_log.rows]
            values = [row[key] for row in phase_log.rows]
            axis.plot(iterations, values, color=color, alpha=0.12, linewidth=1.0)
            line, = axis.plot(iterations, rolling_mean(values), color=color, linewidth=2.2)

            if len(legend_handles) < len(phase_logs):
                legend_handles.append(line)
                legend_labels.append(phase_log.complexities)

        axis.set_xlabel("PPO iteration within phase")
        axis.set_ylabel(label)

    fig.suptitle("Gumbel PPO+MCTS JAX diagnostics", fontsize=18, fontweight="bold")
    fig.tight_layout(rect=(0.0, 0.16, 1.0, 0.94))
    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        ncol=5,
        bbox_to_anchor=(0.5, 0.015),
        fontsize=10,
    )
    output = outdir / "gumbel_all_phases_overlaid_diagnostics.png"
    fig.savefig(output, dpi=300)
    plt.close(fig)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("/Users/rohanpandey/Downloads/gumbel outputs"),
        help="Directory containing the Gumbel stdout logs.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("/Users/rohanpandey/Desktop/Research_Projects/Math AI Lab/VSCode/PolyArithmeticCircuitsRL/ICML Submission"),
        help="Directory where the rendered figures should be written.",
    )
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    phase_logs = []
    for log_path in discover_logs(args.log_dir):
        parsed = parse_log(log_path)
        if parsed is not None:
            phase_logs.append(parsed)

    if not phase_logs:
        raise SystemExit(f"No training rows found in {args.log_dir}")

    phase_logs.sort(key=phase_sort_key)
    print(f"Parsed {len(phase_logs)} logs from {args.log_dir}")

    for phase_log in phase_logs:
        output = plot_phase(phase_log, args.outdir)
        print(f"wrote {output}")

    for output in (
        plot_staggered_phases(phase_logs, args.outdir),
        plot_overlaid_phases(phase_logs, args.outdir),
    ):
        print(f"wrote {output}")


if __name__ == "__main__":
    main()
