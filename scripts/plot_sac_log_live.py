#!/usr/bin/env python3
"""Live plots metrics from a SAC training log."""

import argparse
import math
import os
import re
import time
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

ITER_RE = re.compile(
    r"^Iter\s+(\d+):\s+SR\s+([0-9.]+)%\s*,\s*Q\s+([-0-9.]+),\s*Pi\s+([-0-9.]+),\s*CE\s+([-0-9.]+)"
)
REWARD_RE = re.compile(r"\(R:\s*([-0-9.]+),")
LOSS_INLINE_RE = re.compile(r"\bLoss\b\s*[:=]?\s*([-0-9.]+)")
LOSS_ANY_RE = re.compile(r"\bloss\b\s*[:=]?\s*([-0-9.]+)", re.IGNORECASE)


def parse_log(path: str) -> Tuple[List[int], Dict[str, List[Optional[float]]]]:
    data: Dict[int, Dict[str, Optional[float]]] = {}
    current_iter: Optional[int] = None
    rewards: List[float] = []

    try:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                m = ITER_RE.match(line)
                if m:
                    if current_iter is not None:
                        if rewards:
                            data[current_iter]["reward"] = sum(rewards) / len(rewards)
                        else:
                            data[current_iter]["reward"] = None
                    current_iter = int(m.group(1))
                    data[current_iter] = {
                        "success_rate": float(m.group(2)),
                        "q": float(m.group(3)),
                        "pi": float(m.group(4)),
                        "ce": float(m.group(5)),
                        "reward": None,
                        "loss": None,
                    }
                    rewards = []
                    loss_m = LOSS_INLINE_RE.search(line)
                    if loss_m:
                        data[current_iter]["loss"] = float(loss_m.group(1))
                    continue

                if current_iter is None:
                    continue

                r_m = REWARD_RE.search(line)
                if r_m:
                    rewards.append(float(r_m.group(1)))

                if data[current_iter].get("loss") is None:
                    loss_m = LOSS_ANY_RE.search(line)
                    if loss_m:
                        data[current_iter]["loss"] = float(loss_m.group(1))

        if current_iter is not None:
            if rewards:
                data[current_iter]["reward"] = sum(rewards) / len(rewards)
            else:
                data[current_iter]["reward"] = None
    except FileNotFoundError:
        return [], {}

    iters = sorted(data.keys())
    series: Dict[str, List[Optional[float]]] = {
        "success_rate": [],
        "q": [],
        "pi": [],
        "ce": [],
        "reward": [],
        "loss": [],
    }
    for it in iters:
        for key in series:
            series[key].append(data[it].get(key))
    return iters, series


def _clip_window(iters: List[int], series: Dict[str, List[Optional[float]]], window: int):
    if window <= 0 or len(iters) <= window:
        return iters, series
    iters = iters[-window:]
    clipped: Dict[str, List[Optional[float]]] = {}
    for key, values in series.items():
        clipped[key] = values[-window:]
    return iters, clipped


def _plot_series(ax, x, y, label, color):
    ax.clear()
    ax.plot(x, y, color=color, linewidth=2)
    ax.set_title(label)
    ax.grid(True, alpha=0.3)


def main():
    parser = argparse.ArgumentParser(description="Live plot SAC training metrics from a log file.")
    parser.add_argument("--log", default="sac.log", help="Path to sac log file")
    parser.add_argument("--interval", type=int, default=2000, help="Refresh interval (ms)")
    parser.add_argument("--window", type=int, default=200, help="Number of recent iterations to show")
    parser.add_argument("--save", action="store_true", help="Save a PNG instead of showing a GUI")
    parser.add_argument("--out", default="sac_live.png", help="PNG output path when using --save")
    parser.add_argument(
        "--metric",
        default=None,
        choices=["success_rate", "q", "pi", "ce", "reward", "loss"],
        help="Plot a single metric (requires --single)",
    )
    parser.add_argument(
        "--single",
        action="store_true",
        help="Plot only one metric in a single panel",
    )
    args = parser.parse_args()

    headless = args.save or not os.environ.get("DISPLAY")

    metric_defs = [
        ("success_rate", "Success Rate (%)", "#1f77b4"),
        ("q", "Q", "#ff7f0e"),
        ("pi", "Pi", "#2ca02c"),
        ("ce", "CE", "#d62728"),
        ("reward", "Avg Reward", "#9467bd"),
        ("loss", "Loss", "#8c564b"),
    ]
    metric_map = {key: (key, title, color) for key, title, color in metric_defs}

    if args.single:
        metric_key = args.metric or "success_rate"
        metric_defs = [metric_map[metric_key]]
        fig, axes = plt.subplots(1, 1, figsize=(8, 4))
        axes = [axes]
    else:
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        axes = axes.flatten()

    last_size = -1
    last_mtime = 0.0

    def update(_):
        nonlocal last_size, last_mtime
        try:
            stat = os.stat(args.log)
        except FileNotFoundError:
            fig.suptitle(f"Waiting for log: {args.log}")
            return
        if stat.st_size == last_size and stat.st_mtime == last_mtime:
            return
        last_size = stat.st_size
        last_mtime = stat.st_mtime

        iters, series = parse_log(args.log)
        if not iters:
            fig.suptitle(f"No data yet in {args.log}")
            return

        iters, series = _clip_window(iters, series, args.window)

        for ax, (key, title, color) in zip(axes, metric_defs):
            y = series.get(key, [])
            if not y or all(v is None or (isinstance(v, float) and math.isnan(v)) for v in y):
                ax.clear()
                ax.set_title(f"{title} (no data)")
                ax.grid(True, alpha=0.3)
                continue
            _plot_series(ax, iters, y, title, color)
        fig.suptitle(f"SAC metrics: {args.log}")
        fig.tight_layout()

    anim = FuncAnimation(fig, update, interval=args.interval, cache_frame_data=False)
    update(0)
    fig.canvas.draw()
    if headless:
        fig.savefig(args.out, dpi=150)
    else:
        plt.show()


if __name__ == "__main__":
    main()
