"""Inspect cached OnPath route masks for one complexity.

This is a diagnostic tool for clean_onpath reward shaping. It reports how many
cached nodes are mutually route-disjoint and how much of the oracle remains
reachable after each possible first hit under the old lock-on-first-hit mode.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.game_board.on_path import cache_file_path, load_complexity_cache


def _compact_coeffs(coeffs: np.ndarray, limit: int = 8) -> str:
    nz = np.flatnonzero(coeffs)
    terms = [f"{int(i)}:{int(coeffs[i])}" for i in nz[:limit]]
    if nz.size > limit:
        terms.append("...")
    return "{" + ", ".join(terms) + "}"


def _route_sizes(masks: np.ndarray) -> list[int]:
    sizes = []
    for route_idx in range(32):
        bit = np.uint32(1 << route_idx)
        size = int(np.sum((masks & bit) != 0))
        if size:
            sizes.append(size)
    return sizes


def _pairwise_disjoint_count(masks: np.ndarray) -> tuple[int, int]:
    total = 0
    disjoint = 0
    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            total += 1
            if int(masks[i] & masks[j]) == 0:
                disjoint += 1
    return disjoint, total


def _post_first_hit_reachability(masks: np.ndarray) -> list[int]:
    counts = []
    for i, mask in enumerate(masks):
        compatible = (masks & mask) != 0
        # Exclude the first-hit node itself because duplicate hits are ignored.
        counts.append(int(np.sum(compatible) - 1))
    return counts


def inspect(cache_dir: Path, complexity: int, limit: int) -> None:
    comp = load_complexity_cache(cache_file_path(cache_dir, complexity))
    lengths = np.diff(comp.on_path_offsets)
    disjoint_counts = []
    disjoint_totals = []
    reachability_values = []

    print(f"Cache: {cache_dir}")
    print(f"Complexity: C{complexity}")
    print(f"Targets: {len(comp.target_ids)}")
    print(f"Metadata route_cap_hit_rate: {comp.metadata.get('route_cap_hit_rate', 0.0):.2%}")
    print(
        "Metadata route_working_cap_hit_node_rate: "
        f"{comp.metadata.get('route_working_cap_hit_node_rate', 0.0):.2%}"
    )
    print(
        "OnPath sizes min/median/max: "
        f"{int(lengths.min())}/{float(np.median(lengths)):.1f}/{int(lengths.max())}"
    )

    for target_pos, target_id in enumerate(comp.target_ids):
        start = int(comp.on_path_offsets[target_pos])
        end = int(comp.on_path_offsets[target_pos + 1])
        ids = comp.on_path_flat_ids[start:end]
        masks = comp.on_path_route_masks[start:end]
        steps = comp.node_steps[ids]
        disjoint, total_pairs = _pairwise_disjoint_count(masks)
        reachability = _post_first_hit_reachability(masks)
        disjoint_counts.append(disjoint)
        disjoint_totals.append(total_pairs)
        reachability_values.extend(reachability)

        if target_pos >= limit:
            continue

        target_mask = 0
        target_matches = np.nonzero(ids == int(target_id))[0]
        if target_matches.size:
            target_mask = int(masks[int(target_matches[0])])
        print()
        print(
            f"target_id={int(target_id)} "
            f"target_seq_step={int(comp.node_steps[int(target_id)])} "
            f"on_path_total={len(ids)} "
            f"target_mask=0x{target_mask:08x}"
        )
        print(
            f"  route_sizes={_route_sizes(masks)} "
            f"pairwise_disjoint={disjoint}/{total_pairs} "
            f"post_first_hit_reachable={reachability}"
        )
        for node_id, step, mask in zip(ids, steps, masks):
            print(
                f"  node_id={int(node_id)} step={int(step)} "
                f"mask=0x{int(mask):08x} "
                f"coeffs={_compact_coeffs(comp.node_coeffs[int(node_id)])}"
            )

    print()
    total_disjoint = int(np.sum(disjoint_counts))
    total_pairs = int(np.sum(disjoint_totals))
    print(f"Summary pairwise_disjoint={total_disjoint}/{total_pairs}")
    if reachability_values:
        reach = np.array(reachability_values, dtype=np.int32)
        print(
            "Summary post_first_hit_reachable min/median/max: "
            f"{int(reach.min())}/{float(np.median(reach)):.1f}/{int(reach.max())}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--complexity", type=int, required=True)
    parser.add_argument("--limit", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inspect(args.cache_dir, args.complexity, args.limit)


if __name__ == "__main__":
    main()
