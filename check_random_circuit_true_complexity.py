#!/usr/bin/env python3
"""Check true sequential complexity of random-circuit-labeled targets.

The training sampler labels targets by the number of random generation actions.
This script estimates the minimum sequential route size by enumerating coherent
construction routes up to a cap, then reports how often random "Ck" targets are
actually lower-complexity.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Set

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Config  # noqa: E402
from src.environment.fast_polynomial import FastPoly  # noqa: E402
from src.game_board.generator import generate_random_circuit  # noqa: E402

Route = frozenset[bytes]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Histogram true route complexity for random generated targets."
    )
    parser.add_argument("--labels", type=int, nargs="+", default=[5, 6, 7, 8])
    parser.add_argument("--samples-per-label", type=int, default=1000)
    parser.add_argument("--max-true-complexity", type=int, default=8)
    parser.add_argument("--n-variables", type=int, default=2)
    parser.add_argument("--mod", type=int, default=5)
    parser.add_argument("--max-degree", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--route-cap", type=int, default=512)
    parser.add_argument(
        "--max-polys",
        type=int,
        default=300000,
        help="Safety cap on indexed polynomials before stopping enumeration.",
    )
    parser.add_argument(
        "--out-prefix",
        type=str,
        default="results/random_circuit_true_complexity",
        help="Output prefix for .csv and .json files.",
    )
    return parser.parse_args()


def make_config(args: argparse.Namespace) -> Config:
    return Config(
        n_variables=args.n_variables,
        mod=args.mod,
        max_complexity=max(max(args.labels), args.max_true_complexity),
        max_degree=args.max_degree,
    )


def base_keys(config: Config) -> Set[bytes]:
    keys = set()
    for i in range(config.n_variables):
        keys.add(
            FastPoly.variable(
                i, config.n_variables, config.effective_max_degree, config.mod
            ).canonical_key()
        )
    keys.add(
        FastPoly.constant(
            1, config.n_variables, config.effective_max_degree, config.mod
        ).canonical_key()
    )
    return keys


def initial_polys(config: Config) -> List[FastPoly]:
    polys = []
    for i in range(config.n_variables):
        polys.append(
            FastPoly.variable(
                i, config.n_variables, config.effective_max_degree, config.mod
            )
        )
    polys.append(
        FastPoly.constant(
            1, config.n_variables, config.effective_max_degree, config.mod
        )
    )
    return polys


def add_route(
    routes_by_key: Dict[bytes, Set[Route]],
    key: bytes,
    route: Route,
    route_cap: int,
) -> bool:
    """Add a route while keeping only a subset-minimal antichain."""
    routes = routes_by_key.setdefault(key, set())
    for existing in routes:
        if existing.issubset(route):
            return False

    dominated = [existing for existing in routes if route.issubset(existing)]
    for existing in dominated:
        routes.remove(existing)

    routes.add(route)
    if len(routes) > route_cap:
        sorted_routes = sorted(routes, key=lambda r: (len(r), sorted(r)[:1]))
        routes.clear()
        routes.update(sorted_routes[:route_cap])
    return True


def build_route_index(
    config: Config,
    max_cost: int,
    route_cap: int,
    max_polys: int,
) -> tuple[Dict[bytes, int], Dict[str, int]]:
    """Enumerate coherent route sets up to max_cost.

    Cost is number of unique non-base nodes in a route. Parent routes are
    unioned, then the child is added if it is non-base.
    """
    bases = base_keys(config)
    poly_by_key: Dict[bytes, FastPoly] = {}
    routes_by_key: Dict[bytes, Set[Route]] = {}

    for poly in initial_polys(config):
        key = poly.canonical_key()
        poly_by_key[key] = poly
        routes_by_key[key] = {frozenset()}

    stats = {
        "route_cap_trims": 0,
        "stopped_by_max_polys": 0,
    }

    for cost in range(1, max_cost + 1):
        t0 = time.time()
        additions = 0
        keys = list(poly_by_key.keys())
        eligible_routes = {
            key: [r for r in routes_by_key.get(key, ()) if len(r) <= cost - 1]
            for key in keys
        }

        for left_pos, left_key in enumerate(keys):
            left_routes = eligible_routes[left_key]
            if not left_routes:
                continue
            left_poly = poly_by_key[left_key]

            for right_key in keys[left_pos:]:
                right_routes = eligible_routes[right_key]
                if not right_routes:
                    continue
                right_poly = poly_by_key[right_key]

                for op_name, child in (
                    ("add", left_poly + right_poly),
                    ("mul", left_poly * right_poly),
                ):
                    del op_name  # kept only to make the op tuple readable
                    child_key = child.canonical_key()
                    child_is_base = child_key in bases
                    child_node = frozenset() if child_is_base else frozenset([child_key])

                    for left_route in left_routes:
                        for right_route in right_routes:
                            route = left_route | right_route | child_node
                            if len(route) != cost:
                                continue

                            if child_key not in poly_by_key:
                                if len(poly_by_key) >= max_polys:
                                    stats["stopped_by_max_polys"] = 1
                                    print(
                                        f"Stopped at max_polys={max_polys} "
                                        f"during cost {cost}.",
                                        flush=True,
                                    )
                                    return route_min_cost(routes_by_key), stats
                                poly_by_key[child_key] = child

                            before = len(routes_by_key.get(child_key, ()))
                            changed = add_route(
                                routes_by_key, child_key, route, route_cap
                            )
                            after = len(routes_by_key.get(child_key, ()))
                            if before >= route_cap and after == route_cap and changed:
                                stats["route_cap_trims"] += 1
                            if changed:
                                additions += 1

        print(
            f"route_cost={cost}: polys={len(poly_by_key)} "
            f"route_additions={additions} elapsed={time.time() - t0:.1f}s",
            flush=True,
        )

    return route_min_cost(routes_by_key), stats


def route_min_cost(routes_by_key: Dict[bytes, Set[Route]]) -> Dict[bytes, int]:
    costs = {}
    for key, routes in routes_by_key.items():
        if routes:
            costs[key] = min(len(route) for route in routes)
    return costs


def sample_targets(
    config: Config,
    labels: Iterable[int],
    samples_per_label: int,
) -> dict[int, dict[str, object]]:
    by_label = {}
    for label in labels:
        unique: Dict[bytes, FastPoly] = {}
        duplicate_count = 0
        for _ in range(samples_per_label):
            poly, _actions = generate_random_circuit(config, label)
            key = poly.canonical_key()
            if key in unique:
                duplicate_count += 1
            else:
                unique[key] = poly
        by_label[label] = {
            "unique": unique,
            "duplicates": duplicate_count,
        }
        print(
            f"sampled label C{label}: total={samples_per_label} "
            f"unique={len(unique)} duplicates={duplicate_count}",
            flush=True,
        )
    return by_label


def summarize(
    sampled: dict[int, dict[str, object]],
    min_cost_by_key: Dict[bytes, int],
    max_true_complexity: int,
) -> tuple[List[dict], dict]:
    rows = []
    summary = {}

    for label, payload in sampled.items():
        unique: Dict[bytes, FastPoly] = payload["unique"]  # type: ignore[assignment]
        hist = Counter()
        unknown = 0

        for key in unique:
            cost = min_cost_by_key.get(key)
            if cost is None or cost > max_true_complexity:
                unknown += 1
            else:
                hist[cost] += 1

        total_unique = len(unique)
        row = {
            "label": label,
            "sampled": total_unique + int(payload["duplicates"]),
            "unique": total_unique,
            "duplicates": int(payload["duplicates"]),
            "unknown_gt_max": unknown,
            "true_C0": hist[0],
        }
        for cost in range(1, max_true_complexity + 1):
            row[f"true_C{cost}"] = hist[cost]
        rows.append(row)

        summary[label] = {
            "hist": dict(hist),
            "unknown_gt_max": unknown,
            "unique": total_unique,
            "duplicates": int(payload["duplicates"]),
        }

    return rows, summary


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    config = make_config(args)
    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    print(
        "Random-circuit true-complexity check "
        f"labels={args.labels} samples_per_label={args.samples_per_label} "
        f"max_true_complexity={args.max_true_complexity} "
        f"n_vars={args.n_variables} mod={args.mod} max_degree={args.max_degree}",
        flush=True,
    )

    sampled = sample_targets(config, args.labels, args.samples_per_label)
    min_cost_by_key, index_stats = build_route_index(
        config=config,
        max_cost=args.max_true_complexity,
        route_cap=args.route_cap,
        max_polys=args.max_polys,
    )
    rows, summary = summarize(sampled, min_cost_by_key, args.max_true_complexity)

    csv_path = out_prefix.with_suffix(".csv")
    json_path = out_prefix.with_suffix(".json")

    fieldnames = [
        "label",
        "sampled",
        "unique",
        "duplicates",
        "true_C0",
        *[f"true_C{c}" for c in range(1, args.max_true_complexity + 1)],
        "unknown_gt_max",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "args": vars(args),
                "index_stats": index_stats,
                "summary": summary,
            },
            f,
            indent=2,
            sort_keys=True,
        )

    print("\nHistogram by generated label:")
    for row in rows:
        parts = [
            f"generated_C{row['label']}",
            f"unique={row['unique']}",
            f"dupes={row['duplicates']}",
            f"true_C0={row['true_C0']}",
        ]
        parts.extend(
            f"true_C{c}={row[f'true_C{c}']}"
            for c in range(1, args.max_true_complexity + 1)
        )
        parts.append(f"unknown>{args.max_true_complexity}={row['unknown_gt_max']}")
        print("  " + " ".join(parts), flush=True)

    print(f"\nSaved {csv_path}", flush=True)
    print(f"Saved {json_path}", flush=True)


if __name__ == "__main__":
    main()
