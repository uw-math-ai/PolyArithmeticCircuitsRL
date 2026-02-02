"""
Generate consistent game-board JSONL files for training.

This wraps Game-Board-Generation/interesting_polynomial_generator.py to ensure
nodes/edges/analysis match the same variable naming scheme.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load_generator_module() -> object:
    root = Path(__file__).resolve().parents[1]
    gb_dir = root / "Game-Board-Generation"
    sys.path.insert(0, str(gb_dir))
    import interesting_polynomial_generator as ipg  # type: ignore
    return ipg


def generate_board(
    steps: int,
    num_vars: int,
    output_dir: Path,
    prefix: str | None = None,
    max_nodes: int | None = None,
    max_successors_per_node: int | None = None,
    max_samples: int = 5,
    only_multipath: bool = False,
    analysis_max_step: int | None = None,
    skip_plot: bool = False,
) -> tuple[Path, Path, Path]:
    ipg = _load_generator_module()

    prefix = prefix or f"game_board_C{steps}_V{num_vars}"
    base_path = output_dir / prefix

    graph = ipg.build_game_graph(
        steps=steps,
        num_vars=num_vars,
        max_nodes=max_nodes,
        max_successors_per_node=max_successors_per_node,
    )

    ipg.save_graph_files(graph, base_path)
    nodes_jsonl = base_path.with_suffix(".nodes.jsonl")
    edges_jsonl = base_path.with_suffix(".edges.jsonl")
    ipg.write_graph_jsonl(graph, nodes_jsonl, edges_jsonl)

    analysis_records = ipg.analyze_graph(
        graph,
        max_samples=max_samples,
        only_multipath=only_multipath,
        max_step=analysis_max_step,
    )
    analysis_path = base_path.with_suffix(".analysis.jsonl")
    analysis_path.parent.mkdir(parents=True, exist_ok=True)
    with analysis_path.open("w", encoding="utf-8") as handle:
        for record in analysis_records:
            handle.write(json.dumps(record) + "\n")

    if not skip_plot:
        png_path = base_path.with_suffix(".png")
        ipg.plot_graph(graph, png_path, with_labels=False, with_arrows=False)

    return nodes_jsonl, edges_jsonl, analysis_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate game-board JSONL data for training")
    parser.add_argument("--steps", type=int, required=True, help="Expansion steps (complexity C)")
    parser.add_argument("--num-vars", type=int, default=1, help="Number of variables")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory")
    parser.add_argument("--prefix", type=str, default=None, help="File prefix (default: game_board_C<steps>)")
    parser.add_argument("--max-nodes", type=int, default=None, help="Hard cap on number of nodes")
    parser.add_argument(
        "--max-successors-per-node",
        type=int,
        default=None,
        help="Limit expansions per node",
    )
    parser.add_argument("--max-samples", type=int, default=5, help="Max shortest-path samples per node")
    parser.add_argument("--only-multipath", action="store_true", help="Only emit multipath records")
    parser.add_argument(
        "--analysis-max-step",
        type=int,
        default=None,
        help="Only keep analysis records up to this step",
    )
    parser.add_argument("--skip-plot", action="store_true", help="Skip PNG plot")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    nodes_jsonl, edges_jsonl, analysis_path = generate_board(
        steps=args.steps,
        num_vars=args.num_vars,
        output_dir=args.output_dir,
        prefix=args.prefix,
        max_nodes=args.max_nodes,
        max_successors_per_node=args.max_successors_per_node,
        max_samples=args.max_samples,
        only_multipath=args.only_multipath,
        analysis_max_step=args.analysis_max_step,
        skip_plot=args.skip_plot,
    )

    print(f"Wrote {nodes_jsonl}")
    print(f"Wrote {edges_jsonl}")
    print(f"Wrote {analysis_path}")


if __name__ == "__main__":
    main()
