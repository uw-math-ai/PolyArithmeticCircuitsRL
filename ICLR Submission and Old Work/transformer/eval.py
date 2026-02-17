"""
Evaluate a trained transformer on a held-out set.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
import sympy
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

try:
    from transformer.polynomial_to_circuit import (
        CharTokenizer,
        Seq2SeqTransformer,
        build_dataset,
        generate_square_subsequent_mask,
        load_checkpoint,
        greedy_decode,
    )
    from transformer.build_training_data import generate_board
except ModuleNotFoundError:
    from polynomial_to_circuit import (
        CharTokenizer,
        Seq2SeqTransformer,
        build_dataset,
        generate_square_subsequent_mask,
        load_checkpoint,
        greedy_decode,
    )
    from build_training_data import generate_board


class PolyCircuitDataset(Dataset):
    def __init__(self, examples, tokenizer: CharTokenizer):
        self.examples = examples
        self.tokenizer = tokenizer
        self.src = [tokenizer.encode(ex.polynomial, add_bos=True, add_eos=True) for ex in examples]
        self.tgt = [tokenizer.encode(ex.circuit, add_bos=True, add_eos=True) for ex in examples]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        return self.src[idx], self.tgt[idx]


def _collate(batch, pad_id: int):
    src_seqs, tgt_seqs = zip(*batch)
    src_len = max(len(s) for s in src_seqs)
    tgt_len = max(len(t) for t in tgt_seqs)

    src = torch.full((src_len, len(batch)), pad_id, dtype=torch.long)
    tgt_in = torch.full((tgt_len - 1, len(batch)), pad_id, dtype=torch.long)
    tgt_out = torch.full((tgt_len - 1, len(batch)), pad_id, dtype=torch.long)

    for i, (s, t) in enumerate(zip(src_seqs, tgt_seqs)):
        src[: len(s), i] = torch.tensor(s, dtype=torch.long)
        tgt_tensor = torch.tensor(t, dtype=torch.long)
        tgt_in[: len(t) - 1, i] = tgt_tensor[:-1]
        tgt_out[: len(t) - 1, i] = tgt_tensor[1:]

    src_key_padding_mask = (src == pad_id).transpose(0, 1)
    tgt_key_padding_mask = (tgt_in == pad_id).transpose(0, 1)
    return src, tgt_in, tgt_out, src_key_padding_mask, tgt_key_padding_mask


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate polynomial->circuit transformer")
    parser.add_argument("--nodes", type=Path, default=None, help="Nodes JSONL path")
    parser.add_argument("--edges", type=Path, default=None, help="Edges JSONL path")
    parser.add_argument("--analysis", type=Path, default=None, help="Analysis JSONL path")
    parser.add_argument("--board-dir", type=Path, default=None, help="Directory containing board JSONL files")
    parser.add_argument("--prefix", type=str, default=None, help="Board prefix like game_board_C4")
    parser.add_argument(
        "--auto-generate-board",
        action="store_true",
        help="Generate a game board if inputs are missing",
    )
    parser.add_argument("--steps", type=int, default=None, help="Complexity steps for auto-generation")
    parser.add_argument("--num-vars", type=int, default=1, help="Number of variables for auto-generation")
    parser.add_argument(
        "--no-constant",
        action="store_true",
        help="Disable seeding the constant 1 node when auto-generating a board",
    )
    parser.add_argument(
        "--board-out-dir",
        type=Path,
        default=Path("transformer/boards"),
        help="Output directory for auto-generated boards",
    )
    parser.add_argument("--max-complexity", type=int, default=None, help="Max shortest path length")
    parser.add_argument("--include-non-multipath", action="store_true")
    parser.add_argument("--allow-all", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--expand-circuit",
        action="store_true",
        help="Evaluate against expanded polynomial targets instead of circuit expressions",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--unseen-samples", type=int, default=0, help="Number of unseen polynomials to test")
    parser.add_argument("--unseen-steps", type=int, default=None, help="Random poly steps for unseen eval")
    parser.add_argument(
<<<<<<< HEAD
        "--episodes",
        type=int,
        default=None,
        help="Alias for unseen evaluation sample count (PPO-style naming).",
    )
    parser.add_argument(
        "--steps-per-episode",
        type=int,
        default=None,
        help="Alias for unseen polynomial generation steps (PPO-style naming).",
    )
    parser.add_argument(
=======
>>>>>>> 11b48741e682c6fc7ea309bcbc3750e60bf7594b
        "--unseen-max-coeff",
        type=int,
        default=5,
        help="Max absolute coefficient for random unseen polynomials",
    )
    parser.add_argument("--unseen-seed", type=int, default=7, help="RNG seed for unseen eval")
    parser.add_argument(
        "--unseen-allow-inboard",
        action="store_true",
        help="Allow unseen samples that appear in the game board",
    )
    return parser.parse_args()


def _sympify(expr: str) -> sympy.Expr | None:
    try:
        return sympy.sympify(expr.replace("^", "**"))
    except Exception:
        return None


def _equivalent(expr_a: str, expr_b: str) -> bool:
    a = _sympify(expr_a)
    b = _sympify(expr_b)
    if a is None or b is None:
        return False
    try:
        return sympy.simplify(a - b) == 0
    except Exception:
        return False


def _random_polynomial(num_vars: int, steps: int, max_coeff: int, rng: random.Random) -> str:
    symbols = sympy.symbols(f"x0:{num_vars}") if num_vars > 1 else (sympy.symbols("x"),)
    pool: list[sympy.Expr] = list(symbols)

    if max_coeff > 0:
        for coeff in range(1, max_coeff + 1):
            pool.append(sympy.Integer(coeff))
            pool.append(sympy.Integer(-coeff))

    for _ in range(max(1, steps)):
        left = rng.choice(pool)
        right = rng.choice(pool)
        if rng.random() < 0.5:
            expr = sympy.expand(left + right)
        else:
            expr = sympy.expand(left * right)
        pool.append(expr)

    return str(rng.choice(pool)).replace("**", "^")

def _progress(iterable=None, **kwargs):
    if iterable is None:
        return tqdm(disable=not sys.stderr.isatty(), **kwargs)
    return tqdm(iterable, disable=not sys.stderr.isatty(), **kwargs)


def main() -> None:
    args = _parse_args()
    nodes_path = args.nodes
    edges_path = args.edges
    analysis_path = args.analysis
    if args.board_dir and args.prefix:
        nodes_path = nodes_path or (args.board_dir / f"{args.prefix}.nodes.jsonl")
        edges_path = edges_path or (args.board_dir / f"{args.prefix}.edges.jsonl")
        analysis_path = analysis_path or (args.board_dir / f"{args.prefix}.analysis.jsonl")

    if args.auto_generate_board and (nodes_path is None or edges_path is None):
        if args.steps is None:
            raise SystemExit("Auto-generate requested but --steps is missing.")
        out_dir = args.board_dir or args.board_out_dir
        prefix = args.prefix or f"game_board_C{args.steps}_V{args.num_vars}"
        nodes_path, edges_path, analysis_path = generate_board(
            steps=args.steps,
            num_vars=args.num_vars,
            output_dir=out_dir,
            prefix=prefix,
            max_nodes=None,
            max_successors_per_node=None,
            max_samples=5,
            only_multipath=not args.include_non_multipath,
            analysis_max_step=args.max_complexity,
            skip_plot=True,
            include_constant=not args.no_constant,
        )

    if nodes_path is None or edges_path is None:
        raise SystemExit("Must provide --nodes and --edges or --board-dir with --prefix")

    if analysis_path is None and not args.allow_all:
        raise SystemExit("Analysis JSONL missing. Provide --analysis or use --allow-all.")

    only_multipath = not args.include_non_multipath
    if analysis_path is None:
        only_multipath = False

    examples = build_dataset(
        nodes_path=nodes_path,
        edges_path=edges_path,
        analysis_path=analysis_path,
        max_complexity=args.max_complexity,
        only_multipath=only_multipath,
        expand_circuit=args.expand_circuit,
    )
    if args.limit is not None:
        examples = examples[: args.limit]

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, tokenizer = load_checkpoint(args.checkpoint, device)
    dataset = PolyCircuitDataset(examples, tokenizer)
    pad_id = tokenizer.vocab["<pad>"]
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: _collate(batch, pad_id),
    )

    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id)
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for src, tgt_in, tgt_out, src_mask, tgt_mask in _progress(
            loader,
            desc="Eval loss",
            unit="batch",
            leave=False,
        ):
            src = src.to(device)
            tgt_in = tgt_in.to(device)
            tgt_out = tgt_out.to(device)
            src_mask = src_mask.to(device)
            tgt_mask = tgt_mask.to(device)
            causal_mask = generate_square_subsequent_mask(tgt_in.size(0), device)

            logits = model(
                src,
                tgt_in,
                tgt_mask=causal_mask,
                src_key_padding_mask=src_mask,
                tgt_key_padding_mask=tgt_mask,
                memory_key_padding_mask=src_mask,
            )
            loss = loss_fn(logits.view(-1, logits.size(-1)), tgt_out.reshape(-1))
            tokens = (tgt_out != pad_id).sum().item()
            total_loss += float(loss.item()) * tokens
            total_tokens += tokens

    avg_loss = total_loss / max(total_tokens, 1)
    print(f"Token-level cross-entropy: {avg_loss:.6f}")

    exact_eq = 0
    decoded_total = 0
    with torch.no_grad():
        pbar = _progress(
            total=len(dataset),
            desc="Eval decode",
            unit="sample",
            leave=False,
        )
        for src, _, tgt_out, _, _ in loader:
            src = src.to(device)
            batch_size = src.size(1)
            for i in range(batch_size):
                decoded = greedy_decode(model, src[:, i : i + 1], tokenizer)
                target = tokenizer.decode(tgt_out[:, i].tolist())
                if _equivalent(decoded, target):
                    exact_eq += 1
                decoded_total += 1
                pbar.update(1)
        pbar.close()
    if decoded_total:
<<<<<<< HEAD
        val_success_pct = 100.0 * exact_eq / decoded_total
        print(
            f"Structural equivalence (val): {exact_eq}/{decoded_total} = {exact_eq / decoded_total:.3f}"
        )
        print(
            f"Val success %: {val_success_pct:.2f}% "
            f"(episodes={decoded_total}, steps/episode=N/A)"
        )

    unseen_eval_episodes = args.episodes if args.episodes is not None else args.unseen_samples
    if unseen_eval_episodes > 0:
=======
        print(f"Structural equivalence (val): {exact_eq}/{decoded_total} = {exact_eq / decoded_total:.3f}")

    if args.unseen_samples > 0:
>>>>>>> 11b48741e682c6fc7ea309bcbc3750e60bf7594b
        rng = random.Random(args.unseen_seed)
        board_keys = set()
        if nodes_path is not None and nodes_path.exists():
            with nodes_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    try:
                        record = json.loads(line)
                    except Exception:
                        continue
                    expr = record.get("expr_str") or record.get("label") or record.get("id")
                    if expr:
                        parsed = _sympify(expr.replace("^", "**"))
                        if parsed is None:
                            continue
                        canon = sympy.srepr(sympy.expand(parsed))
                        board_keys.add(canon)

<<<<<<< HEAD
        unseen_steps = (
            args.steps_per_episode
            if args.steps_per_episode is not None
            else (args.unseen_steps or args.max_complexity or 3)
        )
        unseen_hits = 0
        attempted = 0
        pbar = _progress(
            total=unseen_eval_episodes,
=======
        unseen_steps = args.unseen_steps or args.max_complexity or 3
        unseen_hits = 0
        attempted = 0
        pbar = _progress(
            total=args.unseen_samples,
>>>>>>> 11b48741e682c6fc7ea309bcbc3750e60bf7594b
            desc="Unseen",
            unit="sample",
            leave=False,
        )
<<<<<<< HEAD
        while attempted < unseen_eval_episodes:
=======
        while attempted < args.unseen_samples:
>>>>>>> 11b48741e682c6fc7ea309bcbc3750e60bf7594b
            poly = _random_polynomial(args.num_vars, unseen_steps, args.unseen_max_coeff, rng)
            expr = _sympify(poly)
            if expr is None:
                continue
            canon = sympy.srepr(sympy.expand(expr))
            if not args.unseen_allow_inboard and canon in board_keys:
                continue

            src_ids = tokenizer.encode(poly, add_bos=True, add_eos=True)
            src = torch.tensor(src_ids, dtype=torch.long, device=device).unsqueeze(1)
            decoded = greedy_decode(model, src, tokenizer)
            if _equivalent(decoded, poly):
                unseen_hits += 1
            attempted += 1
            pbar.update(1)
        pbar.close()

        print(
<<<<<<< HEAD
            f"Structural equivalence (unseen): {unseen_hits}/{unseen_eval_episodes} "
            f"= {unseen_hits / max(unseen_eval_episodes, 1):.3f}"
        )
        print(
            f"Unseen success %: {100.0 * unseen_hits / max(unseen_eval_episodes, 1):.2f}% "
            f"(episodes={unseen_eval_episodes}, steps/episode={unseen_steps})"
=======
            f"Structural equivalence (unseen): {unseen_hits}/{args.unseen_samples} "
            f"= {unseen_hits / max(args.unseen_samples, 1):.3f}"
>>>>>>> 11b48741e682c6fc7ea309bcbc3750e60bf7594b
        )


if __name__ == "__main__":
    main()
