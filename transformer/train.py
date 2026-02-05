"""
Train a seq2seq Transformer to map polynomial strings to SymPy circuit strings.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import random
from pathlib import Path
from typing import List, Optional, Sequence

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
        greedy_decode,
        save_checkpoint,
    )
    from transformer.build_training_data import generate_board
except ModuleNotFoundError:
    from polynomial_to_circuit import (
        CharTokenizer,
        Seq2SeqTransformer,
        build_dataset,
        generate_square_subsequent_mask,
        greedy_decode,
        save_checkpoint,
    )
    from build_training_data import generate_board


def _analysis_matches_nodes(analysis_path: Path, nodes_path: Path, sample: int = 200) -> bool:
    node_ids = set()
    with nodes_path.open("r", encoding="utf-8") as handle:
        for i, line in enumerate(handle):
            if i >= sample:
                break
            node_ids.add(json.loads(line)["id"])

    match = 0
    with analysis_path.open("r", encoding="utf-8") as handle:
        for i, line in enumerate(handle):
            if i >= sample:
                break
            if json.loads(line).get("id") in node_ids:
                match += 1
    return match > 0

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


def _progress(iterable, **kwargs):
    return tqdm(iterable, disable=not sys.stderr.isatty(), **kwargs)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train polynomial->circuit transformer")
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
    parser.add_argument(
        "--include-non-multipath",
        action="store_true",
        help="Include nodes without multiple paths (default: multipath-only)",
    )
    parser.add_argument(
        "--allow-all",
        action="store_true",
        help="Allow training without analysis JSONL",
    )
    parser.add_argument("--limit", type=int, default=None, help="Max number of examples")
    parser.add_argument(
        "--expand-circuit",
        action="store_true",
        help="Train on expanded polynomial targets instead of circuit expressions",
    )
    parser.add_argument("--seed", type=int, default=7, help="RNG seed")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--d-model", type=int, default=256, help="Transformer model width")
    parser.add_argument("--nhead", type=int, default=8, help="Transformer attention heads")
    parser.add_argument("--enc-layers", type=int, default=3, help="Encoder layers")
    parser.add_argument("--dec-layers", type=int, default=3, help="Decoder layers")
    parser.add_argument("--ffn-dim", type=int, default=512, help="Feedforward dim")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout")
    parser.add_argument("--output", type=Path, required=True, help="Checkpoint output path")
    parser.add_argument("--metrics-out", type=Path, default=None, help="Write metrics JSONL")
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation split fraction")
    parser.add_argument("--plot-out", type=Path, default=None, help="Write metrics plot PNG")
    parser.add_argument("--device", type=str, default=None, help="Override device")
    return parser.parse_args()


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
    elif not _analysis_matches_nodes(analysis_path, nodes_path):
        raise SystemExit(
            "Analysis JSONL does not match node IDs. "
            "Double-check --analysis or regenerate the analysis for this board."
        )

    examples = build_dataset(
        nodes_path=nodes_path,
        edges_path=edges_path,
        analysis_path=analysis_path,
        max_complexity=args.max_complexity,
        only_multipath=only_multipath,
        expand_circuit=args.expand_circuit,
    )

    random.seed(args.seed)
    random.shuffle(examples)

    if args.limit is not None:
        examples = examples[: args.limit]

    if not examples:
        raise SystemExit("No examples available for training")

    if not (0.0 <= args.val_split < 1.0):
        raise SystemExit("--val-split must be in [0, 1).")

    split_idx = int(len(examples) * (1.0 - args.val_split))
    train_examples = examples[:split_idx]
    val_examples = examples[split_idx:] if args.val_split > 0 else []

    tokenizer = CharTokenizer.build([ex.polynomial for ex in examples] + [ex.circuit for ex in examples])
    dataset = PolyCircuitDataset(train_examples, tokenizer)

    pad_id = tokenizer.vocab["<pad>"]
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: _collate(batch, pad_id),
    )
    val_loader = None
    if val_examples:
        val_dataset = PolyCircuitDataset(val_examples, tokenizer)
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=lambda batch: _collate(batch, pad_id),
        )

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Seq2SeqTransformer(
        vocab_size=len(tokenizer),
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        dim_feedforward=args.ffn_dim,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id)

    model.train()
    if args.metrics_out:
        args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        batch_count = 0
        pbar = _progress(
            loader,
            desc=f"Epoch {epoch}/{args.epochs}",
            unit="batch",
            leave=False,
        )
        for src, tgt_in, tgt_out, src_mask, tgt_mask in pbar:
            src = src.to(device)
            tgt_in = tgt_in.to(device)
            tgt_out = tgt_out.to(device)
            src_mask = src_mask.to(device)
            tgt_mask = tgt_mask.to(device)
            causal_mask = generate_square_subsequent_mask(tgt_in.size(0), device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(
                src,
                tgt_in,
                tgt_mask=causal_mask,
                src_key_padding_mask=src_mask,
                tgt_key_padding_mask=tgt_mask,
                memory_key_padding_mask=src_mask,
            )
            loss = loss_fn(logits.view(-1, logits.size(-1)), tgt_out.reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            batch_count += 1
            if batch_count % 5 == 0:
                avg = total_loss / max(batch_count, 1)
                pbar.set_postfix(loss=f"{avg:.4f}")

        avg_loss = total_loss / max(len(loader), 1)
        print(f"Epoch {epoch}/{args.epochs} - loss: {avg_loss:.4f}")

        val_loss = math.nan
        val_eq = math.nan
        if val_loader is not None:
            model.eval()
            total_val_loss = 0.0
            total_val_tokens = 0
            eq_hits = 0
            eq_total = 0
            with torch.no_grad():
                for src, tgt_in, tgt_out, src_mask, tgt_mask in _progress(
                    val_loader,
                    desc="Val",
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
                    total_val_loss += float(loss.item()) * tokens
                    total_val_tokens += tokens

                    batch_size = src.size(1)
                    for i in range(batch_size):
                        decoded = greedy_decode(model, src[:, i : i + 1], tokenizer)
                        target = tokenizer.decode(tgt_out[:, i].tolist())
                        if _equivalent(decoded, target):
                            eq_hits += 1
                        eq_total += 1
            val_loss = total_val_loss / max(total_val_tokens, 1)
            val_eq = eq_hits / max(eq_total, 1)
            print(f"  val loss: {val_loss:.4f} | val eq: {val_eq:.3f}")
            model.train()

        if args.metrics_out:
            with args.metrics_out.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "epoch": epoch,
                            "loss": avg_loss,
                            "val_loss": val_loss,
                            "val_equivalence": val_eq,
                            "timestamp": time.time(),
                        }
                    )
                    + "\n"
                )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(args.output, model, tokenizer)
    print(f"Saved checkpoint to {args.output}")

    if args.plot_out and args.metrics_out and args.metrics_out.exists():
        try:
            import matplotlib.pyplot as plt
        except Exception:
            print("matplotlib not available; skipping plot.")
            return

        epochs = []
        losses = []
        val_losses = []
        val_eqs = []
        with args.metrics_out.open("r", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                epochs.append(record.get("epoch"))
                losses.append(record.get("loss"))
                val_losses.append(record.get("val_loss"))
                val_eqs.append(record.get("val_equivalence"))

        args.plot_out.parent.mkdir(parents=True, exist_ok=True)
        fig, ax1 = plt.subplots(figsize=(7, 4))
        ax1.plot(epochs, losses, label="train loss")
        if any(v is not None for v in val_losses):
            ax1.plot(epochs, val_losses, label="val loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.grid(True, alpha=0.3)

        ax2 = ax1.twinx()
        if any(v is not None and not math.isnan(v) for v in val_eqs):
            ax2.plot(epochs, val_eqs, color="tab:green", label="val eq")
        ax2.set_ylabel("Structural equivalence")

        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc="best")
        fig.tight_layout()
        fig.savefig(args.plot_out, dpi=150)
        print(f"Wrote plot to {args.plot_out}")


if __name__ == "__main__":
    main()
