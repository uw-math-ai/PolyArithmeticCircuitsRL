"""
Polynomial -> circuit SymPy string translation using game-board data and a seq2seq transformer.

This module provides two paths:
  1) Deterministic reconstruction from Game-Board-Generation JSONL files.
  2) Transformer-based sequence-to-sequence translation (trainable).
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import sympy
import torch
from torch import nn


# ---------------------------
# Game-board IO + reconstruction
# ---------------------------

def _canonical_key(expr: sympy.Expr) -> str:
    return sympy.srepr(sympy.expand(expr))


def _safe_sympify(expr_str: str) -> sympy.Expr:
    try:
        return sympy.sympify(expr_str)
    except Exception:
        return sympy.sympify(expr_str.replace("^", "**"))


def _format_sympy_string(expr: sympy.Expr) -> str:
    """Format SymPy expression with caret exponentiation for training/output."""
    return str(expr).replace("**", "^")


def load_game_board(
    nodes_path: Path,
    edges_path: Path,
) -> Tuple[
    Dict[str, sympy.Expr],
    Dict[str, int],
    Dict[str, List[Tuple[str, str, str]]],
    List[str],
    Dict[str, str],
]:
    """Load nodes/edges and return data needed to reconstruct circuits."""
    node_exprs: Dict[str, sympy.Expr] = {}
    node_steps: Dict[str, int] = {}
    base_symbol_ids: List[str] = []
    node_expr_str: Dict[str, str] = {}

    with nodes_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            node_id = record["id"]
            expr_str = record.get("expr_str") or record.get("label") or node_id
            expr = _safe_sympify(expr_str)
            node_exprs[node_id] = expr
            node_steps[node_id] = int(record.get("step", 0))
            node_expr_str[node_id] = expr_str
            if record.get("step", 0) == 0 and isinstance(expr, sympy.Symbol):
                base_symbol_ids.append(node_id)

    operations: Dict[str, List[Tuple[str, str, str]]] = {}
    dedup = set()
    with edges_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            source = record.get("source")
            target = record.get("target")
            operand = record.get("operand")
            op = record.get("op")
            if not source or not target or operand is None:
                continue
            if op not in ("add", "mul"):
                continue
            ordered = tuple(sorted([source, operand]))
            key = (target, op, ordered)
            if key in dedup:
                continue
            dedup.add(key)
            operations.setdefault(target, []).append((op, ordered[0], ordered[1]))

    base_symbol_ids = sorted(base_symbol_ids)
    return node_exprs, node_steps, operations, base_symbol_ids, node_expr_str


def resolve_node_id(
    poly_str: str,
    node_exprs: Dict[str, sympy.Expr],
    node_expr_str: Dict[str, str],
) -> str:
    """Resolve a polynomial string to a node id using canonical SymPy keys."""
    expr = _safe_sympify(poly_str)
    key = _canonical_key(expr)
    if key in node_exprs:
        return key

    # Try exact string match fallback
    for node_id, expr_str in node_expr_str.items():
        if expr_str == poly_str:
            return node_id

    # Last resort: compare canonical keys of stored exprs
    for node_id, expr_candidate in node_exprs.items():
        if _canonical_key(expr_candidate) == key:
            return node_id

    raise KeyError(f"Polynomial not found in game board: {poly_str}")


def build_actions_for_target(
    target_id: str,
    operations: Dict[str, List[Tuple[str, str, str]]],
    node_steps: Dict[str, int],
    base_symbol_ids: Sequence[str],
    include_constant: bool = False,
) -> Tuple[List[Tuple[str, int, int]], List[str]]:
    """Reconstruct a circuit (actions list) for a target node id."""
    actions: List[Tuple[str, int, int]] = []
    node_to_idx: Dict[str, int] = {}
    variable_names: List[str] = []

    for input_idx, node_id in enumerate(base_symbol_ids):
        actions.append(("input", input_idx, -1))
        node_to_idx[node_id] = input_idx
        variable_names.append(node_id)

    if include_constant:
        constant_idx = len(actions)
        actions.append(("constant", -1, -1))
        node_to_idx["CONST_1"] = constant_idx

    visiting = set()

    def ensure_node(node_id: str) -> int:
        if node_id in node_to_idx:
            return node_to_idx[node_id]
        if node_id in visiting:
            raise RuntimeError(f"Cyclic dependency detected for {node_id}")

        ops = operations.get(node_id, [])
        if not ops:
            raise KeyError(node_id)

        visiting.add(node_id)
        ops_sorted = sorted(
            ops,
            key=lambda entry: (
                max(
                    node_steps.get(entry[1], math.inf),
                    node_steps.get(entry[2], math.inf),
                ),
                entry[0],
                entry[1],
                entry[2],
            ),
        )
        for op_type, left_id, right_id in ops_sorted:
            try:
                left_idx = ensure_node(left_id)
                right_idx = ensure_node(right_id)
            except KeyError:
                continue

            node_index = len(actions)
            action_type = "add" if op_type == "add" else "mul"
            actions.append((action_type, left_idx, right_idx))
            node_to_idx[node_id] = node_index
            visiting.remove(node_id)
            return node_index

        visiting.remove(node_id)
        raise KeyError(node_id)

    ensure_node(target_id)
    return actions, variable_names


def actions_to_sympy_string(
    actions: Sequence[Tuple[str, int, int]],
    base_exprs: Sequence[sympy.Expr],
) -> str:
    """Render the final circuit output as a SymPy expression string."""
    if not actions:
        return ""

    nodes: List[sympy.Expr] = []
    for op, in1, in2 in actions:
        if op == "input":
            if in1 >= len(base_exprs):
                raise IndexError(f"Input index {in1} outside base exprs size {len(base_exprs)}")
            nodes.append(base_exprs[in1])
        elif op == "constant":
            nodes.append(sympy.Integer(1))
        elif op == "add":
            nodes.append(sympy.expand(nodes[in1] + nodes[in2]))
        elif op == "mul":
            nodes.append(sympy.expand(nodes[in1] * nodes[in2]))
        else:
            raise ValueError(f"Unknown op: {op}")

    return _format_sympy_string(sympy.expand(nodes[-1]))


# ---------------------------
# Dataset builder
# ---------------------------

@dataclass
class CircuitExample:
    polynomial: str
    circuit: str


def build_dataset(
    nodes_path: Path,
    edges_path: Path,
    analysis_path: Optional[Path] = None,
    max_complexity: Optional[int] = None,
    only_multipath: bool = False,
) -> List[CircuitExample]:
    """Build polynomial->circuit pairs from game-board dumps."""
    node_exprs, node_steps, operations, base_symbol_ids, node_expr_str = load_game_board(
        nodes_path, edges_path
    )

    targets: List[Tuple[str, int]] = []
    if analysis_path and analysis_path.exists():
        with analysis_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                shortest_length = record.get("shortest_length")
                if shortest_length is None:
                    continue
                if max_complexity is not None and shortest_length > max_complexity:
                    continue
                if only_multipath and not (
                    record.get("multiple_shortest_paths") or record.get("multiple_paths")
                ):
                    continue
                targets.append((record["id"], int(shortest_length)))
    else:
        for node_id, step in node_steps.items():
            if max_complexity is None or step <= max_complexity:
                targets.append((node_id, step))

    examples: List[CircuitExample] = []
    for node_id, _ in targets:
        if node_id not in node_exprs:
            continue
        try:
            actions, _ = build_actions_for_target(
                node_id, operations, node_steps, base_symbol_ids
            )
        except (KeyError, RuntimeError):
            continue
        base_exprs = [node_exprs[node_id] for node_id in base_symbol_ids]
        circuit_str = actions_to_sympy_string(actions, base_exprs)
        poly_str = node_expr_str.get(node_id) or str(node_exprs[node_id])
        poly_str = poly_str.replace("**", "^")
        examples.append(CircuitExample(polynomial=poly_str, circuit=circuit_str))

    return examples


# ---------------------------
# Transformer model + tokenizer
# ---------------------------

class CharTokenizer:
    def __init__(self, vocab: Optional[Dict[str, int]] = None):
        self.vocab = vocab or {}
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

    @staticmethod
    def build(texts: Iterable[str]) -> "CharTokenizer":
        vocab = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3}
        for text in texts:
            for ch in text:
                if ch not in vocab:
                    vocab[ch] = len(vocab)
        return CharTokenizer(vocab)

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        tokens = []
        if add_bos:
            tokens.append(self.vocab["<bos>"])
        for ch in text:
            tokens.append(self.vocab.get(ch, self.vocab["<unk>"]))
        if add_eos:
            tokens.append(self.vocab["<eos>"])
        return tokens

    def decode(self, ids: Iterable[int]) -> str:
        chars = []
        for idx in ids:
            token = self.inv_vocab.get(idx, "")
            if token in ("<pad>", "<bos>", "<eos>"):
                continue
            chars.append(token)
        return "".join(chars)

    def __len__(self) -> int:
        return len(self.vocab)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class Seq2SeqTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.model_type = "Transformer"
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dim_feedforward = dim_feedforward
        self.src_embed = nn.Embedding(vocab_size, d_model)
        self.tgt_embed = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.pos_decoder = PositionalEncoding(d_model, dropout)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.generator = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        src_emb = self.pos_encoder(self.src_embed(src))
        tgt_emb = self.pos_decoder(self.tgt_embed(tgt))
        output = self.transformer(
            src_emb,
            tgt_emb,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return self.generator(output)

    def encode(self, src: torch.Tensor, src_key_padding_mask: Optional[torch.Tensor] = None):
        return self.transformer.encoder(self.pos_encoder(self.src_embed(src)), src_key_padding_mask)

    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ):
        return self.transformer.decoder(
            self.pos_decoder(self.tgt_embed(tgt)),
            memory,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )


# ---------------------------
# Translator wrapper
# ---------------------------

@dataclass
class TransformerCheckpoint:
    state_dict: Dict[str, torch.Tensor]
    vocab: Dict[str, int]
    config: Dict[str, int]


def save_checkpoint(path: Path, model: Seq2SeqTransformer, tokenizer: CharTokenizer) -> None:
    payload = {
        "state_dict": model.state_dict(),
        "vocab": tokenizer.vocab,
        "config": {
            "d_model": model.d_model,
            "nhead": model.nhead,
            "num_encoder_layers": model.num_encoder_layers,
            "num_decoder_layers": model.num_decoder_layers,
            "dim_feedforward": model.dim_feedforward,
        },
    }
    torch.save(payload, path)


def load_checkpoint(path: Path, device: torch.device) -> Tuple[Seq2SeqTransformer, CharTokenizer]:
    payload = torch.load(path, map_location=device)
    config = payload.get("config", {})
    tokenizer = CharTokenizer(payload["vocab"])
    model = Seq2SeqTransformer(
        vocab_size=len(tokenizer),
        d_model=int(config.get("d_model", 256)),
        nhead=int(config.get("nhead", 8)),
        num_encoder_layers=int(config.get("num_encoder_layers", 3)),
        num_decoder_layers=int(config.get("num_decoder_layers", 3)),
        dim_feedforward=int(config.get("dim_feedforward", 512)),
    )
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return model, tokenizer


def greedy_decode(
    model: Seq2SeqTransformer,
    src: torch.Tensor,
    tokenizer: CharTokenizer,
    max_len: int = 256,
) -> str:
    device = src.device
    bos_id = tokenizer.vocab["<bos>"]
    eos_id = tokenizer.vocab["<eos>"]

    memory = model.encode(src)
    ys = torch.tensor([[bos_id]], device=device)
    for _ in range(max_len):
        out = model.decode(ys, memory)
        logits = model.generator(out[-1])
        next_id = int(torch.argmax(logits, dim=-1).item())
        ys = torch.cat([ys, torch.tensor([[next_id]], device=device)], dim=0)
        if next_id == eos_id:
            break

    return tokenizer.decode(ys.squeeze(1).tolist())


def translate_polynomial(
    poly_str: str,
    nodes_path: Path,
    edges_path: Path,
    analysis_path: Optional[Path] = None,
    checkpoint: Optional[Path] = None,
    max_len: int = 256,
) -> str:
    node_exprs, node_steps, operations, base_symbol_ids, node_expr_str = load_game_board(
        nodes_path, edges_path
    )

    if checkpoint is not None and checkpoint.exists():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, tokenizer = load_checkpoint(checkpoint, device)
        poly_str = poly_str.replace("**", "^")
        src_ids = tokenizer.encode(poly_str, add_bos=True, add_eos=True)
        src = torch.tensor(src_ids, dtype=torch.long, device=device).unsqueeze(1)
        return greedy_decode(model, src, tokenizer, max_len=max_len)

    target_id = resolve_node_id(poly_str, node_exprs, node_expr_str)
    actions, variable_names = build_actions_for_target(
        target_id, operations, node_steps, base_symbol_ids
    )
    base_exprs = [node_exprs[node_id] for node_id in base_symbol_ids]
    return actions_to_sympy_string(actions, base_exprs)


# ---------------------------
# CLI
# ---------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Translate polynomials to SymPy circuit strings")
    parser.add_argument("--poly", type=str, required=True, help="Polynomial string input")
    parser.add_argument("--nodes", type=Path, required=True, help="Nodes JSONL path")
    parser.add_argument("--edges", type=Path, required=True, help="Edges JSONL path")
    parser.add_argument("--analysis", type=Path, default=None, help="Analysis JSONL path")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Transformer checkpoint")
    parser.add_argument("--max-len", type=int, default=256, help="Max decode length")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    circuit = translate_polynomial(
        args.poly,
        nodes_path=args.nodes,
        edges_path=args.edges,
        analysis_path=args.analysis,
        checkpoint=args.checkpoint,
        max_len=args.max_len,
    )
    print(circuit)


if __name__ == "__main__":
    main()
