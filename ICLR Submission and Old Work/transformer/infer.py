"""
Run inference: polynomial string -> SymPy circuit string.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import sympy
import torch

try:
    from transformer.polynomial_to_circuit import load_checkpoint, greedy_decode
except ModuleNotFoundError:
    from polynomial_to_circuit import load_checkpoint, greedy_decode


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Infer circuit from polynomial string")
    parser.add_argument("--poly", type=str, required=True, help="Polynomial string input")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Checkpoint path")
    parser.add_argument("--max-len", type=int, default=256, help="Max decode length")
    parser.add_argument(
        "--auto-map-vars",
        action="store_true",
        help="Auto-map x,y,z,... -> x0,x1,x2 if checkpoint vocab uses indexed vars",
    )
    parser.add_argument("--device", type=str, default=None, help="Override device")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, tokenizer = load_checkpoint(args.checkpoint, device)

    poly = args.poly
    if args.auto_map_vars and "x0" in tokenizer.vocab:
        try:
            expr = sympy.sympify(poly.replace("^", "**"))
            symbols = sorted(expr.free_symbols, key=lambda s: s.name)
            remap = {sym: sympy.Symbol(f"x{i}") for i, sym in enumerate(symbols)}
            if remap:
                poly = str(expr.xreplace(remap))
        except Exception:
            pass

    poly = poly.replace("**", "^")
    src_ids = tokenizer.encode(poly, add_bos=True, add_eos=True)
    src = torch.tensor(src_ids, dtype=torch.long, device=device).unsqueeze(1)
    output = greedy_decode(model, src, tokenizer, max_len=args.max_len)
    print(output)


if __name__ == "__main__":
    main()
