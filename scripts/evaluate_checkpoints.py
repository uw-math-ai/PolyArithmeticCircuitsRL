#!/usr/bin/env python3
"""
Checkpoint evaluation script for arithmetic circuit RL.

Auto-discovers every ``*.pt`` file inside ``<repo_root>/checkpoints/`` (or any
``--checkpoint-dir`` you pass) and evaluates each one — plus a no-ML heuristic
baseline — on a fixed set of 21 polynomials spanning bivariate/trivariate/
univariate structures over F_3 and F_5.

Success criterion: the agent's discovered circuit cost is <= the minimum cost
achieved by all five closed-form baselines (BaselineBundle.min_cost). Two
inference modes are reported for each checkpoint:

  Greedy rollout  — pure policy, no search tree; picks the highest-probability
                    candidate at each frontier step.
  AndOrSearch     — PUCT-guided search (default 32 simulations) with the model
                    as policy/value prior; reports best_cost across the tree.

Usage (from the repo root):
    python scripts/evaluate_checkpoints.py
    python scripts/evaluate_checkpoints.py --search-sims 96
    python scripts/evaluate_checkpoints.py --checkpoint-dir /path/to/ckpts
    python scripts/evaluate_checkpoints.py --checkpoints cycle_001,cycle_012
"""
from __future__ import annotations

import argparse
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

# Make the src/ package importable when running directly from the repo root.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

import torch

from decomp_rl.andor_search import AndOrSearch
from decomp_rl.baseline_cost import BaselineCostModel
from decomp_rl.baselines import BaselineBundle
from decomp_rl.config import SearchConfig
from decomp_rl.decomp_env import DecompEnv
from decomp_rl.model import (
    HeuristicPolicyValueModel,
    TorchPolicyValueNetwork,
    TorchPolicyValueWrapper,
)
from decomp_rl.polynomial import SparsePolynomial


# ---------------------------------------------------------------------------
# Fixed test suite
# ---------------------------------------------------------------------------

def _p(prime: int, variables: tuple[str, ...], terms: tuple) -> SparsePolynomial:
    return SparsePolynomial(prime, variables, terms)


# 21 polynomials chosen to cover diverse structures: factorizable products,
# Horner-amenable chains, elementary symmetric forms, and mixed-degree polys,
# across primes 3 and 5 and variable counts 1–9.
TEST_SUITE: list[tuple[str, SparsePolynomial]] = [
    # ---- Bivariate F_3, vars (x, y) ----------------------------------------
    # xy + x + y  =  (x+1)(y+1) - 1  [classic factorizable, 3 terms]
    ("F3_xy  xy+x+y",
     _p(3, ("x", "y"), ((1,(1,1)),(1,(1,0)),(1,(0,1))))),
    # x²y + xy²  =  xy(x+y)  [2-term, clear common-factor split]
    ("F3_xy  x2y+xy2 [=xy(x+y)]",
     _p(3, ("x", "y"), ((1,(2,1)),(1,(1,2))))),
    # x² + 2xy + y²  =  (x+y)²  [perfect square, 3 terms]
    ("F3_xy  x2+2xy+y2 [=(x+y)^2]",
     _p(3, ("x", "y"), ((1,(2,0)),(2,(1,1)),(1,(0,2))))),
    # x²y² + xy + 1  [3 terms, mixed degree]
    ("F3_xy  x2y2+xy+1",
     _p(3, ("x", "y"), ((1,(2,2)),(1,(1,1)),(1,(0,0))))),
    # x² + xy + y² + x + y  [5 terms, complete low-degree pattern]
    ("F3_xy  x2+xy+y2+x+y",
     _p(3, ("x", "y"), ((1,(2,0)),(1,(1,1)),(1,(0,2)),(1,(1,0)),(1,(0,1))))),
    # x³ + y³  [sum of cubes: (x+y)(x²−xy+y²), 2 terms, degree 3]
    ("F3_xy  x3+y3 [sum of cubes]",
     _p(3, ("x", "y"), ((1,(3,0)),(1,(0,3))))),
    # x²y + xy + x + y  [4 terms, factorable as (x+1)(xy+y) = y(x+1)²]
    ("F3_xy  x2y+xy+x+y",
     _p(3, ("x", "y"), ((1,(2,1)),(1,(1,1)),(1,(1,0)),(1,(0,1))))),
    # x²y² + x² + y²  [3 terms, mixed-degree bivariate]
    ("F3_xy  x2y2+x2+y2",
     _p(3, ("x", "y"), ((1,(2,2)),(1,(2,0)),(1,(0,2))))),

    # ---- Univariate F_3, var (x,) -------------------------------------------
    # x⁴ + x³ + x² + x + 1  [5-term cyclotomic-like; Horner cost = 4]
    ("F3_x   x4+x3+x2+x+1",
     _p(3, ("x",), ((1,(4,)),(1,(3,)),(1,(2,)),(1,(1,)),(1,(0,))))),
    # x⁴ + 2x² + 1  =  (x²+1)²  [perfect square, 3 terms; factorizer finds it]
    ("F3_x   x4+2x2+1 [=(x2+1)^2]",
     _p(3, ("x",), ((1,(4,)),(2,(2,)),(1,(0,))))),
    # x³ + 2x² + 2x + 1  [4 terms, Horner-amenable]
    ("F3_x   x3+2x2+2x+1",
     _p(3, ("x",), ((1,(3,)),(2,(2,)),(2,(1,)),(1,(0,))))),
    # 2x⁴ + x³ + x + 2  [4 terms, mixed coefficients]
    ("F3_x   2x4+x3+x+2",
     _p(3, ("x",), ((2,(4,)),(1,(3,)),(1,(1,)),(2,(0,))))),

    # ---- Trivariate F_3, vars (x, y, z) ------------------------------------
    # xy + xz + yz  [elementary symmetric e₂ in 3 variables]
    ("F3_xyz xy+xz+yz [e2]",
     _p(3, ("x","y","z"), ((1,(1,1,0)),(1,(1,0,1)),(1,(0,1,1))))),
    # xyz + xy + xz + yz  [4 terms; = (xy+xz+yz)(1+…) style]
    ("F3_xyz xyz+xy+xz+yz",
     _p(3, ("x","y","z"), ((1,(1,1,1)),(1,(1,1,0)),(1,(1,0,1)),(1,(0,1,1))))),
    # x² + y² + z² + xy + xz + yz  [6 terms, degree-2 complete]
    ("F3_xyz x2+y2+z2+xy+xz+yz",
     _p(3, ("x","y","z"), ((1,(2,0,0)),(1,(0,2,0)),(1,(0,0,2)),
                           (1,(1,1,0)),(1,(1,0,1)),(1,(0,1,1))))),
    # xyz + x + y + z  [4 terms, mixed degree 3 and 1]
    ("F3_xyz xyz+x+y+z",
     _p(3, ("x","y","z"), ((1,(1,1,1)),(1,(1,0,0)),(1,(0,1,0)),(1,(0,0,1))))),

    # ---- 3x3 permanent over F_3, vars (a,...,i) ---------------------------
    # permanent([[a,b,c],[d,e,f],[g,h,i]])
    ("F3_perm3  3x3 permanent",
     _p(3, ("a","b","c","d","e","f","g","h","i"),
        ((1,(1,0,0,0,1,0,0,0,1)),  # aei
         (1,(1,0,0,0,0,1,0,1,0)),  # afh
         (1,(0,1,0,1,0,0,0,0,1)),  # bdi
         (1,(0,1,0,0,0,1,1,0,0)),  # bfg
         (1,(0,0,1,1,0,0,0,1,0)),  # cdh
         (1,(0,0,1,0,1,0,1,0,0))))),  # ceg

    # ---- Bivariate F_5, vars (x, y) ----------------------------------------
    # x² + 4y²  =  (x+y)(x+4y)  [since −1 ≡ 4 mod 5; factorizable, 2 terms]
    ("F5_xy  x2+4y2 [=(x+y)(x+4y)]",
     _p(5, ("x", "y"), ((1,(2,0)),(4,(0,2))))),
    # x²y + xy² + x + y  =  (x+y)(xy+1)  [4 terms, factorizable product]
    ("F5_xy  x2y+xy2+x+y [=(x+y)(xy+1)]",
     _p(5, ("x", "y"), ((1,(2,1)),(1,(1,2)),(1,(1,0)),(1,(0,1))))),
    # x² + xy + y²  [irreducible quadratic; tests model on hard cases]
    ("F5_xy  x2+xy+y2",
     _p(5, ("x", "y"), ((1,(2,0)),(1,(1,1)),(1,(0,2))))),
    # x³y + xy³  =  xy(x²+y²)  [2 terms, common-factor split, degree 4]
    ("F5_xy  x3y+xy3 [=xy(x2+y2)]",
     _p(5, ("x", "y"), ((1,(3,1)),(1,(1,3))))),
]


# The permanent row has nine variables and can generate a much larger search
# tree than the rest of the demo suite. Cap only that row so the visualizer
# stays responsive even when the global search setting is raised.
PER_POLY_SEARCH_LIMITS: dict[str, dict[str, int]] = {
    "F3_perm3  3x3 permanent": {
        "search_sims": 2,
        "k_candidates": 8,
        "max_depth": 3,
        "expand_top_k": 8,
    },
}


def search_budget_for_poly(
    name: str,
    requested_search_sims: int,
    requested_k_candidates: int,
) -> dict[str, int | bool]:
    """Return the per-polynomial inference budget after safety caps."""
    base = SearchConfig()
    budget: dict[str, int | bool] = {
        "search_sims": max(1, requested_search_sims),
        "k_candidates": max(1, requested_k_candidates),
        "max_depth": base.max_depth,
        "expand_top_k": base.expand_top_k,
        "capped": False,
    }
    for key, limit in PER_POLY_SEARCH_LIMITS.get(name, {}).items():
        current = int(budget[key])
        if current > limit:
            budget[key] = limit
            budget["capped"] = True
    return budget


# ---------------------------------------------------------------------------
# Checkpoint loading with automatic architecture inference
# ---------------------------------------------------------------------------

def _infer_network_kwargs(state_dict: dict) -> dict:
    """Recover TorchPolicyValueNetwork hyper-params from the saved state_dict.

    The checkpoint does not explicitly store architecture metadata, so we
    derive it from tensor shapes:
      - shared.0.weight  →  (hidden_dim, input_dim)
      - shared.*.weight  →  count = shared_layers
      - value_head.0.weight  →  (value_hidden_dim, target_dim)
      - value_head.*.weight  →  count = value_layers
    Activation is assumed 'relu' (the training default; not stored).
    """
    shared_w = sorted(
        k for k in state_dict if k.startswith("shared.") and k.endswith(".weight")
    )
    value_w = sorted(
        k for k in state_dict if k.startswith("value_head.") and k.endswith(".weight")
    )
    s0 = state_dict["shared.0.weight"]
    v0 = state_dict["value_head.0.weight"]
    return dict(
        input_dim=int(s0.shape[1]),
        hidden_dim=int(s0.shape[0]),
        target_dim=int(v0.shape[1]),
        shared_layers=len(shared_w),
        value_hidden_dim=int(v0.shape[0]),
        value_layers=len(value_w),
        activation="relu",
    )


def _read_checkpoint_state(checkpoint_path: Path) -> tuple[dict, dict]:
    """Return (state_dict, metadata) for wrapped or raw state_dict checkpoints."""
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict):
        if "model_state_dict" in payload:
            return payload["model_state_dict"], payload.get("metadata", {}) or {}
        if "state_dict" in payload:
            return payload["state_dict"], payload.get("metadata", {}) or {}
        if "shared.0.weight" in payload and "value_head.0.weight" in payload:
            return payload, {}
    raise ValueError(f"Unsupported checkpoint payload format: {checkpoint_path}")


def load_torch_wrapper(checkpoint_path: Path) -> tuple[TorchPolicyValueWrapper, dict]:
    """Load a checkpoint, return (wrapper, metadata)."""
    sd, metadata = _read_checkpoint_state(checkpoint_path)
    kwargs = _infer_network_kwargs(sd)
    net = TorchPolicyValueNetwork(**kwargs)
    net.load_state_dict(sd)
    net.eval()
    wrapper = TorchPolicyValueWrapper(net, device="cpu")
    return wrapper, metadata


def discover_checkpoints(ckpt_dir: Path) -> list[Path]:
    """Return every ``*.pt`` file under ``ckpt_dir``, naturally sorted.

    A trailing integer in the stem (e.g. ``cycle_001``, ``cycle_012``) is
    treated as a numeric sort key so ``cycle_002`` orders before ``cycle_010``.
    Files without a trailing number fall back to lexicographic order.
    """
    if not ckpt_dir.is_dir():
        return []

    def sort_key(p: Path) -> tuple:
        m = re.search(r"(\d+)\s*$", p.stem)
        if m:
            return (0, int(m.group(1)), p.stem)
        return (1, 0, p.stem)

    return sorted(ckpt_dir.glob("*.pt"), key=sort_key)


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def greedy_rollout(
    env: DecompEnv,
    model,
    poly: SparsePolynomial,
    k: int = 16,
    max_steps: int = 64,
) -> int:
    """Build a circuit via greedy policy (no search tree). Returns acc_cost.

    At each step the model scores all split candidates and the highest-
    probability one is chosen.  If no candidates exist the polynomial is
    solved directly by the baseline model.  Any frontier items still
    remaining after max_steps are also solved directly.
    """
    state = env.reset(poly)
    for _ in range(max_steps):
        if not state.frontier:
            break
        candidates = env.get_candidate_splits(state, 0, k=k)
        if not candidates:
            state, _, _, _ = env.solve_direct(state, 0)
        else:
            priors, _ = model.score_candidates(state.frontier[0], candidates)
            best_idx = max(range(len(priors)), key=lambda i: priors[i])
            state, _, _, _ = env.step(state, 0, candidates[best_idx])
    while state.frontier:
        state, _, _, _ = env.solve_direct(state, 0)
    return state.acc_cost


@dataclass
class PolyResult:
    name: str
    baseline_min: int
    greedy_cost: int
    search_cost: int


def evaluate_model(
    model,
    test_suite: list[tuple[str, SparsePolynomial]],
    search_sims: int,
    k: int = 16,
) -> list[PolyResult]:
    """Evaluate model on every polynomial in test_suite.

    Returns one PolyResult per polynomial containing:
      baseline_min  — min cost across all five closed-form baselines
      greedy_cost   — cost from greedy policy rollout
      search_cost   — cost from AndOrSearch.best_cost
    """
    baseline_model = BaselineCostModel()
    bundle = BaselineBundle()
    env = DecompEnv(baseline_model=baseline_model)
    search = AndOrSearch(
        baseline_model=baseline_model,
        model=model,
        search_config=SearchConfig(simulations=search_sims),
    )
    results: list[PolyResult] = []
    try:
        for name, poly in test_suite:
            budget = search_budget_for_poly(name, search_sims, k)
            bmin = bundle.min_cost(poly)
            gcost = greedy_rollout(env, model, poly, k=int(budget["k_candidates"]))
            search.search_config = SearchConfig(
                simulations=int(budget["search_sims"]),
                max_depth=int(budget["max_depth"]),
                expand_top_k=int(budget["expand_top_k"]),
            )
            sr = search.search(poly)
            scost = int(sr.best_cost) if not math.isinf(sr.best_cost) else bmin + 999
            results.append(PolyResult(
                name=name,
                baseline_min=bmin,
                greedy_cost=gcost,
                search_cost=scost,
            ))
    finally:
        search.close()
    return results


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

_COL = 42  # polynomial name column width


def _pct(n: int, total: int) -> str:
    p = 100.0 * n / total if total else 0.0
    return f"{n}/{total}  ({p:.1f}%)"


def _ok(cost: int, bmin: int) -> str:
    return "Y" if cost <= bmin else " "


def print_checkpoint_results(label: str, results: list[PolyResult]) -> None:
    sep = "=" * (_COL + 34)
    print(f"\n{sep}")
    print(f"  Checkpoint: {label}")
    print(sep)
    hdr = (
        f"  {'Polynomial':<{_COL}}"
        f"  {'BasMin':>6}"
        f"  {'Greedy':>6} {'G✓':>2}"
        f"  {'Search':>6} {'S✓':>2}"
    )
    print(hdr)
    print(f"  {'-' * (_COL + 30)}")

    g_ok = s_ok = 0
    for r in results:
        go = r.greedy_cost <= r.baseline_min
        so = r.search_cost <= r.baseline_min
        g_ok += go
        s_ok += so
        row = (
            f"  {r.name:<{_COL}}"
            f"  {r.baseline_min:>6}"
            f"  {r.greedy_cost:>6} {_ok(r.greedy_cost, r.baseline_min):>2}"
            f"  {r.search_cost:>6} {_ok(r.search_cost, r.baseline_min):>2}"
        )
        print(row)

    n = len(results)
    print(f"  {'-' * (_COL + 30)}")
    print(f"  {'Greedy success rate':<{_COL + 9}}  {_pct(g_ok, n)}")
    print(f"  {'Search success rate':<{_COL + 9}}  {_pct(s_ok, n)}")




# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate RL checkpoints on a fixed 21-polynomial test suite."
    )
    parser.add_argument(
        "--search-sims", type=int, default=32,
        help="Number of PUCT simulations per polynomial (default: 32).",
    )
    parser.add_argument(
        "--checkpoint-dir", type=str, default="",
        help=(
            "Directory to scan for *.pt checkpoint files. "
            "Defaults to <repo_root>/checkpoints/."
        ),
    )
    parser.add_argument(
        "--checkpoints", type=str, default="",
        help=(
            "Optional comma-separated list of checkpoint stems "
            "(e.g. 'cycle_001,cycle_012') to restrict evaluation to. "
            "Defaults to all *.pt files found in --checkpoint-dir."
        ),
    )
    parser.add_argument(
        "--k-candidates", type=int, default=16,
        help="Split candidates generated per frontier step (default: 16).",
    )
    args = parser.parse_args()

    ckpt_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else _REPO_ROOT / "checkpoints"

    discovered = discover_checkpoints(ckpt_dir)
    if args.checkpoints:
        wanted = {s.strip() for s in args.checkpoints.split(",") if s.strip()}
        discovered = [p for p in discovered if p.stem in wanted]
        missing = wanted - {p.stem for p in discovered}
        for m in sorted(missing):
            print(f"[WARN] Requested checkpoint '{m}' not found in {ckpt_dir}",
                  file=sys.stderr)

    print("=" * 64)
    print("  Checkpoint Evaluation — Arithmetic Circuit RL")
    print("=" * 64)
    print(f"  Test suite       : {len(TEST_SUITE)} polynomials (fixed)")
    print(f"  Search sims      : {args.search_sims}")
    print(f"  Split candidates : {args.k_candidates} per step")
    print(f"  Checkpoint dir   : {ckpt_dir}")
    print(f"  Checkpoints      : {[p.stem for p in discovered] or '(none found)'}")
    print(
        "  Success criterion: agent_cost <= min(5 closed-form baselines)"
    )

    baseline_model = BaselineCostModel()

    # Heuristic baseline first, then every discovered checkpoint.
    checkpoints: list[tuple[str, Path | None]] = [("Heuristic (no ML)", None)]
    for p in discovered:
        checkpoints.append((p.stem, p))

    all_results: dict[str, list[PolyResult]] = {}

    for label, ckpt_path in checkpoints:
        if ckpt_path is not None and not ckpt_path.exists():
            print(
                f"\n[SKIP] Checkpoint not found: {ckpt_path}",
                file=sys.stderr,
            )
            continue

        print(f"\nEvaluating '{label}' …", end="", flush=True)
        if ckpt_path is None:
            model = HeuristicPolicyValueModel(baseline_model)
            arch_note = "hand-crafted heuristic"
        else:
            model, meta = load_torch_wrapper(ckpt_path)
            cycle = meta.get("cycle", "?")
            stage = meta.get("stage", "?")
            saved_at = meta.get("saved_at_utc", "")
            # Pull holdout search gain if available
            holdout = meta.get("holdout_eval") or meta.get("holdout_after") or {}
            gain = holdout.get("average_search_gain", None)
            gain_str = f", holdout_gain={gain:.3f}" if gain is not None else ""
            arch_note = f"cycle={cycle}, stage={stage}{gain_str}"
            if saved_at:
                arch_note += f", saved={saved_at[:10]}"
        print(f" [{arch_note}]")

        results = evaluate_model(
            model,
            TEST_SUITE,
            search_sims=args.search_sims,
            k=args.k_candidates,
        )
        all_results[label] = results
        print_checkpoint_results(label, results)

    if len(all_results) > 1:
        n = len(TEST_SUITE)
        sep = "=" * 64
        print(f"\n{sep}")
        print("  Summary")
        print(sep)
        print(f"  {'Checkpoint':<24}  {'Greedy':>16}  {'Search':>16}")
        print(f"  {'-' * 60}")
        for lbl, results in all_results.items():
            g_ok = sum(1 for r in results if r.greedy_cost <= r.baseline_min)
            s_ok = sum(1 for r in results if r.search_cost <= r.baseline_min)
            pg = 100.0 * g_ok / n if n else 0.0
            ps = 100.0 * s_ok / n if n else 0.0
            print(
                f"  {lbl:<24}"
                f"  {g_ok}/{n} ({pg:.1f}%)"
                f"  {s_ok}/{n} ({ps:.1f}%)"
            )
        print(sep)


if __name__ == "__main__":
    main()
