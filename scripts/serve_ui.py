#!/usr/bin/env python3
"""
Local web UI for live inference of arithmetic-circuit RL checkpoints.

Spawns a Flask app that:
  * Serves a single-page demo (scripts/ui/) with KaTeX-rendered polynomials,
    per-model summary cards, and per-polynomial result rows.
  * Loads cycle_001.pt, cycle_012.pt (plus a hand-crafted heuristic) on demand
    and caches the loaded models for subsequent runs.
  * Streams per-polynomial evaluation results back via Server-Sent Events so
    the table fills in live as each polynomial finishes.

Usage:
    pip install flask
    python scripts/serve_ui.py
    # then open http://127.0.0.1:8000 in a browser
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

try:
    from flask import Flask, Response, jsonify, request, send_from_directory
except ImportError:
    sys.exit(
        "This demo requires Flask. Install it with:\n    pip install flask"
    )

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

# Re-use the existing test suite + inference helpers.
from evaluate_checkpoints import (
    TEST_SUITE,
    _infer_network_kwargs,
    greedy_rollout,
)


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def poly_to_latex(poly: SparsePolynomial) -> str:
    """Render a SparsePolynomial as a LaTeX math string for KaTeX."""
    if poly.is_zero:
        return "0"
    parts = []
    for coeff, exp in poly.terms:
        monomial_parts = []
        for var, p in zip(poly.variables, exp):
            if p == 0:
                continue
            monomial_parts.append(var if p == 1 else f"{var}^{{{p}}}")
        mono = "".join(monomial_parts)
        if not mono:
            parts.append(str(coeff))
        elif coeff == 1:
            parts.append(mono)
        else:
            parts.append(f"{coeff}{mono}")
    return " + ".join(parts)


def poly_group(name: str) -> str:
    if name.startswith("F3_xyz"):
        return "Trivariate F₃"
    if name.startswith("F3_xy"):
        return "Bivariate F₃"
    if name.startswith("F3_x"):
        return "Univariate F₃"
    if name.startswith("F5_xy"):
        return "Bivariate F₅"
    return "Other"


# ---------------------------------------------------------------------------
# Model loading (cached across requests)
# ---------------------------------------------------------------------------

_MODEL_CACHE: dict[str, tuple] = {}


def _load_heuristic():
    return (
        HeuristicPolicyValueModel(BaselineCostModel()),
        {"description": "Hand-crafted heuristic policy (no learned weights)"},
    )


def _load_checkpoint(path: Path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    sd = payload["model_state_dict"]
    kwargs = _infer_network_kwargs(sd)
    net = TorchPolicyValueNetwork(**kwargs)
    net.load_state_dict(sd)
    net.eval()
    wrapper = TorchPolicyValueWrapper(net, device="cpu")
    meta = payload.get("metadata", {})
    holdout = meta.get("holdout_eval") or meta.get("holdout_after") or {}
    info = {
        "cycle": meta.get("cycle"),
        "stage": meta.get("stage"),
        "params": sum(p.numel() for p in net.parameters()),
        "hidden_dim": kwargs["hidden_dim"],
        "shared_layers": kwargs["shared_layers"],
        "value_layers": kwargs["value_layers"],
        "saved_at": payload.get("saved_at_utc", ""),
        "holdout_gain": holdout.get("average_search_gain"),
    }
    return wrapper, info


def get_model(label: str, ckpt_dir: Path):
    if label in _MODEL_CACHE:
        return _MODEL_CACHE[label]
    if label == "heuristic":
        model, info = _load_heuristic()
    else:
        ckpt_path = ckpt_dir / f"{label}.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        model, info = _load_checkpoint(ckpt_path)
    _MODEL_CACHE[label] = (model, info)
    return model, info


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------

app = Flask(
    __name__,
    static_folder=str(_REPO_ROOT / "scripts" / "ui"),
    static_url_path="",
)
_BASELINE_MODEL = BaselineCostModel()
_BASELINE_BUNDLE = BaselineBundle()


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/test-suite")
def api_test_suite():
    items = []
    for i, (name, poly) in enumerate(TEST_SUITE):
        items.append({
            "index": i,
            "name": name,
            "group": poly_group(name),
            "latex": poly_to_latex(poly),
            "prime": poly.p,
            "variables": list(poly.variables),
            "support": poly.support_size,
            "degree": poly.total_degree,
            "baseline_min": _BASELINE_BUNDLE.min_cost(poly),
        })
    return jsonify({"polynomials": items})


@app.route("/api/models")
def api_models():
    """Report which models the UI can offer (baseline + any .pt files found)."""
    ckpt_dir = _REPO_ROOT
    available = [{
        "label": "heuristic",
        "display": "Heuristic",
        "kind": "baseline",
        "exists": True,
    }]
    for label in ("cycle_001", "cycle_012"):
        path = ckpt_dir / f"{label}.pt"
        available.append({
            "label": label,
            "display": label,
            "kind": "checkpoint",
            "exists": path.exists(),
        })
    return jsonify({"models": available})


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.route("/api/evaluate")
def api_evaluate():
    """Stream per-polynomial inference results as Server-Sent Events."""
    search_sims = max(1, int(request.args.get("search_sims", 32)))
    k_candidates = max(1, int(request.args.get("k", 16)))
    selected = request.args.getlist("models") or [
        "heuristic", "cycle_001", "cycle_012",
    ]
    ckpt_dir = _REPO_ROOT

    def stream():
        yield _sse("session-start", {
            "search_sims": search_sims,
            "k_candidates": k_candidates,
            "models": selected,
            "total_polys": len(TEST_SUITE),
        })
        for label in selected:
            try:
                model, info = get_model(label, ckpt_dir)
            except FileNotFoundError as e:
                yield _sse("model-error", {"label": label, "error": str(e)})
                continue

            yield _sse("model-start", {"label": label, "info": info})
            t_model = time.perf_counter()
            env = DecompEnv(baseline_model=_BASELINE_MODEL)
            search = AndOrSearch(
                baseline_model=_BASELINE_MODEL,
                model=model,
                search_config=SearchConfig(simulations=search_sims),
            )
            try:
                for i, (name, poly) in enumerate(TEST_SUITE):
                    p_start = time.perf_counter()
                    bmin = _BASELINE_BUNDLE.min_cost(poly)
                    gcost = greedy_rollout(env, model, poly, k=k_candidates)
                    sr = search.search(poly)
                    scost = (
                        int(sr.best_cost)
                        if not math.isinf(sr.best_cost)
                        else bmin + 999
                    )
                    yield _sse("result", {
                        "label": label,
                        "poly_index": i,
                        "poly_name": name,
                        "baseline_min": bmin,
                        "greedy_cost": gcost,
                        "search_cost": scost,
                        "greedy_success": gcost <= bmin,
                        "search_success": scost <= bmin,
                        "node_expansions": sr.stats.node_expansions,
                        "transposition_hits": sr.stats.transposition_hits,
                        "elapsed_ms": int(1000 * (time.perf_counter() - p_start)),
                    })
            finally:
                search.close()
            yield _sse("model-done", {
                "label": label,
                "elapsed_sec": round(time.perf_counter() - t_model, 2),
            })
        yield _sse("session-complete", {})

    return Response(
        stream(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Local web UI for arithmetic-circuit RL inference."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    url = f"http://{args.host}:{args.port}"
    print("=" * 60)
    print("  Circuit RL Inference Demo")
    print("=" * 60)
    print(f"  Serving at: {url}")
    print(f"  Checkpoint dir: {_REPO_ROOT}")
    print("  Press Ctrl+C to stop.")
    print("=" * 60)
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)


if __name__ == "__main__":
    main()
