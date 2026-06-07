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
    discover_checkpoints,
)
from circuit_viz import (
    OPTIMAL_CIRCUITS,
    binarize,
    build_agent_circuit,
    gate_count,
    greedy_splits,
    poly_to_latex,
    splits_from_trace,
)


CHECKPOINTS_DIR = _REPO_ROOT / "checkpoints"


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

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


def _peek_checkpoint_meta(path: Path) -> dict:
    """Cheap metadata read (no tensors) for the model picker."""
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        return {"error": str(exc)}
    meta = payload.get("metadata", {}) or {}
    holdout = meta.get("holdout_eval") or meta.get("holdout_after") or {}
    return {
        "cycle": meta.get("cycle"),
        "stage": meta.get("stage"),
        "saved_at": payload.get("saved_at_utc", ""),
        "holdout_gain": holdout.get("average_search_gain"),
    }


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


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/test-suite")
def api_test_suite():
    items = []
    for i, (name, poly) in enumerate(TEST_SUITE):
        # OPTIMAL_CIRCUITS holds the provably-minimal circuit under the demo's
        # gate model: the only free constant is 1; every other scalar must be
        # built by adding 1s (reuse allowed), and gates are binary + and ×.
        # The stored cost equals the number of operation nodes the diagram shows.
        opt = OPTIMAL_CIRCUITS[name]
        opt_cost = opt["cost"]
        items.append({
            "index": i,
            "name": name,
            "group": poly_group(name),
            "latex": poly_to_latex(poly),
            "prime": poly.p,
            "variables": list(poly.variables),
            "support": poly.support_size,
            "degree": poly.total_degree,
            "baseline_min": opt_cost,
            "optimal_circuit": opt["circuit"],
            "optimal_circuit_cost": opt_cost,
        })
    return jsonify({"polynomials": items})


@app.route("/api/models")
def api_models():
    """Report the heuristic baseline + every *.pt checkpoint discovered.

    The frontend renders the heuristic as an always-on reference and the
    checkpoints as a multi-select picker.
    """
    available = [{
        "label": "heuristic",
        "display": "Heuristic",
        "kind": "baseline",
        "exists": True,
        "meta": {"description": "Hand-crafted heuristic (no learned weights)"},
    }]
    for path in discover_checkpoints(CHECKPOINTS_DIR):
        available.append({
            "label": path.stem,
            "display": path.stem,
            "kind": "checkpoint",
            "exists": True,
            "size_mb": round(path.stat().st_size / (1024 * 1024), 1),
            "meta": _peek_checkpoint_meta(path),
        })
    return jsonify({
        "models": available,
        "checkpoint_dir": str(CHECKPOINTS_DIR),
    })


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _tier(cost: int, optimal: int) -> int:
    """0 = matches optimal (green), 1 = one op away (yellow), 2 = 2+ away (red)."""
    delta = max(0, cost - optimal)
    return min(2, delta)


@app.route("/api/evaluate")
def api_evaluate():
    """Stream per-polynomial inference results as Server-Sent Events."""
    search_sims = max(1, int(request.args.get("search_sims", 32)))
    k_candidates = max(1, int(request.args.get("k", 16)))
    # Whatever the caller asked for; if empty, default to heuristic alone.
    selected = request.args.getlist("models") or ["heuristic"]
    ckpt_dir = CHECKPOINTS_DIR

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
                    optimal = OPTIMAL_CIRCUITS[name]["cost"]

                    # Greedy rollout -> materialised, binarised circuit; the cost
                    # is the new-model gate count (== rendered node count).
                    g_splits, _ = greedy_splits(env, model, poly, k=k_candidates)
                    greedy_circuit = binarize(build_agent_circuit(
                        poly, g_splits, env.factorizer, _BASELINE_MODEL))
                    gcost = gate_count(greedy_circuit)

                    # Guided search (best_trace -> splits -> circuit).
                    sr = search.search(poly)
                    s_splits = splits_from_trace(sr.best_trace)
                    search_circuit = binarize(build_agent_circuit(
                        poly, s_splits, env.factorizer, _BASELINE_MODEL))
                    scost = gate_count(search_circuit)

                    yield _sse("result", {
                        "label": label,
                        "poly_index": i,
                        "poly_name": name,
                        "optimal": optimal,
                        "greedy_cost": gcost,
                        "search_cost": scost,
                        # Tier vs optimal: 0 = optimal, 1 = one op away, 2 = 2+.
                        "greedy_tier": _tier(gcost, optimal),
                        "search_tier": _tier(scost, optimal),
                        "greedy_circuit": greedy_circuit,
                        "search_circuit": search_circuit,
                        "greedy_circuit_cost": gcost,
                        "search_circuit_cost": scost,
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
    discovered = discover_checkpoints(CHECKPOINTS_DIR)
    print(f"  Serving at: {url}")
    print(f"  Checkpoint dir: {CHECKPOINTS_DIR}")
    print(f"  Discovered checkpoints ({len(discovered)}): "
          f"{', '.join(p.stem for p in discovered) or '(none)'}")
    print("  Press Ctrl+C to stop.")
    print("=" * 60)
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)


if __name__ == "__main__":
    main()
