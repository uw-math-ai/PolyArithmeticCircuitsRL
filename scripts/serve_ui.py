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
    python scripts/serve_ui.py --checkpoint-dir models/top-down-ppo-ck-mcts-1000
    # then open http://127.0.0.1:8000 in a browser
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from dataclasses import dataclass, field
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
from decomp_rl.split_proposals import SplitAction

# Re-use the existing test suite + inference helpers.
from evaluate_checkpoints import (
    TEST_SUITE,
    _infer_network_kwargs,
    discover_checkpoints,
    search_budget_for_poly,
)
from circuit_viz import (
    OPTIMAL_CIRCUITS,
    binarize,
    build_agent_circuit,
    direct_circuit,
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
    if name.startswith("F3_perm3"):
        return "3×3 Permanent F₃"
    if name.startswith("F3_xy"):
        return "Bivariate F₃"
    if name.startswith("F3_x"):
        return "Univariate F₃"
    if name.startswith("F5_xy"):
        return "Bivariate F₅"
    return "Other"


KNOWN_CHECKPOINT_PRIMES = {
    # These checkpoints are raw state_dicts with no embedded metadata. The
    # repository handoff and launch scripts identify the current fixed-prime
    # training runs as F_3, so the UI must not silently evaluate them on F_5.
    "Top Down Best": (3,),
    "PPO MCTS Final": (3,),
    "SAC": (3,),
    "final": (3,),
}


def _coerce_primes(value) -> list[int]:
    if value is None or isinstance(value, bool):
        return []
    if isinstance(value, int):
        return [value] if value > 1 else []
    if isinstance(value, str):
        parts = [p.strip() for p in value.replace(";", ",").split(",")]
        out = []
        for part in parts:
            if part:
                try:
                    out.append(int(part))
                except ValueError:
                    pass
        return sorted({p for p in out if p > 1})
    if isinstance(value, (list, tuple, set)):
        out: list[int] = []
        for item in value:
            out.extend(_coerce_primes(item))
        return sorted(set(out))
    return []


def _checkpoint_field_info(path: Path, meta: dict) -> dict:
    prime_keys = ("supported_primes", "prime_pool", "primes")
    single_prime_keys = ("prime", "field_prime", "base_prime")
    for key in prime_keys + single_prime_keys:
        primes = _coerce_primes(meta.get(key))
        if primes:
            return {
                "supported_primes": primes,
                "field_label": ", ".join(f"F{p}" for p in primes),
                "field_source": f"checkpoint metadata: {key}",
            }
    known = KNOWN_CHECKPOINT_PRIMES.get(path.stem)
    if known:
        primes = list(known)
        return {
            "supported_primes": primes,
            "field_label": ", ".join(f"F{p}" for p in primes),
            "field_source": "repo fixed-prime run record",
        }
    return {
        "supported_primes": None,
        "field_label": "unknown",
        "field_source": "checkpoint has no prime metadata",
    }


# ---------------------------------------------------------------------------
# Model loading (cached across requests)
# ---------------------------------------------------------------------------

_MODEL_CACHE: dict[tuple[str, str], tuple] = {}


def _load_heuristic():
    return (
        HeuristicPolicyValueModel(BaselineCostModel()),
        {
            "description": "Hand-crafted heuristic policy (no learned weights)",
            "supported_primes": None,
            "field_label": "all finite fields",
            "field_source": "algorithmic baseline",
        },
    )


def _read_checkpoint_state(path: Path) -> tuple[dict, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict):
        if "model_state_dict" in payload:
            meta = dict(payload.get("metadata", {}) or {})
            if payload.get("saved_at_utc") and "saved_at_utc" not in meta:
                meta["saved_at_utc"] = payload["saved_at_utc"]
            return payload["model_state_dict"], meta
        if "state_dict" in payload:
            meta = dict(payload.get("metadata", {}) or {})
            if payload.get("saved_at_utc") and "saved_at_utc" not in meta:
                meta["saved_at_utc"] = payload["saved_at_utc"]
            return payload["state_dict"], meta
        if "shared.0.weight" in payload and "value_head.0.weight" in payload:
            return payload, {}
    raise ValueError(f"Unsupported checkpoint payload format: {path}")


def _load_checkpoint(path: Path):
    sd, meta = _read_checkpoint_state(path)
    kwargs = _infer_network_kwargs(sd)
    net = TorchPolicyValueNetwork(**kwargs)
    net.load_state_dict(sd)
    net.eval()
    wrapper = TorchPolicyValueWrapper(net, device="cpu")
    holdout = meta.get("holdout_eval") or meta.get("holdout_after") or {}
    info = {
        "cycle": meta.get("cycle"),
        "stage": meta.get("stage"),
        "params": sum(p.numel() for p in net.parameters()),
        "hidden_dim": kwargs["hidden_dim"],
        "shared_layers": kwargs["shared_layers"],
        "value_layers": kwargs["value_layers"],
        "saved_at": meta.get("saved_at_utc", ""),
        "holdout_gain": holdout.get("average_search_gain"),
        **_checkpoint_field_info(path, meta),
    }
    return wrapper, info


def _peek_checkpoint_meta(path: Path) -> dict:
    """Cheap metadata read (no tensors) for the model picker."""
    try:
        _sd, meta = _read_checkpoint_state(path)
    except Exception as exc:
        return {"error": str(exc)}
    holdout = meta.get("holdout_eval") or meta.get("holdout_after") or {}
    return {
        "cycle": meta.get("cycle"),
        "stage": meta.get("stage"),
        "saved_at": meta.get("saved_at_utc", ""),
        "holdout_gain": holdout.get("average_search_gain"),
        **_checkpoint_field_info(path, meta),
    }


def get_model(label: str, ckpt_dir: Path):
    cache_key = (str(ckpt_dir.resolve()), label)
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]
    if label == "heuristic":
        model, info = _load_heuristic()
    else:
        ckpt_path = ckpt_dir / f"{label}.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        model, info = _load_checkpoint(ckpt_path)
    _MODEL_CACHE[cache_key] = (model, info)
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


@dataclass
class PlaySession:
    session_id: str
    root_poly: SparsePolynomial
    poly_index: int
    model_label: str
    model: object
    model_info: dict
    env: DecompEnv
    state: object
    k_candidates: int
    splits: dict[str, SplitAction] = field(default_factory=dict)
    history_actors: list[str] = field(default_factory=list)
    status: str = "active"
    started_at: float = field(default_factory=time.time)


_PLAY_SESSIONS: dict[str, PlaySession] = {}
_MAX_PLAY_SESSIONS = 32


def _play_error(message: str, status: int = 400):
    return jsonify({"error": message}), status


def _cleanup_play_sessions() -> None:
    if len(_PLAY_SESSIONS) <= _MAX_PLAY_SESSIONS:
        return
    removable = [
        sid for sid, sess in _PLAY_SESSIONS.items()
        if sess.status in {"complete", "forfeit", "error"}
    ]
    while len(_PLAY_SESSIONS) > _MAX_PLAY_SESSIONS and removable:
        _PLAY_SESSIONS.pop(removable.pop(0), None)
    while len(_PLAY_SESSIONS) > _MAX_PLAY_SESSIONS:
        oldest = min(_PLAY_SESSIONS.items(), key=lambda item: item[1].started_at)[0]
        _PLAY_SESSIONS.pop(oldest, None)


def _poly_summary(poly: SparsePolynomial) -> dict:
    return {
        "key": poly.to_key(),
        "latex": poly_to_latex(poly),
        "prime": poly.p,
        "variables": list(poly.variables),
        "support": poly.support_size,
        "degree": poly.total_degree,
        "coefficients": sorted({coeff for coeff, _exp in poly.terms}),
        "terms": [[coeff, list(exp)] for coeff, exp in poly.terms],
    }


def _action_label(action: SplitAction) -> str:
    if action.kind == "factor":
        return "Factor whole polynomial"
    labels = {
        "horner": "Horner split",
        "support_partition": "Support split",
        "random_mask": "Mask split",
        "common_factor": "Common-factor split",
        "family_template": "Family split",
        "library_match": "Library split",
    }
    return labels.get(action.source, action.source.replace("_", " ").title())


def _action_summary(
    action: SplitAction,
    index: int | None = None,
    prior: float | None = None,
    is_agent: bool = False,
) -> dict:
    data = {
        "index": index,
        "kind": action.kind,
        "source": action.source,
        "label": _action_label(action),
        "score_hint": action.score_hint,
        "is_agent": is_agent,
        "g": _poly_summary(action.g),
        "h": _poly_summary(action.h),
        "metadata": [[k, v] for k, v in action.metadata],
    }
    if prior is not None:
        data["prior"] = prior
    return data


def _direct_summary(poly: SparsePolynomial, is_agent: bool = False) -> dict:
    return {
        "index": None,
        "kind": "direct",
        "source": "direct",
        "label": "Direct build",
        "is_agent": is_agent,
        "direct_cost": _BASELINE_MODEL.direct_construction_cost(poly),
        "g": _poly_summary(poly),
        "h": _poly_summary(SparsePolynomial.zero(poly.p, poly.variables)),
    }


def _preview_action_circuit(
    env: DecompEnv,
    poly: SparsePolynomial,
    action: SplitAction | None,
) -> dict:
    if action is None:
        circuit = direct_circuit(poly, _BASELINE_MODEL)
    else:
        circuit = build_agent_circuit(
            poly,
            {poly.to_key(): action},
            env.factorizer,
            _BASELINE_MODEL,
        )
    circuit = binarize(circuit)
    return {"circuit": circuit, "cost": gate_count(circuit)}


def _session_circuit(session: PlaySession) -> dict:
    circuit = binarize(build_agent_circuit(
        session.root_poly,
        session.splits,
        session.env.factorizer,
        _BASELINE_MODEL,
    ))
    return {"circuit": circuit, "cost": gate_count(circuit)}


def _history_summary(session: PlaySession) -> list[dict]:
    out = []
    for i, info in enumerate(session.state.history):
        actor = session.history_actors[i] if i < len(session.history_actors) else "player"
        item = {
            "index": i,
            "actor": actor,
            "action_kind": info.action_kind,
            "active": _poly_summary(info.active_poly),
            "baseline_before": info.baseline_before,
            "baseline_after": info.baseline_after,
            "reward": info.immediate_reward,
            "acc_cost_after": None,
            "children": [_poly_summary(child) for child in info.children],
            "cache_hits": list(info.cache_hits),
        }
        if info.split is not None:
            item["action"] = _action_summary(info.split)
        if info.action_kind == "direct":
            item["direct_cost"] = info.direct_cost
        out.append(item)
    running = 0
    for item, info in zip(out, session.state.history):
        if info.action_kind == "direct":
            running += info.direct_cost
        else:
            running += (0 if info.action_kind == "factor" else 1) + info.rebuild_g + info.rebuild_h
            for key in info.cache_hits:
                running += session.state.memo.get(key, 0)
        item["acc_cost_after"] = running
    return out


def _play_snapshot(session: PlaySession) -> dict:
    if session.status == "active" and not session.state.frontier:
        session.status = "complete"

    active_poly = session.state.frontier[0] if session.status == "active" and session.state.frontier else None
    candidates: list[SplitAction] = []
    candidate_payloads: list[dict] = []
    agent_move = None
    agent_preview = None
    policy_value = None

    if active_poly is not None:
        candidates = session.env.get_candidate_splits(
            session.state,
            0,
            k=session.k_candidates,
        )
        if candidates:
            priors, policy_value = session.model.score_candidates(active_poly, candidates)
            agent_idx = max(range(len(priors)), key=lambda i: priors[i])
            candidate_payloads = [
                _action_summary(
                    action,
                    index=i,
                    prior=priors[i],
                    is_agent=(i == agent_idx),
                )
                for i, action in enumerate(candidates)
            ]
            agent_move = candidate_payloads[agent_idx]
            agent_preview = _preview_action_circuit(
                session.env,
                active_poly,
                candidates[agent_idx],
            )
        else:
            agent_move = _direct_summary(active_poly, is_agent=True)
            agent_preview = _preview_action_circuit(session.env, active_poly, None)

    reference = OPTIMAL_CIRCUITS[TEST_SUITE[session.poly_index][0]]
    session_preview = _session_circuit(session)
    return {
        "session_id": session.session_id,
        "status": session.status,
        "model_label": session.model_label,
        "model_info": session.model_info,
        "poly_index": session.poly_index,
        "poly_name": TEST_SUITE[session.poly_index][0],
        "root": _poly_summary(session.root_poly),
        "active": _poly_summary(active_poly) if active_poly is not None else None,
        "frontier": [_poly_summary(poly) for poly in session.state.frontier],
        "candidate_count": len(candidate_payloads),
        "candidates": candidate_payloads,
        "direct_action": _direct_summary(active_poly) if active_poly is not None else None,
        "agent_move": agent_move,
        "agent_preview": agent_preview,
        "policy_value": policy_value,
        "acc_cost": session.state.acc_cost,
        "current_circuit": session_preview["circuit"],
        "current_circuit_cost": session_preview["cost"],
        "reference_cost": reference["cost"],
        "reference_circuit": reference["circuit"],
        "history": _history_summary(session),
        "terminal": session.status in {"complete", "forfeit"},
        "k_candidates": session.k_candidates,
    }


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/test-suite")
def api_test_suite():
    items = []
    for i, (name, poly) in enumerate(TEST_SUITE):
        # OPTIMAL_CIRCUITS holds the reference circuit under the demo's gate
        # model: the only free constant is 1; every other scalar must be built
        # by adding 1s (reuse allowed), and gates are binary + and ×. The stored
        # cost equals the number of operation nodes the diagram shows.
        opt = OPTIMAL_CIRCUITS[name]
        opt_cost = opt["cost"]
        items.append({
            "index": i,
            "name": name,
            "group": poly_group(name),
            "latex": poly_to_latex(poly),
            "prime": poly.p,
            "coefficients": sorted({coeff for coeff, _exp in poly.terms}),
            "terms": [[coeff, list(exp)] for coeff, exp in poly.terms],
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
        "meta": {
            "description": "Hand-crafted heuristic (no learned weights)",
            "supported_primes": None,
            "field_label": "all finite fields",
            "field_source": "algorithmic baseline",
        },
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


@app.route("/api/play/start", methods=["POST"])
def api_play_start():
    payload = request.get_json(silent=True) or {}
    poly_index = int(payload.get("poly_index", 0))
    model_label = payload.get("model", "heuristic")
    k_candidates = max(1, min(32, int(payload.get("k", 12))))
    if poly_index < 0 or poly_index >= len(TEST_SUITE):
        return _play_error(f"Unknown polynomial index: {poly_index}")

    _name, root_poly = TEST_SUITE[poly_index]
    try:
        model, info = get_model(model_label, CHECKPOINTS_DIR)
    except FileNotFoundError as exc:
        return _play_error(str(exc), 404)

    supported_primes = info.get("supported_primes")
    if supported_primes is not None and root_poly.p not in supported_primes:
        supported = ", ".join(f"F{p}" for p in supported_primes)
        return _play_error(
            f"{model_label} supports {supported}; selected row is F{root_poly.p}."
        )

    env = DecompEnv(baseline_model=BaselineCostModel())
    state = env.reset(root_poly)
    splits = {
        info_item.active_poly.to_key(): info_item.split
        for info_item in state.history
        if info_item.split is not None
    }
    session_id = uuid.uuid4().hex
    session = PlaySession(
        session_id=session_id,
        root_poly=root_poly,
        poly_index=poly_index,
        model_label=model_label,
        model=model,
        model_info=info,
        env=env,
        state=state,
        k_candidates=k_candidates,
        splits=splits,
        history_actors=["environment"] * len(state.history),
    )
    _PLAY_SESSIONS[session_id] = session
    _cleanup_play_sessions()
    return jsonify(_play_snapshot(session))


@app.route("/api/play/step", methods=["POST"])
def api_play_step():
    payload = request.get_json(silent=True) or {}
    session_id = payload.get("session_id")
    session = _PLAY_SESSIONS.get(session_id)
    if session is None:
        return _play_error("Unknown play session.", 404)
    if session.status != "active":
        return jsonify(_play_snapshot(session))
    if not session.state.frontier:
        session.status = "complete"
        return jsonify(_play_snapshot(session))

    action_kind = payload.get("action", "candidate")
    if action_kind == "direct":
        session.state, _reward, done, _info = session.env.solve_direct(session.state, 0)
        session.history_actors.append("player")
        if done:
            session.status = "complete"
        return jsonify(_play_snapshot(session))

    if action_kind != "candidate":
        return _play_error(f"Unsupported play action: {action_kind}")

    try:
        action_index = int(payload.get("action_index"))
    except (TypeError, ValueError):
        return _play_error("Missing or invalid candidate action_index.")

    candidates = session.env.get_candidate_splits(
        session.state,
        0,
        k=session.k_candidates,
    )
    if action_index < 0 or action_index >= len(candidates):
        return _play_error(f"Candidate index out of range: {action_index}")

    action = candidates[action_index]
    active_key = session.state.frontier[0].to_key()
    session.state, _reward, done, _info = session.env.step(session.state, 0, action)
    session.splits[active_key] = action
    session.history_actors.append("player")
    if done:
        session.status = "complete"
    return jsonify(_play_snapshot(session))


@app.route("/api/play/forfeit", methods=["POST"])
def api_play_forfeit():
    payload = request.get_json(silent=True) or {}
    session_id = payload.get("session_id")
    session = _PLAY_SESSIONS.get(session_id)
    if session is None:
        return _play_error("Unknown play session.", 404)
    if session.status == "active":
        session.status = "forfeit"
    return jsonify(_play_snapshot(session))


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _tier(cost: int, optimal: int) -> int:
    """0 = matches reference (green), 1 = one op away (yellow), 2 = 2+ away (red)."""
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
            supported_primes = info.get("supported_primes")
            env = DecompEnv(baseline_model=_BASELINE_MODEL)
            search = AndOrSearch(
                baseline_model=_BASELINE_MODEL,
                model=model,
                search_config=SearchConfig(simulations=search_sims),
            )
            try:
                for i, (name, poly) in enumerate(TEST_SUITE):
                    p_start = time.perf_counter()
                    budget = search_budget_for_poly(name, search_sims, k_candidates)
                    optimal = OPTIMAL_CIRCUITS[name]["cost"]
                    if supported_primes is not None and poly.p not in supported_primes:
                        yield _sse("skipped", {
                            "label": label,
                            "poly_index": i,
                            "poly_name": name,
                            "prime": poly.p,
                            "supported_primes": supported_primes,
                            "reason": (
                                f"checkpoint field support is "
                                f"{', '.join('F' + str(p) for p in supported_primes)}; "
                                f"row is F{poly.p}"
                            ),
                        })
                        continue

                    # Greedy rollout -> materialised, binarised circuit; the cost
                    # is the new-model gate count (== rendered node count).
                    g_splits, _ = greedy_splits(
                        env, model, poly, k=int(budget["k_candidates"]))
                    greedy_circuit = binarize(build_agent_circuit(
                        poly, g_splits, env.factorizer, _BASELINE_MODEL))
                    gcost = gate_count(greedy_circuit)

                    # Guided search (best_trace -> splits -> circuit).
                    search.search_config = SearchConfig(
                        simulations=int(budget["search_sims"]),
                        max_depth=int(budget["max_depth"]),
                        expand_top_k=int(budget["expand_top_k"]),
                    )
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
                        # Tier vs reference: 0 = match, 1 = one op away, 2 = 2+.
                        "greedy_tier": _tier(gcost, optimal),
                        "search_tier": _tier(scost, optimal),
                        "greedy_circuit": greedy_circuit,
                        "search_circuit": search_circuit,
                        "greedy_circuit_cost": gcost,
                        "search_circuit_cost": scost,
                        "node_expansions": sr.stats.node_expansions,
                        "transposition_hits": sr.stats.transposition_hits,
                        "search_sims_used": budget["search_sims"],
                        "k_candidates_used": budget["k_candidates"],
                        "search_max_depth": budget["max_depth"],
                        "search_expand_top_k": budget["expand_top_k"],
                        "budget_capped": budget["capped"],
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
    global CHECKPOINTS_DIR

    parser = argparse.ArgumentParser(
        description="Local web UI for arithmetic-circuit RL inference."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=CHECKPOINTS_DIR,
        help="Directory containing *.pt checkpoints to expose in the UI.",
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    CHECKPOINTS_DIR = args.checkpoint_dir.expanduser().resolve()

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
