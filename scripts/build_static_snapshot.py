#!/usr/bin/env python3
"""
Bake the live inference demo into a static snapshot for GitHub Pages.

The interactive demo (``scripts/serve_ui.py``) needs a running Flask server to
perform torch inference and MCTS search. GitHub Pages only serves static files,
so this script drives the very same Flask app in-process via its test client,
captures the JSON that each ``/api/*`` endpoint returns, and writes those
payloads to ``docs/data/*.json``. A small client-side shim
(``docs/static-shim.js``) then replays these payloads so the unmodified
frontend runs entirely in the browser.

Because we call the real endpoints, the captured JSON is byte-for-byte the same
shape the frontend already consumes -- no schema drift.

Usage:
    python scripts/build_static_snapshot.py
    python scripts/build_static_snapshot.py --checkpoint-dir models --search-sims 32
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))


def _parse_sse(raw: str) -> list[dict]:
    """Parse a Server-Sent-Events stream into ``[{event, data}, ...]``.

    Each SSE record is ``event: <name>\\ndata: <json>\\n\\n``; the serve_ui
    ``_sse`` helper always emits exactly that pair, so a simple block split is
    sufficient (no multi-line data payloads to reassemble).
    """
    events: list[dict] = []
    for block in raw.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        event_name = None
        data_json = None
        for line in block.splitlines():
            if line.startswith("event:"):
                event_name = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data_json = line[len("data:"):].strip()
        if event_name is None:
            continue
        events.append({
            "event": event_name,
            "data": json.loads(data_json) if data_json else {},
        })
    return events


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=_REPO_ROOT / "models",
        help="Directory of *.pt checkpoints to expose (default: models/).",
    )
    parser.add_argument(
        "--search-sims",
        type=int,
        default=32,
        help="Search simulations to bake into the evaluate snapshot.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=_REPO_ROOT / "docs" / "data",
        help="Output directory for the JSON snapshot (default: docs/data/).",
    )
    args = parser.parse_args()

    # Point serve_ui at the requested checkpoint dir *before* using its app.
    import serve_ui

    serve_ui.CHECKPOINTS_DIR = args.checkpoint_dir.expanduser().resolve()
    client = serve_ui.app.test_client()

    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    print(f"Checkpoint dir : {serve_ui.CHECKPOINTS_DIR}")
    print(f"Output dir     : {out}")

    # 1. Test suite (fast, no inference).
    print("Capturing /api/test-suite ...", flush=True)
    suite = client.get("/api/test-suite").get_json()
    (out / "test-suite.json").write_text(json.dumps(suite, indent=2))
    print(f"  {len(suite['polynomials'])} polynomials")

    # 2. Models (fast, peeks checkpoint metadata).
    print("Capturing /api/models ...", flush=True)
    models = client.get("/api/models").get_json()
    (out / "models.json").write_text(json.dumps(models, indent=2))
    labels = [m["label"] for m in models["models"]]
    print(f"  models: {', '.join(labels)}")

    # 3. Evaluate every model over the whole suite (slow: torch + MCTS).
    print(f"Capturing /api/evaluate (search_sims={args.search_sims}) ...", flush=True)
    query = f"/api/evaluate?search_sims={args.search_sims}"
    for label in labels:
        query += f"&models={label}"
    resp = client.get(query)
    raw = resp.get_data(as_text=True)
    events = _parse_sse(raw)
    n_results = sum(1 for e in events if e["event"] == "result")
    n_skipped = sum(1 for e in events if e["event"] == "skipped")
    (out / "evaluate.json").write_text(json.dumps({
        "search_sims": args.search_sims,
        "events": events,
    }, indent=2))
    print(f"  {len(events)} events ({n_results} results, {n_skipped} skipped)")

    # Small manifest so the shim / debugging knows what was baked.
    (out / "manifest.json").write_text(json.dumps({
        "checkpoint_dir": str(serve_ui.CHECKPOINTS_DIR),
        "search_sims": args.search_sims,
        "models": labels,
        "num_polynomials": len(suite["polynomials"]),
    }, indent=2))

    print("Done. Snapshot written to", out)


if __name__ == "__main__":
    main()
