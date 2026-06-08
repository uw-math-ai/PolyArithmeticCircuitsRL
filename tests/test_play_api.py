import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import serve_ui  # noqa: E402


def test_play_api_starts_after_root_factorization_and_can_finish():
    serve_ui._PLAY_SESSIONS.clear()
    client = serve_ui.app.test_client()

    started = client.post(
        "/api/play/start",
        json={"poly_index": 1, "model": "heuristic", "k": 8},
    )
    assert started.status_code == 200
    data = started.get_json()

    assert data["status"] == "active"
    assert data["acc_cost"] == 2
    assert data["history"][0]["actor"] == "environment"
    assert data["history"][0]["action_kind"] == "factor"
    assert data["active"]["latex"] == "y + x"
    assert data["agent_move"]["kind"] in {"split", "direct", "factor"}

    stepped = client.post(
        "/api/play/step",
        json={
            "session_id": data["session_id"],
            "action": "candidate",
            "action_index": data["agent_move"]["index"],
        },
    )
    assert stepped.status_code == 200
    done = stepped.get_json()
    assert done["status"] == "complete"
    assert done["acc_cost"] == 3
    assert done["frontier"] == []
    assert done["terminal"] is True


def test_play_api_forfeit_marks_active_session_terminal():
    serve_ui._PLAY_SESSIONS.clear()
    client = serve_ui.app.test_client()

    started = client.post(
        "/api/play/start",
        json={"poly_index": 0, "model": "heuristic", "k": 8},
    )
    assert started.status_code == 200
    session_id = started.get_json()["session_id"]

    forfeited = client.post("/api/play/forfeit", json={"session_id": session_id})
    assert forfeited.status_code == 200
    data = forfeited.get_json()
    assert data["status"] == "forfeit"
    assert data["terminal"] is True
