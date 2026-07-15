"""Live loopback WebSocket integration through Flask-Sock/simple-websocket."""

from __future__ import annotations

import json
import threading
import uuid

import pytest
import requests
from simple_websocket import Client, ConnectionClosed
from werkzeug.serving import make_server

from tests.mobile_api.conftest import chat_payload, create_user


def _auth_frame(token: str) -> dict:
    return {
        "type": "auth",
        "access_token": token,
        "client": {"platform": "ios", "app_version": "0.1.0", "device_id": "live-dev-1"},
    }


@pytest.fixture()
def live_server(app):
    server = make_server("127.0.0.1", 0, app, threaded=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join(timeout=5)


def _login(base_url: str) -> str:
    response = requests.post(
        f"{base_url}/mobile/auth/login",
        json={
            "username": "james",
            "password": "pw-123456",
            "device": {
                "device_id": str(uuid.uuid4()),
                "name": "loopback",
                "platform": "ios",
                "app_version": "0.1.0",
            },
        },
        timeout=5,
    )
    response.raise_for_status()
    return str(response.json()["access_token"])


def _ws_url(base_url: str) -> str:
    return base_url.replace("http://", "ws://") + "/mobile/chat/stream"


def _receive_json(ws: Client, timeout: float = 3) -> dict:
    return json.loads(ws.receive(timeout=timeout))


def test_live_upgrade_auth_chat_dedupe_close_codes_and_revocation(
    live_server, fake_chat_service, monkeypatch
) -> None:
    """Actual upgrade/auth/chat plus HTTP dedupe and security close paths."""
    create_user("james", "pw-123456")
    token = _login(live_server)
    url = _ws_url(live_server)
    assert "token" not in url.lower()

    ws = Client.connect(url)
    ws.send(json.dumps(_auth_frame(token)))
    assert _receive_json(ws)["type"] == "auth_ok"

    payload = {"type": "chat", **chat_payload("live loopback")}
    ws.send(json.dumps(payload))
    events = []
    while not events or events[-1]["type"] != "message_done":
        events.append(_receive_json(ws))
    assert events[0]["type"] == "ack"
    assert any(event["type"] == "token" for event in events)
    assert events[-1]["message_id"] == payload["message_id"]

    # Same ID through authenticated HTTP returns the stored result and does
    # not execute the Assistant a second time.
    http_body = {key: value for key, value in payload.items() if key != "type"}
    replay = requests.post(
        f"{live_server}/mobile/chat",
        json=http_body,
        headers={
            "Authorization": f"Bearer {token}",
            "Idempotency-Key": payload["message_id"],
        },
        timeout=5,
    )
    assert replay.status_code == 200
    assert replay.json()["message_id"] == payload["message_id"]
    assert len(fake_chat_service.calls) == 1
    assert fake_chat_service.calls[0][0] == "live loopback"
    ws.close()

    invalid = Client.connect(url)
    invalid.send(json.dumps(_auth_frame("invalid-token")))
    assert _receive_json(invalid)["type"] == "auth_error"
    with pytest.raises(ConnectionClosed) as invalid_close:
        invalid.receive(timeout=3)
    assert int(invalid_close.value.reason) == 4401

    monkeypatch.setattr("rex.mobile_api.websocket.AUTH_TIMEOUT_SECONDS", 0.1)
    timed_out = Client.connect(url)
    with pytest.raises(ConnectionClosed) as timeout_close:
        timed_out.receive(timeout=3)
    assert int(timeout_close.value.reason) == 4408

    live_token = _login(live_server)
    revoked = Client.connect(url)
    revoked.send(json.dumps(_auth_frame(live_token)))
    assert _receive_json(revoked)["type"] == "auth_ok"
    logout = requests.post(
        f"{live_server}/mobile/auth/logout",
        headers={"Authorization": f"Bearer {live_token}"},
        timeout=5,
    )
    assert logout.status_code == 200
    revoked.send(json.dumps({"type": "chat", **chat_payload("must not run")}))
    with pytest.raises(ConnectionClosed) as revoked_close:
        revoked.receive(timeout=3)
    assert int(revoked_close.value.reason) == 4401
    assert len(fake_chat_service.calls) == 1
