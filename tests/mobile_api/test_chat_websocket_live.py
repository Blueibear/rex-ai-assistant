"""Live loopback WebSocket integration through Flask-Sock/simple-websocket."""

from __future__ import annotations

import json
import threading
import uuid

import pytest
import requests
from simple_websocket import Client, ConnectionClosed
from werkzeug.serving import make_server

from rex.mobile_api.device_proof import (
    canonical_session_transcript,
    canonical_transcript,
    generate_p256_private_key,
    public_key_spki_b64,
    sign_transcript,
)
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


def _paired_login(base_url: str, services, user_id: str) -> str:
    response = requests.post(
        f"{base_url}/mobile/auth/login",
        json={
            "username": "james",
            "password": "pw-123456",  # pragma: allowlist secret
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
    bootstrap = response.json()
    private_key = generate_p256_private_key()
    public_key = public_key_spki_b64(private_key)
    challenge = services.pairing_authority.create_challenge(
        user_id=user_id,
        scopes=["chat.send", "chat.history.read", "voice.use"],
    )
    transcript = canonical_transcript(
        desktop_id=challenge.desktop_id,
        challenge_id=challenge.challenge_id,
        nonce_b64=challenge.nonce_b64,
        mobile_public_key_b64=public_key,
        user_id=challenge.user_id,
        scopes=challenge.scopes,
        code=challenge.code,
        server_url=challenge.server_url,
        certificate_fingerprint=challenge.certificate_fingerprint,
        spki_pins=challenge.spki_pins,
    )
    submitted = services.pairing_authority.submit_proof(
        {
            "desktop_id": challenge.desktop_id,
            "challenge_id": challenge.challenge_id,
            "nonce": challenge.nonce_b64,
            "code": challenge.code,
            "user_id": challenge.user_id,
            "scopes": list(challenge.scopes),
            "public_key": public_key,
            "signature": sign_transcript(private_key, transcript),
            "device_name": "Loopback iPhone",
            "platform": "ios",
        }
    )
    grant = services.pairing_authority.approve(
        submitted.request_id,
        approved_by=r"TEST\DesktopOwner",
    )
    headers = {"Authorization": f"Bearer {bootstrap['access_token']}"}
    challenge_response = requests.post(
        f"{base_url}/mobile/auth/device-challenge",
        json={"device_id": grant.device_id, "grant_id": grant.grant_id},
        headers=headers,
        timeout=5,
    )
    challenge_response.raise_for_status()
    activation = challenge_response.json()
    activation_transcript = canonical_session_transcript(
        desktop_id=activation["desktop_id"],
        bootstrap_session_id=activation["bootstrap_session_id"],
        challenge_id=activation["challenge_id"],
        nonce_b64=activation["nonce"],
        device_id=activation["device_id"],
        grant_id=activation["grant_id"],
        grant_version=activation["grant_version"],
        user_id=activation["user_id"],
    )
    activated = requests.post(
        f"{base_url}/mobile/auth/activate-device",
        json={
            "challenge_id": activation["challenge_id"],
            "signature": sign_transcript(private_key, activation_transcript),
        },
        headers=headers,
        timeout=5,
    )
    activated.raise_for_status()
    return str(activated.json()["access_token"])


def _ws_url(base_url: str) -> str:
    return base_url.replace("http://", "ws://") + "/mobile/chat/stream"


def _receive_json(ws: Client, timeout: float = 3) -> dict:
    return json.loads(ws.receive(timeout=timeout))


def test_live_upgrade_auth_chat_dedupe_close_codes_and_revocation(
    live_server, services, fake_chat_service, monkeypatch
) -> None:
    """Actual upgrade/auth/chat plus HTTP dedupe and security close paths."""
    user_id = create_user("james", "pw-123456")
    token = _paired_login(live_server, services, user_id)
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

    live_token = _paired_login(live_server, services, user_id)
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
