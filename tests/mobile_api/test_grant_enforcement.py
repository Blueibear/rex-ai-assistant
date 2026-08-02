"""Adversarial device-session grant enforcement tests (S6)."""

from __future__ import annotations

import json

from rex.mobile_api.device_proof import (
    canonical_session_transcript,
    canonical_transcript,
    generate_p256_private_key,
    public_key_spki_b64,
    sign_transcript,
)
from rex.mobile_api.pairing import PairingError
from rex.mobile_api.websocket import CLOSE_UNAUTHENTICATED, MobileWebSocketServer
from tests.mobile_api.conftest import (
    auth_header,
    chat_payload,
    create_user,
    login_tokens,
    paired_login_tokens,
    parse_sse_events,
)

SCOPES = ["chat.send", "chat.history.read", "voice.use"]


def _approved_device(services, user_id: str, *, private_key=None, scopes=None):
    private_key = private_key or generate_p256_private_key()
    scopes = scopes or SCOPES
    public_key = public_key_spki_b64(private_key)
    challenge = services.pairing_authority.create_challenge(user_id=user_id, scopes=scopes)
    transcript = canonical_transcript(
        desktop_id=challenge.desktop_id,
        challenge_id=challenge.challenge_id,
        nonce_b64=challenge.nonce_b64,
        mobile_public_key_b64=public_key,
        user_id=challenge.user_id,
        scopes=challenge.scopes,
        code=challenge.code,
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
            "device_name": "Test iPhone",
            "platform": "ios",
        }
    )
    grant = services.pairing_authority.approve(
        submitted.request_id, approved_by="TEST\\DesktopOwner"
    )
    return private_key, grant


def _activation_challenge(client, bootstrap: dict, grant) -> dict:
    response = client.post(
        "/mobile/auth/device-challenge",
        json={"device_id": grant.device_id, "grant_id": grant.grant_id},
        headers=auth_header(bootstrap["access_token"]),
    )
    assert response.status_code == 200, response.get_json()
    return response.get_json()


def _activation_signature(private_key, challenge: dict) -> str:
    transcript = canonical_session_transcript(
        desktop_id=challenge["desktop_id"],
        bootstrap_session_id=challenge["bootstrap_session_id"],
        challenge_id=challenge["challenge_id"],
        nonce_b64=challenge["nonce"],
        device_id=challenge["device_id"],
        grant_id=challenge["grant_id"],
        grant_version=challenge["grant_version"],
        user_id=challenge["user_id"],
    )
    return sign_transcript(private_key, transcript)


def test_password_session_is_bootstrap_only_and_client_scopes_are_ignored(client, services):
    create_user("bootstrap-only", "pw-123456")
    response = client.post(
        "/mobile/auth/login",
        json={
            "username": "bootstrap-only",
            "password": "pw-123456",  # pragma: allowlist secret
            "scopes": ["chat.send", "home.control"],
            "device": {"device_id": "client-claimed-device"},
        },
    )
    assert response.status_code == 200
    tokens = response.get_json()
    session = client.get("/mobile/auth/session", headers=auth_header(tokens["access_token"]))
    assert session.status_code == 200
    assert session.get_json()["paired"] is False
    assert session.get_json()["scopes"] == []
    denied = client.post(
        "/mobile/chat",
        json=chat_payload(),
        headers=auth_header(tokens["access_token"]),
    )
    assert denied.status_code == 403
    assert denied.get_json()["error"]["code"] == "FORBIDDEN"


def test_activation_binds_server_grant_and_revokes_bootstrap_family(client, services):
    user_id = create_user("activate", "pw-123456")
    bootstrap = login_tokens(client, "activate", "pw-123456")
    private_key, grant = _approved_device(services, user_id)
    challenge = _activation_challenge(client, bootstrap, grant)
    activated = client.post(
        "/mobile/auth/activate-device",
        json={
            "challenge_id": challenge["challenge_id"],
            "signature": _activation_signature(private_key, challenge),
        },
        headers=auth_header(bootstrap["access_token"]),
    )
    assert activated.status_code == 200
    body = activated.get_json()
    assert body["session_id"] != bootstrap["session_id"]
    old_access = client.get("/mobile/auth/session", headers=auth_header(bootstrap["access_token"]))
    assert old_access.status_code == 401
    old_refresh = client.post(
        "/mobile/auth/refresh", json={"refresh_token": bootstrap["refresh_token"]}
    )
    assert old_refresh.status_code == 401
    current = client.get(
        "/mobile/auth/session", headers=auth_header(body["access_token"])
    ).get_json()
    assert current["paired"] is True
    assert current["device_id"] == grant.device_id
    assert current["grant_id"] == grant.grant_id
    assert current["grant_version"] == 1
    assert current["scopes"] == sorted(SCOPES)
    assert current["strong_auth_at"] is None


def test_activation_wrong_key_expiry_and_replay_fail_closed(client, services, clock):
    user_id = create_user("activation-guards", "pw-123456")
    bootstrap = login_tokens(client, "activation-guards", "pw-123456")
    private_key, grant = _approved_device(services, user_id)
    challenge = _activation_challenge(client, bootstrap, grant)
    wrong = client.post(
        "/mobile/auth/activate-device",
        json={
            "challenge_id": challenge["challenge_id"],
            "signature": _activation_signature(generate_p256_private_key(), challenge),
        },
        headers=auth_header(bootstrap["access_token"]),
    )
    assert wrong.status_code == 403
    good = client.post(
        "/mobile/auth/activate-device",
        json={
            "challenge_id": challenge["challenge_id"],
            "signature": _activation_signature(private_key, challenge),
        },
        headers=auth_header(bootstrap["access_token"]),
    )
    assert good.status_code == 200
    replay = client.post(
        "/mobile/auth/activate-device",
        json={
            "challenge_id": challenge["challenge_id"],
            "signature": _activation_signature(private_key, challenge),
        },
        headers=auth_header(good.get_json()["access_token"]),
    )
    assert replay.status_code == 403

    create_user("activation-expired", "pw-123456")
    expired_bootstrap = login_tokens(client, "activation-expired", "pw-123456")
    from rex.mobile_api.db import connect

    conn = connect(services.db_path)
    try:
        expired_user = conn.execute(
            "SELECT id FROM users WHERE username = ?", ("activation-expired",)
        ).fetchone()["id"]
    finally:
        conn.close()
    expired_key, expired_grant = _approved_device(services, str(expired_user))
    expired_challenge = _activation_challenge(client, expired_bootstrap, expired_grant)
    clock.advance(seconds=121)
    expired = client.post(
        "/mobile/auth/activate-device",
        json={
            "challenge_id": expired_challenge["challenge_id"],
            "signature": _activation_signature(expired_key, expired_challenge),
        },
        headers=auth_header(expired_bootstrap["access_token"]),
    )
    assert expired.status_code == 403


def test_cross_user_device_grant_cannot_be_claimed(client, services):
    alice_id = create_user("grant-alice", "pw-123456")
    create_user("grant-bob", "pw-123456")
    bob_bootstrap = login_tokens(client, "grant-bob", "pw-123456")
    _key, alice_grant = _approved_device(services, alice_id)
    response = client.post(
        "/mobile/auth/device-challenge",
        json={"device_id": alice_grant.device_id, "grant_id": alice_grant.grant_id},
        headers=auth_header(bob_bootstrap["access_token"]),
    )
    assert response.status_code == 403


def test_device_revoke_immediately_denies_http_and_refresh(client, services):
    create_user("revoke-live", "pw-123456")
    tokens = paired_login_tokens(client, "revoke-live", "pw-123456")
    assert services.pairing_authority.revoke_device(
        tokens["_paired_device_id"],
        revoked_by="TEST\\DesktopOwner",
        reason="owner_revoked",
    )
    denied = client.post(
        "/mobile/chat",
        json=chat_payload(),
        headers=auth_header(tokens["access_token"]),
    )
    assert denied.status_code == 401
    refreshed = client.post("/mobile/auth/refresh", json={"refresh_token": tokens["refresh_token"]})
    assert refreshed.status_code == 401


def test_existing_device_key_cannot_be_reassigned_to_another_user(client, services):
    alice_id = create_user("device-owner-alice", "pw-123456")
    bob_id = create_user("device-owner-bob", "pw-123456")
    private_key, _grant = _approved_device(services, alice_id)
    import pytest

    with pytest.raises(PairingError, match="cannot be reassigned"):
        _approved_device(services, bob_id, private_key=private_key)


def test_new_grant_version_revokes_sessions_bound_to_old_version(client, services):
    create_user("grant-version", "pw-123456")
    tokens = paired_login_tokens(client, "grant-version", "pw-123456")
    from rex.mobile_api.db import connect

    conn = connect(services.db_path)
    try:
        user_id = str(
            conn.execute("SELECT id FROM users WHERE username = ?", ("grant-version",)).fetchone()[
                "id"
            ]
        )
    finally:
        conn.close()
    _same_key, replacement = _approved_device(
        services,
        user_id,
        private_key=tokens["_private_key"],
        scopes=["chat.history.read"],
    )
    assert replacement.device_id == tokens["_paired_device_id"]
    assert replacement.version == 2
    denied = client.post(
        "/mobile/chat",
        json=chat_payload(),
        headers=auth_header(tokens["access_token"]),
    )
    assert denied.status_code == 401


def test_sse_stops_before_emitting_chunk_produced_after_revoke(client, services, fake_chat_service):
    create_user("sse-revoke", "pw-123456")
    tokens = paired_login_tokens(client, "sse-revoke", "pw-123456")

    def revoking_stream(
        message: str,
        *,
        user_id: str,
        capability_scopes: frozenset[str],
        capability_permissions: frozenset[str],
        authorization_check,
    ):
        assert "chat.send" in capability_scopes
        assert capability_permissions == frozenset()
        authorization_check()
        yield "first"
        services.pairing_authority.revoke_device(
            tokens["_paired_device_id"],
            revoked_by="TEST\\DesktopOwner",
            reason="test_midstream_revoke",
        )
        yield "must-not-emit"

    fake_chat_service.stream = revoking_stream
    response = client.post(
        "/mobile/chat/stream",
        json=chat_payload("revoke during stream"),
        headers=auth_header(tokens["access_token"]),
    )
    events = parse_sse_events(response.data)
    contents = [event.get("content") for event in events if event["type"] == "token"]
    assert contents == ["first"]
    assert all("must-not-emit" not in json.dumps(event) for event in events)
    assert events[-1]["type"] == "error"


class _FakeWs:
    def __init__(self, frames: list[str]) -> None:
        self.incoming = list(frames)
        self.sent: list[dict] = []
        self.closed: tuple[int, str] | None = None

    def receive(self, timeout: float | None = None):
        if self.closed is not None or not self.incoming:
            raise RuntimeError("closed")
        return self.incoming.pop(0)

    def send(self, data: str) -> None:
        if self.closed is not None:
            raise RuntimeError("closed")
        self.sent.append(json.loads(data))

    def close(self, code: int = 1000, reason: str = "") -> None:
        self.closed = (code, reason)


def _ws_auth(token: str) -> str:
    return json.dumps(
        {
            "type": "auth",
            "access_token": token,
            "client": {"platform": "ios", "app_version": "0.1.0", "device_id": "dev-1"},
        }
    )


def test_websocket_stops_output_and_closes_after_midstream_revoke(
    client, services, fake_chat_service
):
    create_user("ws-revoke", "pw-123456")
    tokens = paired_login_tokens(client, "ws-revoke", "pw-123456")

    def revoking_stream(
        message: str,
        *,
        user_id: str,
        capability_scopes: frozenset[str],
        capability_permissions: frozenset[str],
        authorization_check,
    ):
        assert "chat.send" in capability_scopes
        assert capability_permissions == frozenset()
        authorization_check()
        yield "first"
        services.pairing_authority.revoke_device(
            tokens["_paired_device_id"],
            revoked_by="TEST\\DesktopOwner",
            reason="test_ws_revoke",
        )
        yield "must-not-emit"

    fake_chat_service.stream = revoking_stream
    frame = json.dumps({"type": "chat", **chat_payload("ws revoke")})
    ws = _FakeWs([_ws_auth(tokens["access_token"]), frame])
    MobileWebSocketServer(services).handle(ws, "10.0.0.1")
    contents = [event.get("content") for event in ws.sent if event["type"] == "token"]
    assert contents == ["first"]
    assert all("must-not-emit" not in json.dumps(event) for event in ws.sent)
    assert ws.closed is not None
    assert ws.closed[0] == CLOSE_UNAUTHENTICATED


def test_device_scope_is_intersected_with_live_user_permission(client, services):
    import pytest

    from rex.mobile_api.auth import authenticate_token, revalidate_principal
    from rex.mobile_api.errors import MobileApiError
    from rex.permissions import grant_permission, revoke_permission

    user_id = create_user("home-permission", "pw-123456")
    tokens = paired_login_tokens(
        client,
        "home-permission",
        "pw-123456",
        scopes=["home.control"],
    )

    with pytest.raises(MobileApiError) as denied:
        authenticate_token(
            services,
            tokens["access_token"],
            required_scope="home.control",
        )
    assert denied.value.http_status == 403

    grant_permission(user_id, "ha_control")
    principal = authenticate_token(
        services,
        tokens["access_token"],
        required_scope="home.control",
    )
    assert "ha_control" in principal.permissions

    revoke_permission(user_id, "ha_control")
    with pytest.raises(MobileApiError) as revoked:
        revalidate_principal(
            services,
            principal,
            required_scope="home.control",
        )
    assert revoked.value.http_status == 401
    assert revoked.value.code == "AUTH_SESSION_REVOKED"
