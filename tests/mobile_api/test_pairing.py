"""Desktop-owned pairing authority tests (S5)."""

from __future__ import annotations

import sqlite3

import pytest

from rex.mobile_api.device_proof import (
    canonical_transcript,
    generate_p256_private_key,
    public_key_spki_b64,
    sign_transcript,
)
from rex.mobile_api.pairing import PairingAuthority, PairingError

SCOPES = ["chat.send", "chat.history.read", "voice.use"]


def _authority(services, *, code: str = "12345678") -> PairingAuthority:
    authority = PairingAuthority(
        services.db_path,
        clock=services.clock,
        code_generator=lambda: code,
    )
    services.pairing_authority = authority
    return authority


def _payload(challenge, private_key, **updates):
    public_key = public_key_spki_b64(private_key)
    transcript = canonical_transcript(
        desktop_id=challenge.desktop_id,
        challenge_id=challenge.challenge_id,
        nonce_b64=challenge.nonce_b64,
        mobile_public_key_b64=public_key,
        user_id=challenge.user_id,
        scopes=challenge.scopes,
        code=challenge.code,
    )
    payload = {
        "desktop_id": challenge.desktop_id,
        "challenge_id": challenge.challenge_id,
        "nonce": challenge.nonce_b64,
        "code": challenge.code,
        "user_id": challenge.user_id,
        "scopes": list(challenge.scopes),
        "public_key": public_key,
        "signature": sign_transcript(private_key, transcript),
        "device_name": "James's iPhone",
        "platform": "ios",
    }
    payload.update(updates)
    return payload


def _count(db_path, table: str) -> int:
    conn = sqlite3.connect(db_path)
    try:
        return int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
    finally:
        conn.close()


def test_password_login_does_not_create_device_or_grant(client, services, mobile_env):
    from tests.mobile_api.conftest import create_user, login

    create_user("pair-login", "correct horse battery staple")
    response = login(client, "pair-login", "correct horse battery staple")
    assert response.status_code == 200
    assert _count(services.db_path, "mobile_paired_devices") == 0
    assert _count(services.db_path, "mobile_device_grants") == 0


def test_submit_requires_explicit_desktop_approval(client, services, mobile_env):
    from tests.mobile_api.conftest import create_user

    user_id = create_user("pair-user", "correct horse battery staple")
    authority = _authority(services)
    challenge = authority.create_challenge(user_id=user_id, scopes=SCOPES)
    private_key = generate_p256_private_key()

    response = client.post("/mobile/pairing/submit", json=_payload(challenge, private_key))
    assert response.status_code == 202
    submitted = response.get_json()
    assert submitted["status"] == "pending"
    assert _count(services.db_path, "mobile_device_grants") == 0

    pending = authority.list_pending()
    assert pending[0]["request_id"] == submitted["request_id"]
    grant = authority.approve(submitted["request_id"], approved_by="DESKTOP\\James")
    assert grant.scopes == tuple(sorted(SCOPES))
    assert grant.version == 1

    status = client.post(
        "/mobile/pairing/status",
        json={"request_id": submitted["request_id"], "poll_token": submitted["poll_token"]},
    )
    assert status.status_code == 200
    body = status.get_json()
    assert body["status"] == "approved"
    assert body["grant_id"] == grant.grant_id
    assert body["scopes"] == list(grant.scopes)


def test_challenge_is_single_use(client, services, mobile_env):
    from tests.mobile_api.conftest import create_user

    user_id = create_user("pair-replay", "correct horse battery staple")
    authority = _authority(services)
    challenge = authority.create_challenge(user_id=user_id, scopes=SCOPES)
    payload = _payload(challenge, generate_p256_private_key())
    assert client.post("/mobile/pairing/submit", json=payload).status_code == 202
    replay = client.post("/mobile/pairing/submit", json=payload)
    assert replay.status_code == 400
    assert replay.get_json()["error"]["code"] == "PAIRING_INVALID"
    assert "already been used" in replay.get_json()["error"]["message"]


def test_expired_challenge_fails_closed(client, services, clock, mobile_env):
    from tests.mobile_api.conftest import create_user

    user_id = create_user("pair-expired", "correct horse battery staple")
    authority = _authority(services)
    challenge = authority.create_challenge(user_id=user_id, scopes=SCOPES)
    clock.advance(seconds=121)
    response = client.post(
        "/mobile/pairing/submit",
        json=_payload(challenge, generate_p256_private_key()),
    )
    assert response.status_code == 400
    assert "expired" in response.get_json()["error"]["message"]
    assert _count(services.db_path, "mobile_pairing_requests") == 0


def test_wrong_desktop_and_scope_tampering_fail_closed(client, services, mobile_env):
    from tests.mobile_api.conftest import create_user

    user_id = create_user("pair-binding", "correct horse battery staple")
    authority = _authority(services)
    challenge = authority.create_challenge(user_id=user_id, scopes=SCOPES)
    private_key = generate_p256_private_key()

    wrong_desktop = client.post(
        "/mobile/pairing/submit",
        json=_payload(challenge, private_key, desktop_id="wrong-desktop"),
    )
    assert wrong_desktop.status_code == 400
    assert "does not match" in wrong_desktop.get_json()["error"]["message"]

    tampered = client.post(
        "/mobile/pairing/submit",
        json=_payload(challenge, private_key, scopes=["chat.send"]),
    )
    assert tampered.status_code == 400
    assert "does not match" in tampered.get_json()["error"]["message"]


def test_key_mismatch_fails_proof_without_consuming_challenge(client, services, mobile_env):
    from tests.mobile_api.conftest import create_user

    user_id = create_user("pair-key", "correct horse battery staple")
    authority = _authority(services)
    challenge = authority.create_challenge(user_id=user_id, scopes=SCOPES)
    signed_by = generate_p256_private_key()
    claimed = generate_p256_private_key()
    payload = _payload(challenge, signed_by)
    payload["public_key"] = public_key_spki_b64(claimed)

    bad = client.post("/mobile/pairing/submit", json=payload)
    assert bad.status_code == 400
    assert "proof" in bad.get_json()["error"]["message"].lower()

    good = client.post(
        "/mobile/pairing/submit",
        json=_payload(challenge, signed_by),
    )
    assert good.status_code == 202


def test_unknown_scope_and_malformed_fields_are_rejected(services, mobile_env):
    from tests.mobile_api.conftest import create_user

    user_id = create_user("pair-scope", "correct horse battery staple")
    authority = _authority(services)
    with pytest.raises(PairingError, match="not permitted"):
        authority.create_challenge(user_id=user_id, scopes=["desktop.shell.root"])


def test_denial_and_revocation_are_terminal(client, services, mobile_env):
    from tests.mobile_api.conftest import create_user

    user_id = create_user("pair-revoke", "correct horse battery staple")
    authority = _authority(services)
    first = authority.create_challenge(user_id=user_id, scopes=SCOPES)
    submitted = authority.submit_proof(_payload(first, generate_p256_private_key()))
    authority.deny(submitted.request_id, denied_by="DESKTOP\\James")
    assert authority.poll_status(submitted.request_id, submitted.poll_token)["status"] == "denied"
    with pytest.raises(PairingError, match="not pending"):
        authority.approve(submitted.request_id, approved_by="DESKTOP\\James")

    second = authority.create_challenge(user_id=user_id, scopes=SCOPES)
    approved = authority.submit_proof(_payload(second, generate_p256_private_key()))
    grant = authority.approve(approved.request_id, approved_by="DESKTOP\\James")
    assert authority.revoke_device(
        grant.device_id, revoked_by="DESKTOP\\James", reason="owner_revoked"
    )
    device = next(item for item in authority.list_devices() if item["device_id"] == grant.device_id)
    assert device["revoked_at"] is not None
    assert device["grant_revoked_at"] is not None


def test_invalid_poll_token_does_not_reveal_request(client, services, mobile_env):
    from tests.mobile_api.conftest import create_user

    user_id = create_user("pair-poll", "correct horse battery staple")
    authority = _authority(services)
    challenge = authority.create_challenge(user_id=user_id, scopes=SCOPES)
    submitted = authority.submit_proof(_payload(challenge, generate_p256_private_key()))
    response = client.post(
        "/mobile/pairing/status",
        json={"request_id": submitted.request_id, "poll_token": "wrong-token"},
    )
    assert response.status_code == 401
    assert response.get_json()["error"]["code"] == "PAIRING_INVALID"
