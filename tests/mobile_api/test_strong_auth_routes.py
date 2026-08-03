from __future__ import annotations

from rex.mobile_api.device_proof import sign_transcript
from rex.mobile_api.strong_auth import StrongAuthChallenge, canonical_strong_auth_transcript
from tests.mobile_api.conftest import auth_header, create_user, paired_login_tokens


def _action() -> dict:
    return {
        "domain": "light",
        "service": "turn_off",
        "entity_id": "light.downstairs",
        "data": {"transition": 1},
    }


def _challenge_from_response(body: dict) -> StrongAuthChallenge:
    return StrongAuthChallenge(
        challenge_id=body["challenge_id"],
        action_name=body["action_name"],
        action_hash=body["action_hash"],
        risk_level=body["risk_level"],
        required_scope=body["required_scope"],
        nonce_b64=body["nonce"],
        desktop_id=body["desktop_id"],
        session_id=body["session_id"],
        user_id=body["user_id"],
        device_id=body["device_id"],
        grant_id=body["grant_id"],
        grant_version=body["grant_version"],
        expires_at=body["expires_at"],
    )


def test_strong_auth_routes_require_authentication(client):
    response = client.post(
        "/mobile/auth/strong-auth/challenge",
        json={"action_name": "home_assistant_call_service", "action": _action()},
    )
    assert response.status_code == 401
    assert response.get_json()["error"]["code"] == "AUTH_TOKEN_INVALID"


def test_challenge_and_verify_issue_single_use_action_approval(client):
    create_user("james", "correct-horse", admin=True)
    tokens = paired_login_tokens(
        client,
        "james",
        "correct-horse",
        scopes=["chat.send", "home.control"],
    )
    headers = auth_header(tokens["access_token"])
    response = client.post(
        "/mobile/auth/strong-auth/challenge",
        json={"action_name": "home_assistant_call_service", "action": _action()},
        headers=headers,
    )
    assert response.status_code == 200, response.get_json()
    body = response.get_json()
    assert body["risk_level"] == "high"
    assert body["required_scope"] == "home.control"
    assert body["single_use"] if "single_use" in body else True

    challenge = _challenge_from_response(body)
    signature = sign_transcript(
        tokens["_private_key"],
        canonical_strong_auth_transcript(challenge),
    )
    verified = client.post(
        "/mobile/auth/strong-auth/verify",
        json={"challenge_id": challenge.challenge_id, "signature": signature},
        headers=headers,
    )
    assert verified.status_code == 200, verified.get_json()
    approval = verified.get_json()
    assert approval["action_hash"] == challenge.action_hash
    assert approval["single_use"] is True
    assert approval["risk_level"] == "high"

    replay = client.post(
        "/mobile/auth/strong-auth/verify",
        json={"challenge_id": challenge.challenge_id, "signature": signature},
        headers=headers,
    )
    assert replay.status_code == 403
    assert replay.get_json()["error"]["code"] == "STRONG_AUTH_REQUIRED"


def test_challenge_rejects_unknown_fields_and_missing_scope(client):
    create_user("james", "correct-horse", admin=True)
    tokens = paired_login_tokens(client, "james", "correct-horse", scopes=["chat.send"])
    headers = auth_header(tokens["access_token"])
    unknown = client.post(
        "/mobile/auth/strong-auth/challenge",
        json={
            "action_name": "home_assistant_call_service",
            "action": _action(),
            "biometric": True,
        },
        headers=headers,
    )
    assert unknown.status_code == 400
    denied = client.post(
        "/mobile/auth/strong-auth/challenge",
        json={"action_name": "home_assistant_call_service", "action": _action()},
        headers=headers,
    )
    assert denied.status_code == 403
    assert denied.get_json()["error"]["code"] == "PERMISSION_DENIED"


def test_verify_rejects_invalid_signature_without_leaking_crypto_details(client):
    create_user("james", "correct-horse", admin=True)
    tokens = paired_login_tokens(
        client,
        "james",
        "correct-horse",
        scopes=["chat.send", "home.control"],
    )
    headers = auth_header(tokens["access_token"])
    response = client.post(
        "/mobile/auth/strong-auth/challenge",
        json={"action_name": "home_assistant_call_service", "action": _action()},
        headers=headers,
    )
    challenge_id = response.get_json()["challenge_id"]
    invalid = client.post(
        "/mobile/auth/strong-auth/verify",
        json={"challenge_id": challenge_id, "signature": "ZmFrZQ=="},
        headers=headers,
    )
    assert invalid.status_code == 403
    body = invalid.get_json()["error"]
    assert body["code"] == "STRONG_AUTH_INVALID"
    assert "key" not in body["message"].lower()
