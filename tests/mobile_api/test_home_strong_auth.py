from __future__ import annotations

from unittest.mock import patch

from rex.mobile_api.device_proof import sign_transcript
from rex.mobile_api.strong_auth import StrongAuthChallenge, canonical_strong_auth_transcript
from tests.mobile_api.conftest import auth_header, create_user, paired_login_tokens


def _action(*, domain: str = "light", service: str = "turn_off") -> dict:
    return {
        "domain": domain,
        "service": service,
        "entity_id": f"{domain}.downstairs",
        "data": {"transition": 1},
    }


def _challenge(body: dict) -> StrongAuthChallenge:
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


def _approval(client, tokens: dict, action: dict) -> tuple[dict[str, str], str]:
    headers = auth_header(tokens["access_token"])
    created = client.post(
        "/mobile/auth/strong-auth/challenge",
        json={"action_name": "home_assistant_call_service", "action": action},
        headers=headers,
    )
    assert created.status_code == 200, created.get_json()
    challenge = _challenge(created.get_json())
    verified = client.post(
        "/mobile/auth/strong-auth/verify",
        json={
            "challenge_id": challenge.challenge_id,
            "signature": sign_transcript(
                tokens["_private_key"],
                canonical_strong_auth_transcript(challenge),
            ),
        },
        headers=headers,
    )
    assert verified.status_code == 200, verified.get_json()
    return headers, verified.get_json()["approval_id"]


def _command_payload(action: dict, approval_id: str) -> dict:
    return {**action, "strong_auth_approval_id": approval_id}


def test_home_command_consumes_exact_approval_before_verified_dispatch(client):
    create_user("james", "correct-horse", admin=True)
    tokens = paired_login_tokens(
        client,
        "james",
        "correct-horse",
        scopes=["chat.send", "home.control"],
    )
    action = _action()
    headers, approval_id = _approval(client, tokens, action)
    verified = {
        "status": "verified",
        "success": True,
        "entity_id": action["entity_id"],
        "detail": "Verified state.",
    }
    with patch("rex.mobile_api.routes.home.ha_call_service", return_value=verified) as call:
        response = client.post(
            "/mobile/home/command",
            json=_command_payload(action, approval_id),
            headers=headers,
        )
    assert response.status_code == 200, response.get_json()
    body = response.get_json()
    assert body["approval_consumed"] is True
    assert body["result"]["status"] == "verified"
    assert body["result"]["success"] is True
    assert "confirmation_token" not in body["result"]
    call.assert_called_once()

    replay = client.post(
        "/mobile/home/command",
        json=_command_payload(action, approval_id),
        headers=headers,
    )
    assert replay.status_code == 403
    assert replay.get_json()["error"]["code"] == "STRONG_AUTH_REQUIRED"


def test_changed_home_action_does_not_dispatch_or_consume_exact_approval(client):
    create_user("james", "correct-horse", admin=True)
    tokens = paired_login_tokens(
        client,
        "james",
        "correct-horse",
        scopes=["chat.send", "home.control"],
    )
    action = _action()
    headers, approval_id = _approval(client, tokens, action)
    changed = {**action, "service": "turn_on"}
    with patch("rex.mobile_api.routes.home.ha_call_service") as call:
        denied = client.post(
            "/mobile/home/command",
            json=_command_payload(changed, approval_id),
            headers=headers,
        )
    assert denied.status_code == 403
    assert denied.get_json()["error"]["code"] == "STRONG_AUTH_INVALID"
    call.assert_not_called()

    # The exact approval remains usable because the mismatched request did not consume it.
    verified = {
        "status": "verified",
        "success": True,
        "entity_id": action["entity_id"],
    }
    with patch("rex.mobile_api.routes.home.ha_call_service", return_value=verified):
        exact = client.post(
            "/mobile/home/command",
            json=_command_payload(action, approval_id),
            headers=headers,
        )
    assert exact.status_code == 200


def test_sensitive_home_action_uses_internal_confirmation_without_exposing_token(client):
    create_user("james", "correct-horse", admin=True)
    tokens = paired_login_tokens(
        client,
        "james",
        "correct-horse",
        scopes=["chat.send", "home.control"],
    )
    action = _action(domain="lock", service="unlock")
    headers, approval_id = _approval(client, tokens, action)
    first = {
        "status": "confirmation_required",
        "success": False,
        "confirmation_token": "server-only-confirmation",
        "entity_id": action["entity_id"],
    }
    second = {
        "status": "verified",
        "success": True,
        "entity_id": action["entity_id"],
        "confirmation_token": "must-not-leak",
    }
    with patch(
        "rex.mobile_api.routes.home.ha_call_service",
        side_effect=[first, second],
    ) as call:
        response = client.post(
            "/mobile/home/command",
            json=_command_payload(action, approval_id),
            headers=headers,
        )
    assert response.status_code == 200, response.get_json()
    assert call.call_count == 2
    assert call.call_args_list[1].kwargs["context"]["confirmation_token"] == (
        "server-only-confirmation"
    )
    assert "confirmation_token" not in response.get_json()["result"]
    assert response.get_json()["risk_level"] == "critical"


def test_attempted_but_unverified_home_action_returns_202_truthfully(client):
    create_user("james", "correct-horse", admin=True)
    tokens = paired_login_tokens(
        client,
        "james",
        "correct-horse",
        scopes=["chat.send", "home.control"],
    )
    action = _action()
    headers, approval_id = _approval(client, tokens, action)
    result = {
        "status": "attempted_unverified",
        "success": False,
        "entity_id": action["entity_id"],
        "detail": "The requested state was not observed.",
    }
    with patch("rex.mobile_api.routes.home.ha_call_service", return_value=result):
        response = client.post(
            "/mobile/home/command",
            json=_command_payload(action, approval_id),
            headers=headers,
        )
    assert response.status_code == 202
    body = response.get_json()
    assert body["result"]["status"] == "attempted_unverified"
    assert body["result"]["success"] is False


def test_home_command_requires_live_user_permission_even_with_device_scope(client):
    create_user("james", "correct-horse")
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
    assert response.status_code == 403
    assert response.get_json()["error"]["code"] == "PERMISSION_DENIED"
