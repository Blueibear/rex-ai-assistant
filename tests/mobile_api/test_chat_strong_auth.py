from __future__ import annotations

from rex.mobile_api.action_context import authorized_mobile_tool
from rex.mobile_api.chat import MobileChatService
from rex.mobile_api.device_proof import sign_transcript
from rex.mobile_api.strong_auth import (
    StrongAuthChallenge,
    canonical_action,
    canonical_strong_auth_transcript,
)
from tests.mobile_api.conftest import (
    auth_header,
    chat_payload,
    create_user,
    paired_login_tokens,
    parse_sse_events,
)

_ACTION = {
    "domain": "light",
    "service": "turn_off",
    "entity_id": "light.downstairs",
    "data": {"transition": 1},
}


class PrivilegedAssistant:
    async def generate_reply(self, _message, *, voice_mode=False, active_user_id=None):
        del voice_mode, active_user_id
        with authorized_mobile_tool(
            "home_assistant_call_service",
            operation="mutation",
            arguments=_ACTION,
        ):
            return "must not execute before approval"

    async def stream_reply(self, _message, *, active_user_id=None):
        del active_user_id
        with authorized_mobile_tool(
            "home_assistant_call_service",
            operation="mutation",
            arguments=_ACTION,
        ):
            yield "must not stream before approval"


def _challenge_from_wire(body: dict) -> StrongAuthChallenge:
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


def _paired(client, services):
    create_user("james", "correct-horse", admin=True)
    tokens = paired_login_tokens(
        client,
        "james",
        "correct-horse",
        scopes=["chat.send", "home.control"],
    )
    services.chat_service = MobileChatService(assistant_factory=PrivilegedAssistant)
    return tokens, auth_header(tokens["access_token"])


def _assert_details(error: dict) -> StrongAuthChallenge:
    assert error["code"] == "STRONG_AUTH_REQUIRED"
    strong = error["details"]["strong_auth"]
    assert strong["action"] == {
        "action_name": "home_assistant_call_service",
        "payload": _ACTION,
    }
    assert strong["execute"] == {
        "method": "POST",
        "path": "/mobile/home/command",
        "approval_field": "strong_auth_approval_id",
    }
    challenge = _challenge_from_wire(strong["challenge"])
    assert challenge.action_hash == canonical_action("home_assistant_call_service", _ACTION)[2]
    return challenge


def test_http_chat_returns_signable_exact_action_challenge(client, services):
    tokens, headers = _paired(client, services)
    response = client.post("/mobile/chat", json=chat_payload("turn off the light"), headers=headers)
    assert response.status_code == 403
    challenge = _assert_details(response.get_json()["error"])

    signature = sign_transcript(tokens["_private_key"], canonical_strong_auth_transcript(challenge))
    verified = client.post(
        "/mobile/auth/strong-auth/verify",
        json={"challenge_id": challenge.challenge_id, "signature": signature},
        headers=headers,
    )
    assert verified.status_code == 200, verified.get_json()
    assert verified.get_json()["action_hash"] == challenge.action_hash


def test_sse_chat_returns_same_structured_challenge_without_tokens(client, services):
    _tokens, headers = _paired(client, services)
    response = client.post(
        "/mobile/chat/stream",
        json=chat_payload("turn off the light"),
        headers=headers,
    )
    assert response.status_code == 200
    events = parse_sse_events(response.data)
    assert [event["type"] for event in events] == ["error"]
    challenge = _assert_details(events[0])
    assert challenge.risk_level == "high"
