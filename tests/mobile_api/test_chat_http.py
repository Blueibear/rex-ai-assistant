"""HTTP chat route tests (POST /mobile/chat).

Matrix rows: CHAT-001..CHAT-009, CHAT-013..CHAT-017, IDP-002, IDP-007.
"""

from __future__ import annotations

import uuid

from rex.mobile_api import errors as merr
from rex.mobile_api.errors import MobileApiError
from tests.mobile_api.conftest import (
    auth_header,
    chat_payload,
    create_user,
    login_tokens,
)


def _authed(client, username: str = "james", password: str = "pw-123456") -> tuple[str, dict]:
    user_id = create_user(username, password)
    tokens = login_tokens(client, username, password)
    return user_id, auth_header(tokens["access_token"])


class TestChatAuthentication:
    def test_missing_auth_rejected_before_execution(self, client, fake_chat_service) -> None:
        """CHAT-002: no principal → no idempotency, no Assistant call."""
        response = client.post("/mobile/chat", json=chat_payload())
        assert response.status_code == 401
        assert fake_chat_service.calls == []

    def test_client_user_id_rejected(self, client, fake_chat_service) -> None:
        """CHAT-003: client identity fields are rejected outright."""
        _, headers = _authed(client)
        response = client.post(
            "/mobile/chat", json=chat_payload(user_id="someone-else"), headers=headers
        )
        assert response.status_code == 400
        assert fake_chat_service.calls == []

    def test_client_authorization_fields_rejected(self, client, fake_chat_service) -> None:
        """CHAT-004: role/permissions/risk/approval claims never execute."""
        _, headers = _authed(client)
        for field in ("role", "permissions", "risk", "approval", "biometric_verified"):
            response = client.post(
                "/mobile/chat", json=chat_payload(**{field: "admin"}), headers=headers
            )
            assert response.status_code == 400, field
        assert fake_chat_service.calls == []


class TestChatValidation:
    def test_empty_message_rejected(self, client) -> None:
        _, headers = _authed(client)
        response = client.post("/mobile/chat", json=chat_payload(message="  "), headers=headers)
        assert response.status_code == 400

    def test_oversized_message_rejected(self, client, fake_chat_service) -> None:
        """CHAT-005: size validation happens before the Assistant."""
        _, headers = _authed(client)
        response = client.post(
            "/mobile/chat", json=chat_payload(message="x" * 8_001), headers=headers
        )
        assert response.status_code == 400
        assert fake_chat_service.calls == []

    def test_invalid_message_id_rejected(self, client, services) -> None:
        """CHAT-006: invalid IDs fail before reservation."""
        _, headers = _authed(client)
        payload = chat_payload(message_id="not-a-uuid")
        response = client.post("/mobile/chat", json=payload, headers=headers)
        assert response.status_code == 400

    def test_invalid_conversation_id_rejected(self, client) -> None:
        """CHAT-007."""
        _, headers = _authed(client)
        response = client.post(
            "/mobile/chat", json=chat_payload(conversation_id="nope"), headers=headers
        )
        assert response.status_code == 400

    def test_invalid_sent_at_rejected(self, client) -> None:
        _, headers = _authed(client)
        response = client.post(
            "/mobile/chat", json=chat_payload(sent_at="yesterday"), headers=headers
        )
        assert response.status_code == 400

    def test_wrong_mode_rejected(self, client) -> None:
        _, headers = _authed(client)
        response = client.post("/mobile/chat", json=chat_payload(mode="desktop"), headers=headers)
        assert response.status_code == 400

    def test_idempotency_header_mismatch_conflicts(self, client, fake_chat_service) -> None:
        """CHAT-008: header/body message ID mismatch → conflict, no execution."""
        _, headers = _authed(client)
        headers["Idempotency-Key"] = str(uuid.uuid4())
        response = client.post("/mobile/chat", json=chat_payload(), headers=headers)
        assert response.status_code == 409
        assert response.get_json()["error"]["code"] == "IDEMPOTENCY_CONFLICT"
        assert fake_chat_service.calls == []


class TestChatExecution:
    def test_valid_chat_calls_assistant_once_with_principal(
        self, client, fake_chat_service
    ) -> None:
        """CHAT-001/CHAT-016: canonical Assistant, explicit principal identity."""
        user_id, headers = _authed(client)
        payload = chat_payload("What time is it?")
        headers["Idempotency-Key"] = payload["message_id"]
        response = client.post("/mobile/chat", json=payload, headers=headers)
        assert response.status_code == 200
        body = response.get_json()
        assert body["message_id"] == payload["message_id"]
        assert body["conversation_id"] == payload["conversation_id"]
        assert body["status"] == "completed"
        assert body["events"] == []
        assert body["request_id"]
        assert body["response"] == f"echo[{user_id}]: What time is it?"
        assert fake_chat_service.calls == [("What time is it?", user_id)]

    def test_normal_response_status_is_completed_not_verified(self, client) -> None:
        """CHAT-009."""
        _, headers = _authed(client)
        response = client.post("/mobile/chat", json=chat_payload(), headers=headers)
        assert response.get_json()["status"] == "completed"

    def test_exact_duplicate_replays_without_second_execution(
        self, client, fake_chat_service
    ) -> None:
        """IDP-002: one Assistant execution for repeated delivery."""
        _, headers = _authed(client)
        payload = chat_payload("only once")
        first = client.post("/mobile/chat", json=payload, headers=headers)
        second = client.post("/mobile/chat", json=payload, headers=headers)
        assert first.status_code == 200 and second.status_code == 200
        assert first.get_json()["response"] == second.get_json()["response"]
        assert len(fake_chat_service.calls) == 1

    def test_same_id_different_payload_conflicts(self, client, fake_chat_service) -> None:
        """IDP-007."""
        _, headers = _authed(client)
        payload = chat_payload("original")
        client.post("/mobile/chat", json=payload, headers=headers)
        changed = dict(payload, message="tampered")
        response = client.post("/mobile/chat", json=changed, headers=headers)
        assert response.status_code == 409
        assert response.get_json()["error"]["code"] == "IDEMPOTENCY_CONFLICT"
        assert len(fake_chat_service.calls) == 1

    def test_backend_failure_is_structured_never_mock(self, client, fake_chat_service) -> None:
        """CHAT-013: BACKEND_UNAVAILABLE, and a retry replays the failure."""
        _, headers = _authed(client)
        fake_chat_service.fail_with = MobileApiError(
            merr.BACKEND_UNAVAILABLE, "Rex is temporarily unavailable.", 503, retryable=True
        )
        payload = chat_payload()
        response = client.post("/mobile/chat", json=payload, headers=headers)
        assert response.status_code == 503
        error = response.get_json()["error"]
        assert error["code"] == "BACKEND_UNAVAILABLE"
        assert error["retryable"] is True

        # The failure is terminal for this message ID; no re-execution.
        fake_chat_service.fail_with = None
        retry = client.post("/mobile/chat", json=payload, headers=headers)
        assert retry.status_code == 503
        assert retry.get_json()["error"]["code"] == "BACKEND_UNAVAILABLE"
        assert fake_chat_service.calls == []

    def test_two_users_are_isolated(self, client, fake_chat_service) -> None:
        """CHAT-014/IDP-008: same message ID, both execute, separate identity."""
        user_a, headers_a = _authed(client, "james", "pw-123456")
        user_b, headers_b = _authed(client, "cole", "pw-abcdef")
        payload = chat_payload("shared prompt")
        response_a = client.post("/mobile/chat", json=payload, headers=headers_a)
        response_b = client.post("/mobile/chat", json=payload, headers=headers_b)
        assert response_a.status_code == 200 and response_b.status_code == 200
        assert response_a.get_json()["response"] == f"echo[{user_a}]: shared prompt"
        assert response_b.get_json()["response"] == f"echo[{user_b}]: shared prompt"
        assert {call[1] for call in fake_chat_service.calls} == {user_a, user_b}

    def test_message_body_not_logged(self, client, caplog) -> None:
        """CHAT-017: chat text never appears in server logs."""
        _, headers = _authed(client)
        secret_text = "the-vault-code-is-9137"  # pragma: allowlist secret
        with caplog.at_level("DEBUG"):
            client.post("/mobile/chat", json=chat_payload(secret_text), headers=headers)
        assert secret_text not in caplog.text
