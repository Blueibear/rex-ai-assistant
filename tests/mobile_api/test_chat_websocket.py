"""WebSocket protocol state-machine tests (WebSocket /mobile/chat/stream).

Matrix rows: WS-002..WS-005, WS-007..WS-019, WS-021, IDP-003..IDP-005.

The protocol handler is exercised directly with a fake socket so the tests
are deterministic; one live loopback integration is exercised separately in
the local smoke run.
"""

from __future__ import annotations

import json
import uuid

import pytest

from rex.mobile_api.websocket import (
    CLOSE_AUTH_TIMEOUT,
    CLOSE_RATE_LIMITED,
    CLOSE_UNAUTHENTICATED,
    MAX_FRAME_BYTES,
    MobileWebSocketServer,
    SlidingWindowLimiter,
)
from tests.mobile_api.conftest import chat_payload, create_user, paired_login_tokens

TIMEOUT = object()


class FakeWs:
    """Scripted WebSocket connection for protocol tests."""

    def __init__(self, frames: list) -> None:
        self.incoming = list(frames)
        self.sent: list[dict] = []
        self.closed: tuple[int, str] | None = None

    def receive(self, timeout: float | None = None):
        if self.closed is not None or not self.incoming:
            raise RuntimeError("connection closed")
        item = self.incoming.pop(0)
        if item is TIMEOUT:
            return None
        return item

    def send(self, data: str) -> None:
        if self.closed is not None:
            raise RuntimeError("connection closed")
        self.sent.append(json.loads(data))

    def close(self, code: int = 1000, reason: str = "") -> None:
        self.closed = (code, reason)


def _auth_frame(token: str) -> str:
    return json.dumps(
        {
            "type": "auth",
            "access_token": token,
            "client": {"platform": "ios", "app_version": "0.1.0", "device_id": "dev-1"},
        }
    )


def _chat_frame(message: str = "Hello Rex", **overrides) -> str:
    return json.dumps({"type": "chat", **chat_payload(message, **overrides)})


def _login(client, username: str = "james", password: str = "pw-123456") -> tuple[str, str]:
    user_id = create_user(username, password)
    tokens = paired_login_tokens(client, username, password)
    return user_id, tokens["access_token"]


def _run(services, frames: list, remote: str = "10.0.0.1") -> FakeWs:
    ws = FakeWs(frames)
    MobileWebSocketServer(services).handle(ws, remote)
    return ws


class TestAuthentication:
    def test_valid_auth_gets_auth_ok(self, client, services) -> None:
        """WS-002: snake_case auth_ok with session and user projection."""
        user_id, token = _login(client)
        ws = _run(services, [_auth_frame(token)])
        assert ws.sent[0]["type"] == "auth_ok"
        assert ws.sent[0]["session_id"]
        assert ws.sent[0]["user"]["id"] == user_id
        assert {"id", "name", "role", "permissions"} <= set(ws.sent[0]["user"].keys())

    def test_first_frame_chat_is_rejected(self, client, services, fake_chat_service) -> None:
        """WS-003: nothing is processed before authentication."""
        _login(client)
        ws = _run(services, [_chat_frame()])
        assert ws.sent[0]["type"] == "auth_error"
        assert ws.closed[0] == CLOSE_UNAUTHENTICATED
        assert fake_chat_service.calls == []

    def test_first_frame_ping_is_rejected(self, client, services) -> None:
        _login(client)
        ws = _run(services, [json.dumps({"type": "ping"})])
        assert ws.closed[0] == CLOSE_UNAUTHENTICATED

    def test_auth_timeout_closes_4408(self, services) -> None:
        """WS-004."""
        ws = _run(services, [TIMEOUT])
        assert ws.closed[0] == CLOSE_AUTH_TIMEOUT

    def test_invalid_token_closes_4401(self, services) -> None:
        """WS-005."""
        ws = _run(services, [_auth_frame("not-a-real-token")])
        assert ws.sent[0]["type"] == "auth_error"
        assert ws.sent[0]["code"] in ("AUTH_TOKEN_INVALID", "AUTH_TOKEN_EXPIRED")
        assert ws.closed[0] == CLOSE_UNAUTHENTICATED

    @pytest.mark.parametrize(
        "client_metadata",
        [
            None,
            {},
            {"platform": "ios", "app_version": "0.1.0"},
            {"platform": "desktop", "app_version": "0.1.0", "device_id": "dev-1"},
            {"platform": "ios", "app_version": 1, "device_id": "dev-1"},
            {"platform": "ios", "app_version": "0.1.0", "device_id": "bad id"},
            {
                "platform": "ios",
                "app_version": "0.1.0",
                "device_id": "dev-1",
                "unknown": "field",
            },
        ],
    )
    def test_client_metadata_is_required_and_strict(
        self, client, services, client_metadata
    ) -> None:
        _, token = _login(client)
        frame = {"type": "auth", "access_token": token}
        if client_metadata is not None:
            frame["client"] = client_metadata
        ws = _run(services, [json.dumps(frame)])
        assert ws.sent[0]["type"] == "auth_error"
        assert ws.closed[0] == CLOSE_UNAUTHENTICATED

    def test_unknown_auth_frame_field_is_rejected(self, client, services) -> None:
        _, token = _login(client)
        frame = json.loads(_auth_frame(token))
        frame["user_id"] = "attacker-controlled"
        ws = _run(services, [json.dumps(frame)])
        assert ws.sent[0]["type"] == "auth_error"
        assert ws.closed[0] == CLOSE_UNAUTHENTICATED

    def test_4403_is_reserved_chat_is_not_permission_gated(self, client, services) -> None:
        """4403 contract truthfulness: no Session 2 path emits it.

        The canonical permission model (rex.permissions) authorizes tools at
        dispatch time, not chat access — so an authenticated user with ZERO
        granted permissions still chats normally and is never closed with
        4403.  The code remains reserved in the wire contract for a future
        server-side authorization denial.
        """
        from rex.mobile_api.websocket import CLOSE_FORBIDDEN
        from rex.permissions import get_permissions

        user_id, token = _login(client)  # created without any permissions
        assert get_permissions(user_id) == []
        ws = _run(services, [_auth_frame(token), _chat_frame("hello with no permissions")])
        assert ws.sent[0]["type"] == "auth_ok"
        assert [e["type"] for e in ws.sent][-1] == "message_done"
        assert ws.closed is None or ws.closed[0] != CLOSE_FORBIDDEN

    def test_revoked_session_token_closes_4401(self, client, services) -> None:
        _, token = _login(client)
        from tests.mobile_api.conftest import auth_header

        client.post("/mobile/auth/logout", headers=auth_header(token))
        ws = _run(services, [_auth_frame(token)])
        assert ws.sent[0]["type"] == "auth_error"
        assert ws.closed[0] == CLOSE_UNAUTHENTICATED

    def test_second_auth_frame_cannot_replace_identity(self, client, services) -> None:
        """WS-008."""
        _, token_a = _login(client, "james", "pw-123456")
        _, token_b = _login(client, "cole", "pw-abcdef")
        ws = _run(services, [_auth_frame(token_a), _auth_frame(token_b)])
        errors = [e for e in ws.sent if e["type"] == "error"]
        assert errors and "authenticated" in errors[0]["message"].lower()

    def test_connection_rate_limit_closes_4429(self, client, services) -> None:
        """WS-007."""
        _, token = _login(client)
        server = MobileWebSocketServer(services)
        server._connection_limiter = SlidingWindowLimiter(1)
        first = FakeWs([_auth_frame(token)])
        server.handle(first, "9.9.9.9")
        assert first.sent[0]["type"] == "auth_ok"
        second = FakeWs([_auth_frame(token)])
        server.handle(second, "9.9.9.9")
        assert second.closed[0] == CLOSE_RATE_LIMITED
        assert second.sent == []


class TestFrames:
    def test_valid_chat_reserves_acks_streams_and_completes(
        self, client, services, fake_chat_service
    ) -> None:
        """WS-010/WS-011/WS-012/WS-013: reservation → ack → events → done."""
        user_id, token = _login(client)
        frame = _chat_frame("Hello Rex")
        message_id = json.loads(frame)["message_id"]
        ws = _run(services, [_auth_frame(token), frame])

        types = [e["type"] for e in ws.sent]
        assert types[0] == "auth_ok"
        assert types[1] == "ack"
        assert types[-1] == "message_done"
        ack = ws.sent[1]
        assert set(ack.keys()) == {"type", "message_id", "accepted_at"}
        assert ack["message_id"] == message_id
        done = ws.sent[-1]
        assert done["status"] == "completed"
        assert done["full_content"] == f"echo[{user_id}]: Hello Rex"
        for event in ws.sent:
            for key in event:
                assert key == key.lower() and " " not in key
        assert fake_chat_service.calls == [("Hello Rex", user_id)]

    def test_websocket_uses_same_privacy_safe_progressive_status_grammar(
        self, client, services, fake_chat_service
    ) -> None:
        from rex.runtime.status import TurnStatus, TurnStatusUpdate

        _, token = _login(client)
        fake_chat_service.status_updates = [
            TurnStatusUpdate("turn-ws", 0, TurnStatus.THINKING, False),
            TurnStatusUpdate("turn-ws", 1, TurnStatus.VERIFYING, False),
            TurnStatusUpdate("turn-ws", 2, TurnStatus.DONE, True),
        ]
        frame = _chat_frame("private websocket prompt")
        message_id = json.loads(frame)["message_id"]

        ws = _run(services, [_auth_frame(token), frame])
        status_events = [event for event in ws.sent if event["type"] == "status"]

        assert [event["status"] for event in status_events] == ["thinking", "verifying", "done"]
        assert all(
            set(event) == {"type", "message_id", "turn_id", "sequence", "status", "terminal"}
            for event in status_events
        )
        assert all(event["message_id"] == message_id for event in status_events)
        assert "private websocket prompt" not in repr(status_events)
        assert ws.sent[-1]["type"] == "message_done"

    def test_client_identity_fields_in_chat_rejected(
        self, client, services, fake_chat_service
    ) -> None:
        """WS-009: bound principal cannot be altered by frame fields."""
        _, token = _login(client)
        bad = json.dumps({"type": "chat", **chat_payload(), "user_id": "someone-else"})
        ws = _run(services, [_auth_frame(token), bad])
        errors = [e for e in ws.sent if e["type"] == "error"]
        assert errors and errors[0]["code"] == "BAD_REQUEST"
        assert fake_chat_service.calls == []

    def test_malformed_json_is_structured_protocol_error(self, client, services) -> None:
        """WS-014: never rendered as assistant text."""
        _, token = _login(client)
        ws = _run(services, [_auth_frame(token), "this is {not json"])
        errors = [e for e in ws.sent if e["type"] == "error"]
        assert errors and errors[0]["code"] == "BAD_REQUEST"
        assert all(e["type"] != "token" for e in ws.sent)

    def test_oversized_frame_rejected_without_processing(
        self, client, services, fake_chat_service
    ) -> None:
        """WS-015."""
        _, token = _login(client)
        huge = json.dumps({"type": "chat", **chat_payload("x" * (MAX_FRAME_BYTES + 10))})
        ws = _run(services, [_auth_frame(token), huge])
        errors = [e for e in ws.sent if e["type"] == "error"]
        assert errors
        assert fake_chat_service.calls == []

    def test_message_flood_closes_4429(self, client, services, fake_chat_service) -> None:
        """WS-016."""
        _, token = _login(client)
        server = MobileWebSocketServer(services)
        server._message_limit = 2
        frames = [_auth_frame(token)] + [_chat_frame(f"m{i}") for i in range(4)]
        ws = FakeWs(frames)
        server.handle(ws, "10.0.0.2")
        assert ws.closed[0] == CLOSE_RATE_LIMITED
        assert len(fake_chat_service.calls) == 2

    def test_ping_gets_pong_without_private_content(self, client, services) -> None:
        """WS-017."""
        _, token = _login(client)
        ws = _run(
            services,
            [_auth_frame(token), json.dumps({"type": "ping", "sent_at": "2026-07-15T00:00:00Z"})],
        )
        pongs = [e for e in ws.sent if e["type"] == "pong"]
        assert len(pongs) == 1
        assert set(pongs[0].keys()) == {"type", "sent_at"}

    def test_session_revoked_mid_connection_closes_4401(self, client, services) -> None:
        """WS-018: the next privileged message is rejected."""
        _, token = _login(client)
        from tests.mobile_api.conftest import auth_header

        ws = FakeWs([_auth_frame(token)])
        server = MobileWebSocketServer(services)
        server.handle(ws, "10.0.0.3")
        assert ws.sent[0]["type"] == "auth_ok"

        # Revoke, then deliver a chat frame on a still-open connection.
        client.post("/mobile/auth/logout", headers=auth_header(token))
        ws2 = FakeWs([_auth_frame(token)])
        server.handle(ws2, "10.0.0.4")
        assert ws2.sent[0]["type"] == "auth_error"


class TestWsIdempotency:
    def test_ws_duplicate_executes_once_and_replays(
        self, client, services, fake_chat_service
    ) -> None:
        """IDP-003/WS-019: reconnect replay never re-executes."""
        _, token = _login(client)
        frame = _chat_frame("dedupe me")
        ws = _run(services, [_auth_frame(token), frame, frame])
        done_events = [e for e in ws.sent if e["type"] == "message_done"]
        ack_events = [e for e in ws.sent if e["type"] == "ack"]
        assert len(done_events) == 2  # replayed terminal result
        assert done_events[0]["full_content"] == done_events[1]["full_content"]
        assert len(ack_events) == 2  # duplicate receives the original ack
        assert len(fake_chat_service.calls) == 1

    def test_ws_then_http_fallback_executes_once(self, client, services, fake_chat_service) -> None:
        """IDP-004: same ID over WS then HTTP → one execution, same result."""
        _, token = _login(client)
        payload = chat_payload("ws first")
        ws = _run(services, [_auth_frame(token), json.dumps({"type": "chat", **payload})])
        done = [e for e in ws.sent if e["type"] == "message_done"][0]

        tokens = paired_login_tokens(client, "james", "pw-123456")
        from tests.mobile_api.conftest import auth_header

        response = client.post(
            "/mobile/chat", json=payload, headers=auth_header(tokens["access_token"])
        )
        assert response.status_code == 200
        assert response.get_json()["response"] == done["full_content"]
        assert len(fake_chat_service.calls) == 1

    def test_http_then_ws_reconnect_executes_once(
        self, client, services, fake_chat_service
    ) -> None:
        """IDP-005."""
        _, token = _login(client)
        payload = chat_payload("http first")
        from tests.mobile_api.conftest import auth_header

        response = client.post("/mobile/chat", json=payload, headers=auth_header(token))
        assert response.status_code == 200
        ws = _run(services, [_auth_frame(token), json.dumps({"type": "chat", **payload})])
        done = [e for e in ws.sent if e["type"] == "message_done"]
        assert len(done) == 1
        assert done[0]["full_content"] == response.get_json()["response"]
        assert len(fake_chat_service.calls) == 1

    def test_same_id_different_payload_conflicts(self, client, services, fake_chat_service) -> None:
        _, token = _login(client)
        payload = chat_payload("original")
        tampered = dict(payload, message="tampered")
        ws = _run(
            services,
            [
                _auth_frame(token),
                json.dumps({"type": "chat", **payload}),
                json.dumps({"type": "chat", **tampered}),
            ],
        )
        errors = [e for e in ws.sent if e["type"] == "error"]
        assert errors and errors[0]["code"] == "IDEMPOTENCY_CONFLICT"
        assert len(fake_chat_service.calls) == 1

    def test_two_users_same_message_id_are_isolated(
        self, client, services, fake_chat_service
    ) -> None:
        """WS-021: connection principals and results stay isolated."""
        user_a, token_a = _login(client, "james", "pw-123456")
        user_b, token_b = _login(client, "cole", "pw-abcdef")
        shared = chat_payload("who am I")
        shared["message_id"] = str(uuid.uuid4())
        frame = json.dumps({"type": "chat", **shared})
        ws_a = _run(services, [_auth_frame(token_a), frame])
        ws_b = _run(services, [_auth_frame(token_b), frame])
        done_a = [e for e in ws_a.sent if e["type"] == "message_done"][0]
        done_b = [e for e in ws_b.sent if e["type"] == "message_done"][0]
        assert done_a["full_content"] == f"echo[{user_a}]: who am I"
        assert done_b["full_content"] == f"echo[{user_b}]: who am I"
        assert len(fake_chat_service.calls) == 2

    def test_no_tokens_or_bodies_in_logs(self, client, services, caplog) -> None:
        """WS-022: only safe correlation data is logged."""
        _, token = _login(client)
        secret_message = "my-secret-garage-code-4242"  # pragma: allowlist secret
        with caplog.at_level("DEBUG"):
            _run(
                services,
                [_auth_frame(token), _chat_frame(secret_message)],
            )
        assert token not in caplog.text
        assert secret_message not in caplog.text
