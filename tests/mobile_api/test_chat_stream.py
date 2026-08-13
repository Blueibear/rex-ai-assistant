"""SSE streaming chat tests (POST /mobile/chat/stream).

Matrix rows: SSE-001..SSE-004, SSE-006, SSE-008, SSE-009.
"""

from __future__ import annotations

from tests.mobile_api.conftest import (
    auth_header,
    chat_payload,
    create_user,
    paired_login_tokens,
    parse_sse_events,
)


def _authed(client, username: str = "james", password: str = "pw-123456") -> tuple[str, dict]:
    user_id = create_user(username, password)
    tokens = paired_login_tokens(client, username, password)
    return user_id, auth_header(tokens["access_token"])


class TestSseGrammar:
    def test_content_type_and_no_cache(self, client) -> None:
        """SSE-001."""
        _, headers = _authed(client)
        response = client.post("/mobile/chat/stream", json=chat_payload(), headers=headers)
        assert response.status_code == 200
        assert response.mimetype == "text/event-stream"
        assert response.headers["Cache-Control"] == "no-cache"

    def test_token_events_are_canonical_json(self, client) -> None:
        """SSE-002: every data payload is valid canonical snake_case JSON."""
        user_id, headers = _authed(client)
        payload = chat_payload("stream me")
        response = client.post("/mobile/chat/stream", json=payload, headers=headers)
        events = parse_sse_events(response.data)
        assert events, "stream produced no events"
        token_events = [e for e in events if e["type"] == "token"]
        assert token_events, "no token events"
        for event in token_events:
            assert set(event.keys()) == {"type", "message_id", "content"}
            assert event["message_id"] == payload["message_id"]
        reconstructed = "".join(e["content"] for e in token_events)
        assert reconstructed == f"echo[{user_id}]: stream me"

    def test_progressive_status_frames_are_privacy_safe_and_precede_terminal_done(
        self, client, fake_chat_service
    ) -> None:
        from rex.runtime.status import TurnStatus, TurnStatusUpdate

        _, headers = _authed(client)
        fake_chat_service.status_updates = [
            TurnStatusUpdate("turn-mobile", 0, TurnStatus.THINKING, False),
            TurnStatusUpdate("turn-mobile", 1, TurnStatus.ACTING, False),
            TurnStatusUpdate("turn-mobile", 2, TurnStatus.DONE, True),
        ]
        payload = chat_payload("status please")

        response = client.post("/mobile/chat/stream", json=payload, headers=headers)
        events = parse_sse_events(response.data)
        status_events = [event for event in events if event["type"] == "status"]

        assert [event["status"] for event in status_events] == ["thinking", "acting", "done"]
        assert all(
            set(event) == {"type", "message_id", "turn_id", "sequence", "status", "terminal"}
            for event in status_events
        )
        assert all(event["message_id"] == payload["message_id"] for event in status_events)
        assert events.index(status_events[-1]) < len(events) - 1
        assert events[-1]["type"] == "message_done"
        assert "status please" not in repr(status_events)

    def test_exactly_one_terminal_message_done(self, client) -> None:
        """SSE-003."""
        _, headers = _authed(client)
        payload = chat_payload()
        response = client.post("/mobile/chat/stream", json=payload, headers=headers)
        events = parse_sse_events(response.data)
        done_events = [e for e in events if e["type"] == "message_done"]
        assert len(done_events) == 1
        done = done_events[0]
        assert events[-1] == done
        assert done["message_id"] == payload["message_id"]
        assert done["conversation_id"] == payload["conversation_id"]
        assert done["status"] == "completed"
        assert done["full_content"]

    def test_midstream_failure_emits_structured_terminal_error(
        self, client, fake_chat_service
    ) -> None:
        """SSE-004/SSE-006: failures are structured errors, never prose."""
        _, headers = _authed(client)
        fake_chat_service.stream_fail_after_chunks = 1
        payload = chat_payload("this will break")
        response = client.post("/mobile/chat/stream", json=payload, headers=headers)
        events = parse_sse_events(response.data)
        assert events[-1]["type"] == "error"
        assert events[-1]["code"] == "BACKEND_UNAVAILABLE"
        assert events[-1]["retryable"] is True
        assert all(e["type"] != "message_done" for e in events)

    def test_terminal_status_is_flushed_before_stream_error(self, client, services) -> None:
        """A canonical terminal status queued by Assistant is not lost when iteration raises."""
        from rex.mobile_api import errors as merr
        from rex.mobile_api.errors import MobileApiError
        from rex.runtime.status import TurnStatus, TurnStatusUpdate

        _, headers = _authed(client)

        def failing_stream(message, *, status_observer=None, **kwargs):  # noqa: ANN001,ARG001
            if status_observer is not None:
                status_observer(TurnStatusUpdate("turn-failed", 1, TurnStatus.ERROR, True))
            raise MobileApiError(
                merr.BACKEND_UNAVAILABLE,
                "Rex is temporarily unavailable.",
                503,
                retryable=True,
            )
            yield "unreachable"

        services.chat_service.stream = failing_stream
        response = client.post(
            "/mobile/chat/stream",
            json=chat_payload("fail after terminal status"),
            headers=headers,
        )
        events = parse_sse_events(response.data)

        assert [event["type"] for event in events[-2:]] == ["status", "error"]
        assert events[-2]["status"] == "error"
        assert events[-2]["terminal"] is True
        assert events[-1]["code"] == "BACKEND_UNAVAILABLE"


class TestSseIdempotency:
    def test_completed_duplicate_replays_without_execution(self, client, fake_chat_service) -> None:
        """SSE-008: stored terminal result is replayed; one execution total."""
        _, headers = _authed(client)
        payload = chat_payload("stream once")
        first = client.post("/mobile/chat/stream", json=payload, headers=headers)
        first_events = parse_sse_events(first.data)
        second = client.post("/mobile/chat/stream", json=payload, headers=headers)
        second_events = parse_sse_events(second.data)
        assert len(fake_chat_service.calls) == 1
        assert [e["type"] for e in second_events] == ["message_done"]
        assert (
            second_events[0]["full_content"]
            == [e for e in first_events if e["type"] == "message_done"][0]["full_content"]
        )

    def test_cross_transport_duplicate_with_http(self, client, fake_chat_service) -> None:
        """IDP-004 (HTTP↔SSE half): same ID over both HTTP routes runs once."""
        _, headers = _authed(client)
        payload = chat_payload("cross transport")
        http_response = client.post("/mobile/chat", json=payload, headers=headers)
        assert http_response.status_code == 200
        sse_response = client.post("/mobile/chat/stream", json=payload, headers=headers)
        events = parse_sse_events(sse_response.data)
        assert [e["type"] for e in events] == ["message_done"]
        assert events[0]["full_content"] == http_response.get_json()["response"]
        assert len(fake_chat_service.calls) == 1

    def test_processing_duplicate_reports_in_progress(self, client, services) -> None:
        """SSE-009: an in-flight duplicate never executes a second time."""
        from rex.mobile_api import idempotency as idem

        user_id, headers = _authed(client)
        payload = chat_payload("in flight")
        # Simulate an in-flight reservation from another transport.
        request_hash = idem.compute_request_hash(
            {
                "message_id": payload["message_id"],
                "conversation_id": payload["conversation_id"],
                "sent_at": payload["sent_at"],
                "message": payload["message"],
                "mode": payload["mode"],
                "client_context": payload["client_context"],
            }
        )
        services.message_store.reserve(
            user_id, payload["message_id"], payload["conversation_id"], request_hash
        )
        response = client.post("/mobile/chat/stream", json=payload, headers=headers)
        assert response.status_code == 409
        assert response.get_json()["error"]["code"] == "REQUEST_IN_PROGRESS"
        assert response.get_json()["error"]["retryable"] is True
