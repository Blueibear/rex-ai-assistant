"""Tests for US-085: SMS receive stub.

Acceptance criteria:
- SmsReceiverStub exposes a method to inject a test inbound message.
- Injected messages are routed through the same handler as real inbound SMS.
- Handler produces a response or triggers the expected downstream action.
- No network calls are made.
"""

from __future__ import annotations

import socket
from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from rex.messaging_backends.sms_receiver_stub import (
    InboundSmsHandlerResult,
    ReceivedSmsRecord,
    SmsReceiverStub,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _no_network(*args, **kwargs):  # type: ignore[no-untyped-def]
    raise AssertionError("SmsReceiverStub must not make network calls")


# ---------------------------------------------------------------------------
# AC1: inject() method exists and accepts a test inbound message
# ---------------------------------------------------------------------------


class TestInjectInterface:
    def test_inject_returns_handler_result(self) -> None:
        stub = SmsReceiverStub()
        result = stub.inject(from_number="+15555551111", body="Hello Rex")
        assert isinstance(result, InboundSmsHandlerResult)

    def test_inject_with_all_fields(self) -> None:
        stub = SmsReceiverStub()
        ts = datetime(2026, 3, 11, 10, 0, 0, tzinfo=timezone.utc)
        result = stub.inject(
            from_number="+15555551111",
            body="Test message",
            to_number="+15555559999",
            sid="stub_rx_abc123",
            received_at=ts,
        )
        assert result.status == "received"
        assert result.sid == "stub_rx_abc123"

    def test_inject_auto_generates_sid(self) -> None:
        stub = SmsReceiverStub()
        result = stub.inject(from_number="+15555551111", body="Auto sid")
        assert result.sid.startswith("stub_rx_")

    def test_inject_uses_default_to_number_when_omitted(self) -> None:
        stub = SmsReceiverStub(default_to_number="+19995550001")
        stub.inject(from_number="+15555551111", body="Hi")
        record = stub.received_messages[0]
        assert record.to_number == "+19995550001"

    def test_inject_uses_provided_to_number(self) -> None:
        stub = SmsReceiverStub()
        stub.inject(from_number="+15555551111", body="Hi", to_number="+12223334444")
        record = stub.received_messages[0]
        assert record.to_number == "+12223334444"

    def test_inject_stores_body_verbatim(self) -> None:
        stub = SmsReceiverStub()
        stub.inject(from_number="+15555551111", body="Special chars: áéí")
        record = stub.received_messages[0]
        assert record.body == "Special chars: áéí"

    def test_inject_preserves_received_at_timestamp(self) -> None:
        ts = datetime(2026, 1, 15, 8, 30, 0, tzinfo=timezone.utc)
        stub = SmsReceiverStub()
        stub.inject(from_number="+15555551111", body="Timed", received_at=ts)
        record = stub.received_messages[0]
        assert record.received_at == ts

    def test_inject_defaults_received_at_to_now(self) -> None:
        stub = SmsReceiverStub()
        before = datetime.now(timezone.utc)
        stub.inject(from_number="+15555551111", body="Now")
        after = datetime.now(timezone.utc)
        record = stub.received_messages[0]
        assert before <= record.received_at <= after


# ---------------------------------------------------------------------------
# AC2: Injected messages routed through the same handler
# ---------------------------------------------------------------------------


class TestHandlerRouting:
    def test_default_handler_sets_status_received(self) -> None:
        stub = SmsReceiverStub()
        result = stub.inject(from_number="+15555551111", body="Hi")
        assert result.status == "received"

    def test_default_handler_echoes_from_number(self) -> None:
        stub = SmsReceiverStub()
        result = stub.inject(from_number="+15555551111", body="Hi")
        assert result.from_number == "+15555551111"

    def test_default_handler_echoes_body(self) -> None:
        stub = SmsReceiverStub()
        result = stub.inject(from_number="+15555551111", body="Echo test")
        assert result.body == "Echo test"

    def test_custom_handler_receives_record(self) -> None:
        received: list[ReceivedSmsRecord] = []

        def custom_handler(record: ReceivedSmsRecord) -> InboundSmsHandlerResult:
            received.append(record)
            return InboundSmsHandlerResult(
                status="received",
                sid=record.sid,
                from_number=record.from_number,
                to_number=record.to_number,
                body=record.body,
            )

        stub = SmsReceiverStub(handler=custom_handler)
        stub.inject(from_number="+15555551111", body="Custom handler test")
        assert len(received) == 1
        assert received[0].body == "Custom handler test"

    def test_custom_handler_return_value_propagated(self) -> None:
        def error_handler(record: ReceivedSmsRecord) -> InboundSmsHandlerResult:
            return InboundSmsHandlerResult(
                status="error",
                sid=record.sid,
                from_number=record.from_number,
                to_number=record.to_number,
                body=record.body,
                error="Simulated handler failure",
            )

        stub = SmsReceiverStub(handler=error_handler)
        result = stub.inject(from_number="+15555551111", body="Will fail")
        assert result.status == "error"
        assert result.error == "Simulated handler failure"

    def test_handler_called_once_per_inject(self) -> None:
        call_count = [0]

        def counting_handler(record: ReceivedSmsRecord) -> InboundSmsHandlerResult:
            call_count[0] += 1
            return InboundSmsHandlerResult(
                status="received",
                sid=record.sid,
                from_number=record.from_number,
                to_number=record.to_number,
                body=record.body,
            )

        stub = SmsReceiverStub(handler=counting_handler)
        stub.inject(from_number="+15555551111", body="Msg 1")
        stub.inject(from_number="+15555552222", body="Msg 2")
        stub.inject(from_number="+15555553333", body="Msg 3")
        assert call_count[0] == 3

    def test_handler_receives_correct_record_fields(self) -> None:
        captured: list[ReceivedSmsRecord] = []

        def capture_handler(record: ReceivedSmsRecord) -> InboundSmsHandlerResult:
            captured.append(record)
            return InboundSmsHandlerResult(
                status="received",
                sid=record.sid,
                from_number=record.from_number,
                to_number=record.to_number,
                body=record.body,
            )

        stub = SmsReceiverStub(handler=capture_handler)
        stub.inject(
            from_number="+15555551111",
            body="Field check",
            to_number="+15555559999",
            sid="sid_xyz",
        )
        r = captured[0]
        assert r.from_number == "+15555551111"
        assert r.to_number == "+15555559999"
        assert r.body == "Field check"
        assert r.sid == "sid_xyz"


# ---------------------------------------------------------------------------
# AC3: Handler produces a response / triggers expected downstream action
# ---------------------------------------------------------------------------


class TestHandlerDownstreamAction:
    def test_handler_responses_recorded(self) -> None:
        stub = SmsReceiverStub()
        stub.inject(from_number="+15555551111", body="Msg A")
        stub.inject(from_number="+15555552222", body="Msg B")
        assert len(stub.handler_responses) == 2

    def test_handler_response_matches_inject_order(self) -> None:
        stub = SmsReceiverStub()
        stub.inject(from_number="+15555551111", body="First")
        stub.inject(from_number="+15555552222", body="Second")
        assert stub.handler_responses[0].body == "First"
        assert stub.handler_responses[1].body == "Second"

    def test_handler_can_trigger_side_effect(self) -> None:
        """Handler can perform arbitrary downstream action (e.g. append to list)."""
        processed_bodies: list[str] = []

        def side_effect_handler(record: ReceivedSmsRecord) -> InboundSmsHandlerResult:
            processed_bodies.append(record.body)
            return InboundSmsHandlerResult(
                status="received",
                sid=record.sid,
                from_number=record.from_number,
                to_number=record.to_number,
                body=record.body,
            )

        stub = SmsReceiverStub(handler=side_effect_handler)
        stub.inject(from_number="+15555551111", body="Action A")
        stub.inject(from_number="+15555552222", body="Action B")
        assert processed_bodies == ["Action A", "Action B"]

    def test_inject_result_is_handler_result(self) -> None:
        stub = SmsReceiverStub()
        result = stub.inject(from_number="+15555551111", body="Match check")
        # inject() return value must match what was stored in handler_responses
        assert stub.handler_responses[-1] is result


# ---------------------------------------------------------------------------
# received_messages / clear helpers
# ---------------------------------------------------------------------------


class TestInboxAccessors:
    def test_received_messages_empty_initially(self) -> None:
        stub = SmsReceiverStub()
        assert stub.received_messages == []

    def test_received_messages_accumulates(self) -> None:
        stub = SmsReceiverStub()
        stub.inject(from_number="+15555551111", body="Msg 1")
        stub.inject(from_number="+15555552222", body="Msg 2")
        assert len(stub.received_messages) == 2

    def test_received_messages_returns_copy(self) -> None:
        stub = SmsReceiverStub()
        stub.inject(from_number="+15555551111", body="Hi")
        copy = stub.received_messages
        copy.clear()
        # Internal state must be unchanged
        assert len(stub.received_messages) == 1

    def test_handler_responses_returns_copy(self) -> None:
        stub = SmsReceiverStub()
        stub.inject(from_number="+15555551111", body="Hi")
        copy = stub.handler_responses
        copy.clear()
        assert len(stub.handler_responses) == 1

    def test_clear_resets_inbox(self) -> None:
        stub = SmsReceiverStub()
        stub.inject(from_number="+15555551111", body="Pre-clear")
        stub.clear()
        assert stub.received_messages == []

    def test_clear_resets_responses(self) -> None:
        stub = SmsReceiverStub()
        stub.inject(from_number="+15555551111", body="Pre-clear")
        stub.clear()
        assert stub.handler_responses == []

    def test_inject_after_clear_starts_fresh(self) -> None:
        stub = SmsReceiverStub()
        stub.inject(from_number="+15555551111", body="Before clear")
        stub.clear()
        stub.inject(from_number="+15555552222", body="After clear")
        assert len(stub.received_messages) == 1
        assert stub.received_messages[0].body == "After clear"


# ---------------------------------------------------------------------------
# No-network guarantee
# ---------------------------------------------------------------------------


class TestNoNetworkCalls:
    def test_inject_makes_no_network_calls(self) -> None:
        stub = SmsReceiverStub()
        with patch.object(socket.socket, "connect", _no_network):
            stub.inject(from_number="+15555551111", body="No network")
        # If we get here, no network calls were made
        assert len(stub.received_messages) == 1

    def test_custom_handler_inject_makes_no_network_calls(self) -> None:
        def simple_handler(record: ReceivedSmsRecord) -> InboundSmsHandlerResult:
            return InboundSmsHandlerResult(
                status="received",
                sid=record.sid,
                from_number=record.from_number,
                to_number=record.to_number,
                body=record.body,
            )

        stub = SmsReceiverStub(handler=simple_handler)
        with patch.object(socket.socket, "connect", _no_network):
            stub.inject(from_number="+15555551111", body="Custom no network")
        assert len(stub.received_messages) == 1


# ---------------------------------------------------------------------------
# ReceivedSmsRecord dataclass
# ---------------------------------------------------------------------------


class TestReceivedSmsRecord:
    def test_record_fields_accessible(self) -> None:
        ts = datetime(2026, 3, 11, 12, 0, 0, tzinfo=timezone.utc)
        r = ReceivedSmsRecord(
            sid="sid_001",
            from_number="+15555551111",
            to_number="+15555559999",
            body="Hello",
            received_at=ts,
        )
        assert r.sid == "sid_001"
        assert r.from_number == "+15555551111"
        assert r.to_number == "+15555559999"
        assert r.body == "Hello"
        assert r.received_at == ts

    def test_record_default_received_at_is_utc(self) -> None:
        before = datetime.now(timezone.utc)
        r = ReceivedSmsRecord(sid="s", from_number="+1", to_number="+2", body="b")
        after = datetime.now(timezone.utc)
        assert before <= r.received_at <= after
        assert r.received_at.tzinfo is not None


# ---------------------------------------------------------------------------
# InboundSmsHandlerResult dataclass
# ---------------------------------------------------------------------------


class TestInboundSmsHandlerResult:
    def test_result_status_received(self) -> None:
        r = InboundSmsHandlerResult(
            status="received",
            sid="sid_001",
            from_number="+1",
            to_number="+2",
            body="Hi",
        )
        assert r.status == "received"
        assert r.error is None

    def test_result_status_error(self) -> None:
        r = InboundSmsHandlerResult(
            status="error",
            sid="sid_002",
            from_number="+1",
            to_number="+2",
            body="Hi",
            error="Something went wrong",
        )
        assert r.status == "error"
        assert r.error == "Something went wrong"
