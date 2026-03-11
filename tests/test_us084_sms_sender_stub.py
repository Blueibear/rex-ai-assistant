"""Tests for US-084: SMS send stub.

Acceptance criteria:
- SmsSenderStub class accepts a phone number and message body
- sent messages written to a structured in-memory log accessible for test assertions
- stub implements the same interface as the real Twilio adapter (US-086)
- calling send on the stub makes no network calls
- Typecheck passes
"""

from __future__ import annotations

import socket
from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from rex.messaging_backends.sms_sender_stub import SentSmsRecord, SmsSenderStub


# ---------------------------------------------------------------------------
# Basic instantiation
# ---------------------------------------------------------------------------


def test_stub_instantiates_with_defaults() -> None:
    stub = SmsSenderStub()
    assert isinstance(stub, SmsSenderStub)


def test_stub_instantiates_with_custom_from() -> None:
    stub = SmsSenderStub(default_from="+12025551234")
    assert stub._default_from == "+12025551234"


# ---------------------------------------------------------------------------
# AC: SmsSenderStub accepts a phone number and message body
# ---------------------------------------------------------------------------


def test_send_accepts_to_and_body() -> None:
    stub = SmsSenderStub()
    result = stub.send(to="+15555550001", body="Hello world")
    assert result["ok"] is True


def test_send_echoes_to_and_body_in_result() -> None:
    stub = SmsSenderStub()
    result = stub.send(to="+15555550002", body="Test message")
    assert result["to"] == "+15555550002"
    assert result["body"] == "Test message"


def test_send_returns_sid() -> None:
    stub = SmsSenderStub()
    result = stub.send(to="+15555550003", body="SID test")
    assert "sid" in result
    assert isinstance(result["sid"], str)
    assert len(result["sid"]) > 0


def test_send_returns_ok_true() -> None:
    stub = SmsSenderStub()
    result = stub.send(to="+15555550004", body="OK test")
    assert result["ok"] is True


# ---------------------------------------------------------------------------
# AC: sent messages written to a structured in-memory log
# ---------------------------------------------------------------------------


def test_sent_messages_initially_empty() -> None:
    stub = SmsSenderStub()
    assert stub.sent_messages == []


def test_sent_messages_contains_record_after_send() -> None:
    stub = SmsSenderStub()
    stub.send(to="+15555550010", body="First message")
    assert len(stub.sent_messages) == 1


def test_sent_messages_record_has_correct_to() -> None:
    stub = SmsSenderStub()
    stub.send(to="+15555550011", body="To test")
    assert stub.sent_messages[0].to == "+15555550011"


def test_sent_messages_record_has_correct_body() -> None:
    stub = SmsSenderStub()
    stub.send(to="+15555550012", body="Body test")
    assert stub.sent_messages[0].body == "Body test"


def test_sent_messages_record_has_sid() -> None:
    stub = SmsSenderStub()
    stub.send(to="+15555550013", body="SID log test")
    record = stub.sent_messages[0]
    assert isinstance(record.sid, str)
    assert record.sid.startswith("stub_")


def test_sent_messages_record_has_timestamp() -> None:
    stub = SmsSenderStub()
    stub.send(to="+15555550014", body="Timestamp test")
    record = stub.sent_messages[0]
    assert isinstance(record.sent_at, datetime)
    assert record.sent_at.tzinfo is not None


def test_sent_messages_record_is_sent_sms_record() -> None:
    stub = SmsSenderStub()
    stub.send(to="+15555550015", body="Type test")
    assert isinstance(stub.sent_messages[0], SentSmsRecord)


def test_multiple_sends_accumulate_in_log() -> None:
    stub = SmsSenderStub()
    stub.send(to="+15555550020", body="Message 1")
    stub.send(to="+15555550021", body="Message 2")
    stub.send(to="+15555550022", body="Message 3")
    assert len(stub.sent_messages) == 3


def test_multiple_sends_preserve_order() -> None:
    stub = SmsSenderStub()
    stub.send(to="+15555550030", body="Alpha")
    stub.send(to="+15555550031", body="Beta")
    messages = stub.sent_messages
    assert messages[0].body == "Alpha"
    assert messages[1].body == "Beta"


def test_sent_messages_returns_copy_not_reference() -> None:
    stub = SmsSenderStub()
    stub.send(to="+15555550040", body="Copy test")
    copy = stub.sent_messages
    copy.clear()
    # Original log should be unaffected
    assert len(stub.sent_messages) == 1


def test_sids_are_unique_across_calls() -> None:
    stub = SmsSenderStub()
    r1 = stub.send(to="+15555550050", body="Msg1")
    r2 = stub.send(to="+15555550051", body="Msg2")
    assert r1["sid"] != r2["sid"]


def test_clear_empties_log() -> None:
    stub = SmsSenderStub()
    stub.send(to="+15555550060", body="Before clear")
    stub.clear()
    assert stub.sent_messages == []


def test_send_after_clear_works() -> None:
    stub = SmsSenderStub()
    stub.send(to="+15555550070", body="Pre-clear")
    stub.clear()
    stub.send(to="+15555550071", body="Post-clear")
    assert len(stub.sent_messages) == 1
    assert stub.sent_messages[0].body == "Post-clear"


# ---------------------------------------------------------------------------
# AC: calling send makes no network calls
# ---------------------------------------------------------------------------


def test_send_makes_no_network_calls() -> None:
    """Verify send() raises if a real socket connection is attempted."""
    stub = SmsSenderStub()

    original_connect = socket.socket.connect

    def raise_on_connect(self: socket.socket, address: object) -> None:  # type: ignore[override]
        raise AssertionError(f"Unexpected network call to {address}")

    with patch.object(socket.socket, "connect", raise_on_connect):
        # Should not raise — no network call should be attempted
        result = stub.send(to="+15555550080", body="No network test")
    assert result["ok"] is True


def test_send_long_body_no_network() -> None:
    stub = SmsSenderStub()
    long_body = "x" * 1600  # longer than a standard SMS but valid for stub
    with patch.object(socket.socket, "connect", side_effect=AssertionError("no network")):
        result = stub.send(to="+15555550081", body=long_body)
    assert result["ok"] is True


# ---------------------------------------------------------------------------
# AC: stub implements the same interface as the real Twilio adapter (US-086)
# ---------------------------------------------------------------------------


def test_stub_has_send_method() -> None:
    stub = SmsSenderStub()
    assert callable(getattr(stub, "send", None))


def test_send_signature_accepts_positional_to_and_body() -> None:
    stub = SmsSenderStub()
    # Must accept (to, body) positionally
    result = stub.send("+15555550090", "Positional test")
    assert result["ok"] is True


def test_send_signature_accepts_keyword_to_and_body() -> None:
    stub = SmsSenderStub()
    result = stub.send(to="+15555550091", body="Keyword test")
    assert result["ok"] is True


def test_stub_has_sent_messages_property() -> None:
    stub = SmsSenderStub()
    assert hasattr(stub, "sent_messages")


def test_stub_has_clear_method() -> None:
    stub = SmsSenderStub()
    assert callable(getattr(stub, "clear", None))


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_send_empty_body() -> None:
    stub = SmsSenderStub()
    result = stub.send(to="+15555550100", body="")
    assert result["ok"] is True
    assert stub.sent_messages[0].body == ""


def test_send_unicode_body() -> None:
    stub = SmsSenderStub()
    result = stub.send(to="+15555550101", body="Héllo Wörld \U0001f600")
    assert result["ok"] is True
    assert "Héllo" in stub.sent_messages[0].body


def test_sent_at_is_utc() -> None:
    stub = SmsSenderStub()
    stub.send(to="+15555550110", body="UTC test")
    record = stub.sent_messages[0]
    assert record.sent_at.tzinfo == timezone.utc
