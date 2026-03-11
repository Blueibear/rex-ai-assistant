"""Tests for US-086: TwilioAdapter Protocol.

Acceptance criteria:
- TwilioAdapter abstract class or Protocol defined with at minimum
  send_sms(to: str, body: str) signature
- SmsSenderStub fully implements TwilioAdapter
- Swapping stub for a real Twilio client requires no changes outside
  the adapter registration point
- Typecheck passes
"""

from __future__ import annotations

import socket
from typing import Any
from unittest.mock import patch

import pytest

from rex.messaging_backends.twilio_adapter import TwilioAdapter
from rex.messaging_backends.sms_sender_stub import SmsSenderStub


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _MinimalAdapter:
    """Minimal class that satisfies TwilioAdapter via structural subtyping."""

    def send_sms(self, to: str, body: str) -> dict[str, Any]:
        return {"ok": True, "sid": "minimal", "to": to, "body": body}


class _NonAdapter:
    """Class that does NOT implement TwilioAdapter (wrong method name)."""

    def send(self, to: str, body: str) -> dict[str, Any]:
        return {"ok": True}


# ---------------------------------------------------------------------------
# AC1: TwilioAdapter Protocol defined with send_sms(to, body) signature
# ---------------------------------------------------------------------------


class TestTwilioAdapterProtocol:
    def test_protocol_is_importable(self):
        assert TwilioAdapter is not None

    def test_protocol_is_runtime_checkable(self):
        """isinstance() should work for runtime_checkable Protocol."""
        adapter = _MinimalAdapter()
        assert isinstance(adapter, TwilioAdapter)

    def test_non_adapter_fails_isinstance(self):
        """A class without send_sms must not satisfy the Protocol."""
        bad = _NonAdapter()
        assert not isinstance(bad, TwilioAdapter)

    def test_send_sms_signature_present_in_protocol(self):
        """Protocol must expose send_sms as a method."""
        assert hasattr(TwilioAdapter, "send_sms")
        assert callable(TwilioAdapter.send_sms)

    def test_protocol_documented(self):
        assert TwilioAdapter.__doc__

    def test_minimal_adapter_send_sms_returns_dict(self):
        adapter = _MinimalAdapter()
        result = adapter.send_sms(to="+15550001234", body="Hello")
        assert isinstance(result, dict)
        assert result["ok"] is True


# ---------------------------------------------------------------------------
# AC2: SmsSenderStub fully implements TwilioAdapter
# ---------------------------------------------------------------------------


class TestSmsSenderStubImplementsTwilioAdapter:
    def test_stub_is_instance_of_protocol(self):
        stub = SmsSenderStub()
        assert isinstance(stub, TwilioAdapter)

    def test_stub_has_send_sms_method(self):
        stub = SmsSenderStub()
        assert hasattr(stub, "send_sms")
        assert callable(stub.send_sms)

    def test_send_sms_returns_ok_true(self):
        stub = SmsSenderStub()
        result = stub.send_sms(to="+15550001234", body="Test message")
        assert result["ok"] is True

    def test_send_sms_echoes_to_number(self):
        stub = SmsSenderStub()
        result = stub.send_sms(to="+15550009999", body="Echo test")
        assert result["to"] == "+15550009999"

    def test_send_sms_echoes_body(self):
        stub = SmsSenderStub()
        result = stub.send_sms(to="+15550001234", body="Echo body")
        assert result["body"] == "Echo body"

    def test_send_sms_returns_sid(self):
        stub = SmsSenderStub()
        result = stub.send_sms(to="+15550001234", body="SID check")
        assert "sid" in result
        assert result["sid"].startswith("stub_")

    def test_send_sms_records_message_in_log(self):
        stub = SmsSenderStub()
        stub.send_sms(to="+15550001234", body="Logged?")
        assert len(stub.sent_messages) == 1
        assert stub.sent_messages[0].body == "Logged?"

    def test_send_sms_multiple_calls_all_logged(self):
        stub = SmsSenderStub()
        for i in range(5):
            stub.send_sms(to=f"+1555000{i:04d}", body=f"msg {i}")
        assert len(stub.sent_messages) == 5

    def test_send_sms_and_send_share_log(self):
        """send_sms delegates to send — both should populate the same log."""
        stub = SmsSenderStub()
        stub.send(to="+15550001111", body="via send")
        stub.send_sms(to="+15550002222", body="via send_sms")
        assert len(stub.sent_messages) == 2

    def test_send_sms_makes_no_network_calls(self):
        stub = SmsSenderStub()
        with patch.object(socket.socket, "connect", side_effect=AssertionError("network call!")):
            stub.send_sms(to="+15550001234", body="No network")

    def test_send_sms_empty_body(self):
        stub = SmsSenderStub()
        result = stub.send_sms(to="+15550001234", body="")
        assert result["ok"] is True
        assert result["body"] == ""

    def test_send_sms_long_body(self):
        stub = SmsSenderStub()
        long_body = "x" * 10_000
        result = stub.send_sms(to="+15550001234", body=long_body)
        assert result["ok"] is True
        assert stub.sent_messages[0].body == long_body

    def test_send_sms_unique_sids(self):
        stub = SmsSenderStub()
        r1 = stub.send_sms(to="+15550001234", body="a")
        r2 = stub.send_sms(to="+15550001234", body="b")
        assert r1["sid"] != r2["sid"]

    def test_clear_resets_after_send_sms(self):
        stub = SmsSenderStub()
        stub.send_sms(to="+15550001234", body="before clear")
        stub.clear()
        assert stub.sent_messages == []


# ---------------------------------------------------------------------------
# AC3: Swapping stub requires no changes outside adapter registration point
# ---------------------------------------------------------------------------


class TestAdapterSwappability:
    """Verify that calling code typed as TwilioAdapter works identically
    whether backed by the stub or any other compliant adapter."""

    def _send_via_adapter(self, adapter: TwilioAdapter, to: str, body: str) -> dict[str, Any]:
        """Simulate calling code that only knows about TwilioAdapter."""
        return adapter.send_sms(to=to, body=body)

    def test_stub_usable_through_protocol_typed_variable(self):
        adapter: TwilioAdapter = SmsSenderStub()
        result = self._send_via_adapter(adapter, "+15550001234", "Protocol typed call")
        assert result["ok"] is True

    def test_minimal_adapter_usable_through_protocol_typed_variable(self):
        adapter: TwilioAdapter = _MinimalAdapter()
        result = self._send_via_adapter(adapter, "+15550001234", "Minimal adapter call")
        assert result["ok"] is True

    def test_registration_swap_does_not_change_caller(self):
        """Registry-style swap: register stub, call through registry, swap to minimal."""

        registry: dict[str, TwilioAdapter] = {}

        def register(name: str, adapter: TwilioAdapter) -> None:
            registry[name] = adapter

        def send_via_registry(name: str, to: str, body: str) -> dict[str, Any]:
            return registry[name].send_sms(to=to, body=body)

        # First: register stub
        register("default", SmsSenderStub())
        result1 = send_via_registry("default", "+15550001234", "from stub")
        assert result1["ok"] is True

        # Swap: register minimal adapter — calling code unchanged
        register("default", _MinimalAdapter())
        result2 = send_via_registry("default", "+15550001234", "from minimal")
        assert result2["ok"] is True

    def test_isinstance_check_allows_type_narrowing(self):
        adapters: list[object] = [SmsSenderStub(), _MinimalAdapter()]
        for obj in adapters:
            assert isinstance(obj, TwilioAdapter), f"{obj!r} must satisfy TwilioAdapter"

    def test_non_adapter_excluded_from_registry(self):
        bad = _NonAdapter()
        assert not isinstance(bad, TwilioAdapter)


# ---------------------------------------------------------------------------
# Module-level import / export checks
# ---------------------------------------------------------------------------


class TestPackageExports:
    def test_twilio_adapter_exported_from_package(self):
        from rex.messaging_backends import TwilioAdapter as TA  # noqa: N811

        assert TA is TwilioAdapter

    def test_sms_sender_stub_exported_from_package(self):
        from rex.messaging_backends import SmsSenderStub as SSS  # noqa: N811

        assert SSS is SmsSenderStub
