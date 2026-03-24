"""Unit tests for rex.integrations.message_router — routing logic."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rex.integrations.message_router import (
    EmailChannel,
    MessageResult,
    MessageRouter,
    NoChannelError,
    SMSChannel,
)
from rex.integrations.models import SMSMessage

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sms_result(msg_id: str = "sms-001", status: str = "stub") -> SMSMessage:
    return SMSMessage(
        id=msg_id,
        thread_id="thread-001",
        direction="outbound",
        body="test",
        from_number="+1",
        to_number="+2",
        status=status,  # type: ignore[arg-type]
    )


def _mock_sms(msg_id: str = "sms-001", status: str = "stub") -> SMSChannel:
    mock = MagicMock(spec=SMSChannel)
    mock.send.return_value = _make_sms_result(msg_id, status)
    return mock  # type: ignore[return-value]


def _mock_email(msg_id: str = "email-001") -> EmailChannel:
    result = MagicMock()
    result.id = msg_id
    mock = MagicMock(spec=EmailChannel)
    mock.send_draft.return_value = result
    return mock  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Basic routing
# ---------------------------------------------------------------------------


class TestRouting:
    def test_routes_to_email_when_email_only(self) -> None:
        router = MessageRouter(email=_mock_email())
        result = router.route("user@example.com", "Hello")
        assert result.channel == "email"

    def test_routes_to_sms_when_sms_only(self) -> None:
        router = MessageRouter(sms=_mock_sms())
        result = router.route("+14155550100", "Hello")
        assert result.channel == "sms"

    def test_prefers_email_when_both_configured(self) -> None:
        router = MessageRouter(sms=_mock_sms(), email=_mock_email())
        result = router.route("user@example.com", "Hello")
        assert result.channel == "email"

    def test_raises_when_neither_configured(self) -> None:
        router = MessageRouter()
        with pytest.raises(NoChannelError):
            router.route("user@example.com", "Hello")

    def test_returns_message_result(self) -> None:
        router = MessageRouter(sms=_mock_sms("sms-xyz", "stub"))
        result = router.route("+14155550100", "Hello")
        assert isinstance(result, MessageResult)
        assert result.message_id == "sms-xyz"
        assert result.status == "stub"


# ---------------------------------------------------------------------------
# Explicit preferred_channel
# ---------------------------------------------------------------------------


class TestPreferredChannel:
    def test_explicit_sms(self) -> None:
        router = MessageRouter(sms=_mock_sms(), email=_mock_email())
        result = router.route("+14155550100", "Hello", preferred_channel="sms")
        assert result.channel == "sms"

    def test_explicit_email(self) -> None:
        router = MessageRouter(sms=_mock_sms(), email=_mock_email())
        result = router.route("user@example.com", "Hello", preferred_channel="email")
        assert result.channel == "email"

    def test_unknown_channel_raises_value_error(self) -> None:
        router = MessageRouter(sms=_mock_sms())
        with pytest.raises(ValueError):
            router.route("+1", "Hello", preferred_channel="fax")

    def test_sms_channel_requested_but_not_configured_raises(self) -> None:
        router = MessageRouter(email=_mock_email())
        with pytest.raises(NoChannelError):
            router.route("+1", "Hello", preferred_channel="sms")

    def test_email_channel_requested_but_not_configured_raises(self) -> None:
        router = MessageRouter(sms=_mock_sms())
        with pytest.raises(NoChannelError):
            router.route("user@example.com", "Hello", preferred_channel="email")


# ---------------------------------------------------------------------------
# Contacts lookup
# ---------------------------------------------------------------------------


class TestContactsLookup:
    def test_uses_contact_preferred_sms(self) -> None:
        contacts = [{"name": "Alice", "number": "+14155550101", "preferred_channel": "sms"}]
        contacts_data = json.dumps(contacts)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            f.write(contacts_data)
            tmp_path = Path(f.name)

        import rex.integrations.message_router as mod

        with patch.object(mod, "_CONTACTS_PATH", tmp_path):
            router = MessageRouter(sms=_mock_sms(), email=_mock_email())
            result = router.route("alice", "Hi")
            assert result.channel == "sms"

        tmp_path.unlink(missing_ok=True)

    def test_uses_contact_preferred_email(self) -> None:
        contacts = [{"name": "Bob", "number": "+14155550202", "preferred_channel": "email"}]
        contacts_data = json.dumps(contacts)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            f.write(contacts_data)
            tmp_path = Path(f.name)

        import rex.integrations.message_router as mod

        with patch.object(mod, "_CONTACTS_PATH", tmp_path):
            router = MessageRouter(sms=_mock_sms(), email=_mock_email())
            result = router.route("bob", "Hi")
            assert result.channel == "email"

        tmp_path.unlink(missing_ok=True)

    def test_falls_back_when_contacts_file_absent(self) -> None:
        import rex.integrations.message_router as mod

        absent_path = Path("/nonexistent/contacts.json")
        with patch.object(mod, "_CONTACTS_PATH", absent_path):
            router = MessageRouter(sms=_mock_sms())
            result = router.route("+14155550100", "Hi")
            assert result.channel == "sms"
