"""Integration tests for the SMS pipeline (SMSService + MessageRouter) — stub mode.

All tests run without real Twilio credentials.  The Twilio client is mocked
so that even if twilio is installed no real API calls are made.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from rex.integrations.message_router import (
    EmailChannel,
    MessageRouter,
    NoChannelError,
)
from rex.integrations.models import SMSMessage
from rex.integrations.sms_service import SMSService

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _stub_sms_service() -> SMSService:
    return SMSService(sms_provider="none")


def _mock_email_channel(msg_id: str = "email-001") -> EmailChannel:
    result = MagicMock()
    result.id = msg_id
    mock = MagicMock(spec=EmailChannel)
    mock.send_draft.return_value = result
    return mock  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# SMSService stub mode send
# ---------------------------------------------------------------------------


class TestSMSServiceStubSend:
    def test_send_returns_sms_message(self) -> None:
        svc = _stub_sms_service()
        msg = svc.send("+14155559999", "Hello")
        assert isinstance(msg, SMSMessage)

    def test_send_status_is_stub(self) -> None:
        svc = _stub_sms_service()
        msg = svc.send("+14155559999", "Hello")
        assert msg.status == "stub"

    def test_send_direction_outbound(self) -> None:
        svc = _stub_sms_service()
        msg = svc.send("+14155559999", "Hello")
        assert msg.direction == "outbound"

    def test_send_to_number_preserved(self) -> None:
        svc = _stub_sms_service()
        msg = svc.send("+14155559999", "Hello")
        assert msg.to_number == "+14155559999"

    def test_send_body_preserved(self) -> None:
        svc = _stub_sms_service()
        msg = svc.send("+14155559999", "Test message content")
        assert msg.body == "Test message content"

    def test_stub_mode_does_not_call_twilio_client(self) -> None:
        """Stub mode should never reach the twilio.rest.Client path."""
        with patch("rex.integrations.sms_service.SMSService._twilio_send") as mock_twilio:
            svc = SMSService(sms_provider="none")
            svc.send("+14155559999", "Hello")
            mock_twilio.assert_not_called()


# ---------------------------------------------------------------------------
# MessageRouter routing integration
# ---------------------------------------------------------------------------


class TestMessageRouterIntegration:
    def test_routes_to_sms_when_sms_only(self) -> None:
        svc = _stub_sms_service()
        router = MessageRouter(sms=svc)
        result = router.route("+14155559999", "Hello via SMS")
        assert result.channel == "sms"
        assert result.status == "stub"

    def test_routes_to_email_when_email_only(self) -> None:
        email = _mock_email_channel("email-xyz")
        router = MessageRouter(email=email)
        result = router.route("user@example.com", "Hello via Email")
        assert result.channel == "email"
        assert result.message_id == "email-xyz"

    def test_raises_no_channel_error_when_neither_configured(self) -> None:
        router = MessageRouter()
        with pytest.raises(NoChannelError):
            router.route("user@example.com", "Hello")

    def test_explicit_sms_uses_sms_service(self) -> None:
        svc = _stub_sms_service()
        router = MessageRouter(sms=svc, email=_mock_email_channel())
        result = router.route("+14155559999", "Force SMS", preferred_channel="sms")
        assert result.channel == "sms"

    def test_explicit_email_uses_email_service(self) -> None:
        email = _mock_email_channel()
        router = MessageRouter(sms=_stub_sms_service(), email=email)
        result = router.route("user@example.com", "Force email", preferred_channel="email")
        assert result.channel == "email"

    def test_sms_message_id_propagated(self) -> None:
        svc = _stub_sms_service()
        router = MessageRouter(sms=svc)
        result = router.route("+14155559999", "Hello")
        assert result.message_id != ""

    def test_end_to_end_sms_pipeline(self) -> None:
        """Full pipeline: stub SMSService → MessageRouter → MessageResult."""
        svc = _stub_sms_service()
        router = MessageRouter(sms=svc)
        result = router.route("+14155559999", "End-to-end test message")
        assert result.channel == "sms"
        assert result.status == "stub"
        assert isinstance(result.message_id, str)
        assert len(result.message_id) > 0
