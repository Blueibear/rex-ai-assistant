"""Tests for the notification email channel wiring.

Verifies that:
- ``_send_to_email`` calls ``EmailService.send()`` when a backend is active
- Account selection via ``email_account_id`` metadata key works
- Stub mode (no backend) still logs without error
- Error propagation works correctly
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rex.notification import NotificationRequest, Notifier


@pytest.fixture
def notifier(tmp_path):
    """Create a Notifier with temporary storage."""
    path = tmp_path / "notif"
    path.mkdir()
    return Notifier(storage_path=path)


class TestNotificationEmailChannel:
    def test_email_channel_calls_send_with_backend(self, notifier):
        """When backend is set and to_email is present, send() is called."""
        mock_service = MagicMock()
        mock_service.active_backend = MagicMock()  # non-None → has backend
        mock_service.send.return_value = {"ok": True, "message_id": "m1", "error": None}

        notif = NotificationRequest(
            priority="normal",
            title="Test Alert",
            body="Something happened",
            channel_preferences=["email"],
            metadata={"to_email": "user@example.com"},
        )

        with patch("rex.email_service.get_email_service", return_value=mock_service):
            notifier._send_to_email(notif)

        mock_service.send.assert_called_once_with(
            to="user@example.com",
            subject="Test Alert",
            body="Something happened",
            account_id=None,
        )

    def test_email_channel_with_account_id(self, notifier):
        """email_account_id metadata is passed to send()."""
        mock_service = MagicMock()
        mock_service.active_backend = MagicMock()
        mock_service.send.return_value = {"ok": True, "message_id": "m1", "error": None}

        notif = NotificationRequest(
            priority="normal",
            title="Alert",
            body="Body",
            channel_preferences=["email"],
            metadata={
                "to_email": "user@example.com",
                "email_account_id": "work",
            },
        )

        with patch("rex.email_service.get_email_service", return_value=mock_service):
            notifier._send_to_email(notif)

        mock_service.send.assert_called_once_with(
            to="user@example.com",
            subject="Alert",
            body="Body",
            account_id="work",
        )

    def test_email_channel_stub_mode_no_backend(self, notifier):
        """When no backend, logs and does not raise."""
        mock_service = MagicMock()
        mock_service.active_backend = None  # stub mode

        notif = NotificationRequest(
            priority="normal",
            title="Stub Alert",
            body="Logged only",
            channel_preferences=["email"],
        )

        with patch("rex.email_service.get_email_service", return_value=mock_service):
            notifier._send_to_email(notif)  # should not raise

        mock_service.send.assert_not_called()

    def test_email_channel_no_to_email_uses_stub(self, notifier):
        """When to_email is absent, falls back to stub log path."""
        mock_service = MagicMock()
        mock_service.active_backend = MagicMock()

        notif = NotificationRequest(
            priority="normal",
            title="No Recipient",
            body="Who gets this?",
            channel_preferences=["email"],
            metadata={},  # no to_email
        )

        with patch("rex.email_service.get_email_service", return_value=mock_service):
            notifier._send_to_email(notif)

        mock_service.send.assert_not_called()

    def test_email_channel_send_failure_raises(self, notifier):
        """When send() returns ok=False, RuntimeError is raised."""
        mock_service = MagicMock()
        mock_service.active_backend = MagicMock()
        mock_service.send.return_value = {"ok": False, "message_id": None, "error": "relay denied"}

        notif = NotificationRequest(
            priority="normal",
            title="Fail",
            body="Oops",
            channel_preferences=["email"],
            metadata={"to_email": "user@example.com"},
        )

        with patch("rex.email_service.get_email_service", return_value=mock_service):
            with pytest.raises(RuntimeError, match="relay denied"):
                notifier._send_to_email(notif)

    def test_urgent_notification_sends_to_email_channel(self, notifier):
        """Urgent notifications reach the email channel when listed."""
        mock_service = MagicMock()
        mock_service.active_backend = MagicMock()
        mock_service.send.return_value = {"ok": True, "message_id": "m1", "error": None}

        notif = NotificationRequest(
            priority="urgent",
            title="Urgent",
            body="Do something",
            channel_preferences=["email", "dashboard"],
            metadata={"to_email": "admin@example.com"},
        )

        with patch("rex.email_service.get_email_service", return_value=mock_service):
            with patch.object(notifier, "_send_to_dashboard"):
                notifier.send(notif)

        mock_service.send.assert_called_once()

    def test_digest_flush_uses_email_channel(self, notifier):
        """Flushed digests route through the email channel."""
        notif = NotificationRequest(
            priority="digest",
            title="Digest Item",
            body="Something",
            channel_preferences=["email"],
        )
        notifier.send(notif)

        with patch.object(notifier, "_dispatch_to_channel") as mock_dispatch:
            notifier.flush_digests()
            mock_dispatch.assert_called_once()
            assert mock_dispatch.call_args[0][0] == "email"
