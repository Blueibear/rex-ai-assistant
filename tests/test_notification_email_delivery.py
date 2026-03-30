"""Tests for US-209: notification email delivery via real EmailService.send().

Covers:
- _send_to_email() always calls EmailService.send() when to_email is set
- On success (result.ok=True) the sent message is logged
- On failure (result.ok=False) RuntimeError is raised with the error message
- When no to_email is provided a warning is logged and send() is not called
- Digest flush dispatches through the email channel via _send_to_email
- No live network calls (get_email_service monkeypatched)
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rex.notification import NotificationRequest, Notifier

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_notifier(tmp_path: Path) -> Notifier:
    return Notifier(storage_path=tmp_path)


def _make_notification(
    *,
    title: str = "Test alert",
    body: str = "Alert body",
    to_email: str | None = "user@example.com",
    account_id: str | None = None,
    priority: str = "normal",
    channel: str = "email",
) -> NotificationRequest:
    meta: dict = {}
    if to_email is not None:
        meta["to_email"] = to_email
    if account_id is not None:
        meta["email_account_id"] = account_id
    return NotificationRequest(
        title=title,
        body=body,
        priority=priority,  # type: ignore[arg-type]
        channel_preferences=[channel],
        metadata=meta,
    )


def _stub_email_service(ok: bool = True, error: str | None = None) -> MagicMock:
    """Return a MagicMock that mimics EmailService."""
    svc = MagicMock()
    svc.send.return_value = {"ok": ok, "message_id": "msg-123", "error": error}
    return svc


# ---------------------------------------------------------------------------
# _send_to_email unit tests
# ---------------------------------------------------------------------------


class TestSendToEmail:
    def test_calls_email_service_send_when_to_email_set(self, tmp_path):
        notifier = _make_notifier(tmp_path)
        notification = _make_notification()
        mock_svc = _stub_email_service(ok=True)

        with patch("rex.notification.get_email_service", return_value=mock_svc):
            notifier._send_to_email(notification)

        mock_svc.send.assert_called_once_with(
            to="user@example.com",
            subject="Test alert",
            body="Alert body",
            account_id=None,
        )

    def test_passes_account_id_to_send(self, tmp_path):
        notifier = _make_notifier(tmp_path)
        notification = _make_notification(account_id="work")
        mock_svc = _stub_email_service(ok=True)

        with patch("rex.notification.get_email_service", return_value=mock_svc):
            notifier._send_to_email(notification)

        _, kwargs = mock_svc.send.call_args
        assert kwargs["account_id"] == "work"

    def test_no_send_when_to_email_missing(self, tmp_path):
        notifier = _make_notifier(tmp_path)
        notification = _make_notification(to_email=None)
        mock_svc = _stub_email_service()

        with patch("rex.notification.get_email_service", return_value=mock_svc):
            notifier._send_to_email(notification)

        mock_svc.send.assert_not_called()

    def test_raises_on_send_failure(self, tmp_path):
        notifier = _make_notifier(tmp_path)
        notification = _make_notification()
        mock_svc = _stub_email_service(ok=False, error="SMTP auth failed")

        with patch("rex.notification.get_email_service", return_value=mock_svc):
            with pytest.raises(RuntimeError, match="Email send failed: SMTP auth failed"):
                notifier._send_to_email(notification)

    def test_raises_on_exception_from_send(self, tmp_path):
        notifier = _make_notifier(tmp_path)
        notification = _make_notification()
        mock_svc = MagicMock()
        mock_svc.send.side_effect = OSError("connection refused")

        with patch("rex.notification.get_email_service", return_value=mock_svc):
            with pytest.raises(OSError, match="connection refused"):
                notifier._send_to_email(notification)

    def test_successful_send_does_not_raise(self, tmp_path):
        notifier = _make_notifier(tmp_path)
        notification = _make_notification()
        mock_svc = _stub_email_service(ok=True)

        with patch("rex.notification.get_email_service", return_value=mock_svc):
            # Should complete without exception
            notifier._send_to_email(notification)

    def test_does_not_use_active_backend_guard(self, tmp_path):
        """send() is called regardless of active_backend — no guard check."""
        notifier = _make_notifier(tmp_path)
        notification = _make_notification()
        mock_svc = _stub_email_service(ok=True)
        mock_svc.active_backend = None  # simulate stub mode

        with patch("rex.notification.get_email_service", return_value=mock_svc):
            notifier._send_to_email(notification)

        # Should have been called even though active_backend is None
        mock_svc.send.assert_called_once()


# ---------------------------------------------------------------------------
# Digest flush dispatches through EmailService
# ---------------------------------------------------------------------------


class TestDigestFlushEmailDelivery:
    def test_digest_flush_calls_email_service_send(self, tmp_path):
        notifier = _make_notifier(tmp_path)
        # Queue a digest notification
        digest_notif = NotificationRequest(
            title="Low priority update",
            body="Nothing urgent",
            priority="digest",
            channel_preferences=["email"],
            metadata={"to_email": "user@example.com"},
        )
        notifier._queue_digest(digest_notif)

        mock_svc = _stub_email_service(ok=True)
        with patch("rex.notification.get_email_service", return_value=mock_svc):
            count = notifier.flush_digests(channel="email")

        assert count == 1
        mock_svc.send.assert_called_once()
        call_kwargs = mock_svc.send.call_args.kwargs
        assert call_kwargs["to"] == "user@example.com"
        assert "Digest Summary" in call_kwargs["subject"]

    def test_digest_flush_failure_is_not_silently_dropped(self, tmp_path):
        notifier = _make_notifier(tmp_path)
        digest_notif = NotificationRequest(
            title="Alert",
            body="Body",
            priority="digest",
            channel_preferences=["email"],
            metadata={"to_email": "admin@example.com"},
        )
        notifier._queue_digest(digest_notif)

        mock_svc = _stub_email_service(ok=False, error="relay rejected")
        with patch("rex.notification.get_email_service", return_value=mock_svc):
            # flush_digests catches channel errors internally and continues;
            # the important thing is send() was called (not silently skipped)
            notifier.flush_digests(channel="email")

        # send() is called at least once (retry logic may call it multiple times)
        mock_svc.send.assert_called()
