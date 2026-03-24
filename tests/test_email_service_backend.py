"""Tests for EmailService backend integration and send().

Verifies:
- EmailService delegates to backend when set
- send() works in stub mode and backend mode
- Backend lifecycle (set_backend, active_backend)
- Envelope-to-summary conversion
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from rex.email_backends.base import EmailEnvelope, SendResult
from rex.email_backends.stub import StubEmailBackend
from rex.email_service import EmailService, EmailSummary


@pytest.fixture
def mock_emails_file(tmp_path):
    data = [
        {
            "id": "e1",
            "from_addr": "a@b.com",
            "subject": "Hello",
            "snippet": "Hi",
            "received_at": "2026-01-01T10:00:00",
            "labels": ["unread"],
            "importance_score": 0.9,
        },
    ]
    path = tmp_path / "mock_emails.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


class TestEmailServiceBackend:
    def test_default_stub_mode(self, mock_emails_file):
        """Without a backend, EmailService uses stub mode."""
        svc = EmailService(mock_data_file=mock_emails_file)
        assert svc.active_backend is None
        assert svc.connect() is True
        unread = svc.fetch_unread()
        assert len(unread) == 1

    def test_set_backend(self, mock_emails_file):
        """set_backend swaps the active backend."""
        svc = EmailService(mock_data_file=mock_emails_file)
        mock_backend = MagicMock()
        svc.set_backend(mock_backend)
        assert svc.active_backend is mock_backend

    def test_connect_delegates_to_backend(self, mock_emails_file):
        mock_backend = MagicMock()
        mock_backend.connect.return_value = True

        svc = EmailService(mock_data_file=mock_emails_file, backend=mock_backend)
        assert svc.connect() is True
        mock_backend.connect.assert_called_once()

    def test_fetch_unread_delegates_to_backend(self, mock_emails_file):
        env = EmailEnvelope(
            message_id="msg-1",
            from_addr="sender@x.com",
            subject="Backend Msg",
            snippet="Hello from backend",
            received_at=datetime(2026, 1, 15, tzinfo=timezone.utc),
            labels=["unread"],
        )
        mock_backend = MagicMock()
        mock_backend.connect.return_value = True
        mock_backend.fetch_unread.return_value = [env]

        svc = EmailService(mock_data_file=mock_emails_file, backend=mock_backend)
        svc.connect()
        result = svc.fetch_unread()

        assert len(result) == 1
        assert isinstance(result[0], EmailSummary)
        assert result[0].id == "msg-1"
        assert result[0].from_addr == "sender@x.com"

    def test_mark_as_read_delegates_to_backend(self, mock_emails_file):
        mock_backend = MagicMock()
        mock_backend.connect.return_value = True
        mock_backend.mark_as_read.return_value = True

        svc = EmailService(mock_data_file=mock_emails_file, backend=mock_backend)
        svc.connect()
        assert svc.mark_as_read("msg-1") is True
        mock_backend.mark_as_read.assert_called_once_with("msg-1")

    def test_send_with_backend(self, mock_emails_file):
        mock_backend = MagicMock()
        mock_backend.connect.return_value = True
        mock_backend.send.return_value = SendResult(ok=True, message_id="sent-1")

        svc = EmailService(mock_data_file=mock_emails_file, backend=mock_backend)
        svc.connect()
        result = svc.send(
            to="recipient@y.com",
            subject="Test Send",
            body="Hello via backend",
        )
        assert result["ok"] is True
        assert result["message_id"] == "sent-1"
        mock_backend.send.assert_called_once()

    def test_send_stub_mode(self, mock_emails_file):
        """Stub mode send logs and returns success."""
        svc = EmailService(mock_data_file=mock_emails_file)
        svc.connect()
        result = svc.send(
            to="recipient@y.com",
            subject="Stub Send",
            body="Logged only",
        )
        assert result["ok"] is True
        assert result["error"] is None

    def test_send_with_list_of_recipients(self, mock_emails_file):
        mock_backend = MagicMock()
        mock_backend.connect.return_value = True
        mock_backend.send.return_value = SendResult(ok=True, message_id="sent-2")

        svc = EmailService(mock_data_file=mock_emails_file, backend=mock_backend)
        svc.connect()
        result = svc.send(
            to=["a@b.com", "c@d.com"],
            subject="Multi",
            body="Multi-recipient",
        )
        assert result["ok"] is True
        call_kwargs = mock_backend.send.call_args[1]
        assert call_kwargs["to_addrs"] == ["a@b.com", "c@d.com"]

    def test_send_failure(self, mock_emails_file):
        mock_backend = MagicMock()
        mock_backend.connect.return_value = True
        mock_backend.send.return_value = SendResult(ok=False, error="relay denied")

        svc = EmailService(mock_data_file=mock_emails_file, backend=mock_backend)
        svc.connect()
        result = svc.send(to="x@y.com", subject="Fail", body="Oops")
        assert result["ok"] is False
        assert "relay denied" in result["error"]


class TestEmailServiceStubBackend:
    """Test EmailService with the StubEmailBackend explicitly set."""

    def test_stub_backend_integration(self, mock_emails_file):
        backend = StubEmailBackend(fixture_path=mock_emails_file)
        svc = EmailService(backend=backend)
        assert svc.connect() is True
        unread = svc.fetch_unread()
        assert len(unread) == 1

    def test_stub_send_through_service(self, mock_emails_file):
        backend = StubEmailBackend(fixture_path=mock_emails_file)
        svc = EmailService(backend=backend)
        svc.connect()
        result = svc.send(to="x@y.com", subject="Hi", body="Hello")
        assert result["ok"] is True
        assert len(backend.sent_messages) == 1
