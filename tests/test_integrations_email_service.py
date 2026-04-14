"""Unit tests for rex.integrations.email_service.

When email_provider='none' (not configured), list_inbox() returns an empty
list so the GUI can show an empty-state prompt instead of fake data.
"""

from __future__ import annotations

from rex.integrations.email_service import EmailService
from rex.integrations.models import EmailMessage


class TestEmailServiceStub:
    """Tests for EmailService running in 'none' (unconfigured) mode."""

    def setup_method(self) -> None:
        self.service = EmailService(email_provider="none")

    def test_list_inbox_returns_list(self) -> None:
        messages = self.service.list_inbox()
        assert isinstance(messages, list)

    def test_list_inbox_empty_when_not_configured(self) -> None:
        """US-315: provider='none' must return [] not stub data."""
        messages = self.service.list_inbox()
        assert messages == []

    def test_list_inbox_respects_limit(self) -> None:
        messages = self.service.list_inbox(limit=2)
        assert len(messages) <= 2

    def test_list_inbox_default_limit(self) -> None:
        messages = self.service.list_inbox()
        assert len(messages) <= 20

    def test_get_thread_unknown_returns_empty_list(self) -> None:
        result = self.service.get_thread("nonexistent-thread")
        assert result == []

    def test_get_thread_returns_empty_when_not_configured(self) -> None:
        """US-315: get_thread with any id returns [] when not configured."""
        result = self.service.get_thread("any-thread-id")
        assert result == []

    def test_send_draft_returns_email_message(self) -> None:
        msg = self.service.send_draft(
            to="colleague@example.com",
            subject="Hello",
            body="Just checking in.",
        )
        assert isinstance(msg, EmailMessage)
        assert msg.recipients == ["colleague@example.com"]
        assert msg.subject == "Hello"
        assert msg.body_text == "Just checking in."
        assert "SENT" in msg.labels

    def test_archive_does_not_raise(self) -> None:
        self.service.archive("msg-001")

    def test_mark_read_does_not_raise(self) -> None:
        self.service.mark_read("msg-001")

    def test_email_message_model_dump_round_trip(self) -> None:
        msg = self.service.send_draft(to="test@example.com", subject="Round trip", body="Test body")
        dumped = msg.model_dump()
        restored = EmailMessage(**dumped)
        assert restored == msg
