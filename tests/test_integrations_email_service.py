"""Unit tests for rex.integrations.email_service — stub mode."""

from __future__ import annotations

from rex.integrations.email_service import EmailService
from rex.integrations.models import EmailMessage


class TestEmailServiceStub:
    """Tests for EmailService running in stub mode (email_provider='none')."""

    def setup_method(self) -> None:
        self.service = EmailService(email_provider="none")

    def test_list_inbox_returns_list_of_email_messages(self) -> None:
        messages = self.service.list_inbox()
        assert isinstance(messages, list)
        assert len(messages) > 0
        assert all(isinstance(m, EmailMessage) for m in messages)

    def test_list_inbox_respects_limit(self) -> None:
        messages = self.service.list_inbox(limit=2)
        assert len(messages) <= 2

    def test_list_inbox_default_limit(self) -> None:
        messages = self.service.list_inbox()
        assert len(messages) <= 20

    def test_get_thread_returns_messages_with_matching_thread_id(self) -> None:
        # Grab any thread_id from the stub inbox
        inbox = self.service.list_inbox()
        assert inbox, "Stub inbox must contain at least one message"
        target_thread = inbox[0].thread_id
        thread_messages = self.service.get_thread(target_thread)
        assert isinstance(thread_messages, list)
        assert all(m.thread_id == target_thread for m in thread_messages)

    def test_get_thread_unknown_returns_empty_list(self) -> None:
        result = self.service.get_thread("nonexistent-thread")
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
        # Stub mode is a no-op — should not raise
        self.service.archive("stub-001")

    def test_mark_read_does_not_raise(self) -> None:
        self.service.mark_read("stub-001")

    def test_all_stub_messages_have_valid_priority(self) -> None:
        valid_priorities = {"low", "medium", "high", "critical"}
        messages = self.service.list_inbox()
        for msg in messages:
            assert msg.priority in valid_priorities, (
                f"Message {msg.id} has invalid priority '{msg.priority}'"
            )

    def test_email_message_model_dump_round_trip(self) -> None:
        messages = self.service.list_inbox(limit=1)
        assert messages
        dumped = messages[0].model_dump()
        restored = EmailMessage(**dumped)
        assert restored == messages[0]
