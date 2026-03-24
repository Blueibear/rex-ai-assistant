"""Unit tests for SMSMessage and SMSThread models in rex.integrations.models."""

from __future__ import annotations

from datetime import datetime, timezone

from rex.integrations.models import SMSMessage, SMSThread

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_message(
    id: str = "sms-001",
    thread_id: str = "thread-001",
    direction: str = "inbound",
    body: str = "Hello!",
    from_number: str = "+14155550100",
    to_number: str = "+14155550200",
) -> SMSMessage:
    return SMSMessage(
        id=id,
        thread_id=thread_id,
        direction=direction,  # type: ignore[arg-type]
        body=body,
        from_number=from_number,
        to_number=to_number,
    )


# ---------------------------------------------------------------------------
# SMSMessage tests
# ---------------------------------------------------------------------------


class TestSMSMessage:
    def test_create_minimal(self) -> None:
        msg = _make_message()
        assert msg.id == "sms-001"
        assert msg.direction == "inbound"
        assert msg.body == "Hello!"
        assert msg.status == "stub"

    def test_default_sent_at_is_utc(self) -> None:
        msg = _make_message()
        assert msg.sent_at.tzinfo is not None

    def test_direction_inbound(self) -> None:
        msg = _make_message(direction="inbound")
        assert msg.direction == "inbound"

    def test_direction_outbound(self) -> None:
        msg = _make_message(direction="outbound")
        assert msg.direction == "outbound"

    def test_status_values(self) -> None:
        for status in ("sent", "delivered", "failed", "stub"):
            msg = SMSMessage(
                id="x",
                thread_id="t",
                direction="outbound",
                body="hi",
                from_number="+1",
                to_number="+2",
                status=status,  # type: ignore[arg-type]
            )
            assert msg.status == status

    def test_model_dump_round_trip(self) -> None:
        msg = _make_message()
        restored = SMSMessage(**msg.model_dump())
        assert restored == msg

    def test_explicit_sent_at(self) -> None:
        dt = datetime(2025, 1, 15, 10, 30, tzinfo=timezone.utc)
        msg = SMSMessage(
            id="x",
            thread_id="t",
            direction="inbound",
            body="hi",
            from_number="+1",
            to_number="+2",
            sent_at=dt,
        )
        assert msg.sent_at == dt


# ---------------------------------------------------------------------------
# SMSThread tests
# ---------------------------------------------------------------------------


class TestSMSThread:
    def test_create_empty_thread(self) -> None:
        thread = SMSThread(
            id="thread-001",
            contact_name="Alice",
            contact_number="+14155550100",
        )
        assert thread.id == "thread-001"
        assert thread.contact_name == "Alice"
        assert thread.messages == []
        assert thread.unread_count == 0

    def test_thread_with_messages(self) -> None:
        msg = _make_message()
        thread = SMSThread(
            id="thread-001",
            contact_name="Alice",
            contact_number="+14155550100",
            messages=[msg],
            unread_count=1,
        )
        assert len(thread.messages) == 1
        assert thread.unread_count == 1

    def test_model_dump_round_trip(self) -> None:
        msg = _make_message()
        thread = SMSThread(
            id="thread-001",
            contact_name="Alice",
            contact_number="+14155550100",
            messages=[msg],
            unread_count=1,
        )
        restored = SMSThread(**thread.model_dump())
        assert restored == thread

    def test_default_last_message_at_utc(self) -> None:
        thread = SMSThread(
            id="t",
            contact_name="Bob",
            contact_number="+1",
        )
        assert thread.last_message_at.tzinfo is not None
