"""Tests for US-078: Email inbox stub and mock data.

Acceptance criteria:
- EmailInboxStub class exists and returns a list of mock email objects
- mock emails cover at least three categories: urgent, action_required, and fyi
- stub implements the same interface as the real email backend (US-044)
- tests can instantiate and query the stub without any live credentials or network calls
- Typecheck passes
"""

from __future__ import annotations

import pytest

from rex.email_backends.base import EmailBackend, EmailEnvelope, SendResult
from rex.email_backends.inbox_stub import EmailInboxStub

# ---------------------------------------------------------------------------
# Instantiation — no credentials or network calls required
# ---------------------------------------------------------------------------


def test_instantiation_requires_no_args() -> None:
    stub = EmailInboxStub()
    assert stub is not None


def test_instantiation_does_not_raise() -> None:
    # Must not raise even when no config files or env vars are present
    stub = EmailInboxStub()
    assert isinstance(stub, EmailInboxStub)


# ---------------------------------------------------------------------------
# Interface compliance — EmailBackend ABC
# ---------------------------------------------------------------------------


def test_is_subclass_of_email_backend() -> None:
    assert issubclass(EmailInboxStub, EmailBackend)


def test_implements_connect() -> None:
    stub = EmailInboxStub()
    result = stub.connect()
    assert result is True


def test_implements_fetch_unread() -> None:
    stub = EmailInboxStub()
    stub.connect()
    emails = stub.fetch_unread()
    assert isinstance(emails, list)


def test_implements_list_mailboxes() -> None:
    stub = EmailInboxStub()
    mailboxes = stub.list_mailboxes()
    assert isinstance(mailboxes, list)
    assert len(mailboxes) >= 1


def test_implements_mark_as_read() -> None:
    stub = EmailInboxStub()
    stub.connect()
    unread = stub.fetch_unread(limit=1)
    if unread:
        result = stub.mark_as_read(unread[0].message_id)
        assert isinstance(result, bool)


def test_implements_send() -> None:
    stub = EmailInboxStub()
    result = stub.send(
        from_addr="test@example.com",
        to_addrs=["dest@example.com"],
        subject="Test",
        body="Hello",
    )
    assert isinstance(result, SendResult)
    assert result.ok is True


def test_implements_disconnect() -> None:
    stub = EmailInboxStub()
    stub.connect()
    assert stub.is_connected is True
    stub.disconnect()
    assert stub.is_connected is False


# ---------------------------------------------------------------------------
# Mock data — EmailEnvelope objects returned
# ---------------------------------------------------------------------------


def test_all_emails_returns_email_envelopes() -> None:
    stub = EmailInboxStub()
    emails = stub.all_emails
    assert len(emails) > 0
    for email in emails:
        assert isinstance(email, EmailEnvelope)


def test_fetch_unread_returns_email_envelopes() -> None:
    stub = EmailInboxStub()
    stub.connect()
    emails = stub.fetch_unread()
    assert isinstance(emails, list)
    for email in emails:
        assert isinstance(email, EmailEnvelope)


def test_envelope_fields_populated() -> None:
    stub = EmailInboxStub()
    for email in stub.all_emails:
        assert email.message_id, "message_id must be non-empty"
        assert email.from_addr, "from_addr must be non-empty"
        assert email.subject, "subject must be non-empty"
        assert email.received_at is not None


# ---------------------------------------------------------------------------
# Category coverage — urgent, action_required, fyi
# ---------------------------------------------------------------------------


def test_urgent_category_present() -> None:
    stub = EmailInboxStub()
    urgent = stub.fetch_by_category("urgent")
    assert len(urgent) >= 1, "At least one mock email must be in the 'urgent' category"


def test_action_required_category_present() -> None:
    stub = EmailInboxStub()
    action = stub.fetch_by_category("action_required")
    assert len(action) >= 1, "At least one mock email must be in 'action_required'"


def test_fyi_category_present() -> None:
    stub = EmailInboxStub()
    fyi = stub.fetch_by_category("fyi")
    assert len(fyi) >= 1, "At least one mock email must be in the 'fyi' category"


def test_three_categories_covered() -> None:
    stub = EmailInboxStub()
    all_labels: set[str] = set()
    for email in stub.all_emails:
        all_labels.update(email.labels)
    required = {"urgent", "action_required", "fyi"}
    missing = required - all_labels
    assert not missing, f"Missing categories in mock data: {missing}"


# ---------------------------------------------------------------------------
# No credentials / no network — fetch works without connect()
# ---------------------------------------------------------------------------


def test_fetch_unread_auto_connects_if_not_connected() -> None:
    stub = EmailInboxStub()
    # Do NOT call connect() — stub should handle it internally
    emails = stub.fetch_unread()
    assert isinstance(emails, list)


def test_no_network_call_on_instantiation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure instantiation raises no import-level or constructor-level network error."""
    # Patch socket to refuse connections — any network call would raise
    import socket

    original_connect = socket.socket.connect

    def no_network(*args: object, **kwargs: object) -> None:
        raise OSError("Network access forbidden in test")

    monkeypatch.setattr(socket.socket, "connect", no_network)

    # Instantiation and basic operations must succeed without touching the network
    stub = EmailInboxStub()
    stub.connect()
    emails = stub.fetch_unread()
    assert isinstance(emails, list)

    monkeypatch.setattr(socket.socket, "connect", original_connect)


# ---------------------------------------------------------------------------
# Send records messages — no real delivery
# ---------------------------------------------------------------------------


def test_send_no_real_delivery() -> None:
    stub = EmailInboxStub()
    result = stub.send(
        from_addr="sender@example.com",
        to_addrs=["recv@example.com"],
        subject="Hello",
        body="World",
    )
    assert result.ok is True
    assert len(stub.sent_messages) == 1
    assert stub.sent_messages[0]["subject"] == "Hello"


def test_send_multiple_records_all() -> None:
    stub = EmailInboxStub()
    for i in range(3):
        stub.send(
            from_addr="a@example.com",
            to_addrs=["b@example.com"],
            subject=f"Msg {i}",
            body="body",
        )
    assert len(stub.sent_messages) == 3


# ---------------------------------------------------------------------------
# mark_as_read removes unread label
# ---------------------------------------------------------------------------


def test_mark_as_read_removes_unread_label() -> None:
    stub = EmailInboxStub()
    stub.connect()
    unread_before = stub.fetch_unread()
    assert len(unread_before) >= 1
    target_id = unread_before[0].message_id
    stub.mark_as_read(target_id)
    unread_after = stub.fetch_unread()
    ids_after = {e.message_id for e in unread_after}
    assert target_id not in ids_after


def test_mark_as_read_unknown_id_returns_false() -> None:
    stub = EmailInboxStub()
    result = stub.mark_as_read("nonexistent-id-xyz")
    assert result is False


# ---------------------------------------------------------------------------
# fetch_unread respects limit
# ---------------------------------------------------------------------------


def test_fetch_unread_respects_limit() -> None:
    stub = EmailInboxStub()
    stub.connect()
    emails = stub.fetch_unread(limit=1)
    assert len(emails) <= 1


def test_fetch_unread_limit_zero_returns_empty() -> None:
    stub = EmailInboxStub()
    stub.connect()
    emails = stub.fetch_unread(limit=0)
    assert emails == []
