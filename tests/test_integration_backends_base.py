"""Tests for transport-layer ABC interfaces and stub implementations (US-205)."""

from __future__ import annotations

import abc
import inspect

import pytest

from rex.integrations.email.backends.base import EmailBackend
from rex.integrations.email.backends.stub import StubEmailBackend
from rex.integrations.calendar.backends.base import CalendarBackend
from rex.integrations.calendar.backends.stub import StubCalendarBackend
from rex.integrations.messaging.backends.base import SMSBackend
from rex.integrations.messaging.backends.stub import StubSMSBackend


# ---------------------------------------------------------------------------
# EmailBackend ABC
# ---------------------------------------------------------------------------


def test_email_backend_is_abstract():
    assert inspect.isabstract(EmailBackend)


def test_email_backend_abstract_methods():
    abstract = {
        name
        for name, val in inspect.getmembers(EmailBackend)
        if getattr(val, "__isabstractmethod__", False)
    }
    assert "fetch_unread" in abstract
    assert "send" in abstract


def test_email_backend_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        EmailBackend()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# StubEmailBackend
# ---------------------------------------------------------------------------


def test_stub_email_backend_implements_interface():
    assert issubclass(StubEmailBackend, EmailBackend)


def test_stub_email_fetch_unread_returns_list_of_dicts():
    backend = StubEmailBackend()
    results = backend.fetch_unread(limit=5)
    assert isinstance(results, list)
    for item in results:
        assert isinstance(item, dict)
        assert "id" in item
        assert "from" in item
        assert "subject" in item


def test_stub_email_fetch_unread_respects_limit():
    backend = StubEmailBackend()
    assert len(backend.fetch_unread(limit=1)) == 1
    assert len(backend.fetch_unread(limit=0)) == 0


def test_stub_email_send_records_message():
    backend = StubEmailBackend()
    backend.send(to="bob@example.com", subject="Hello", body="World")
    assert len(backend.sent_messages) == 1
    sent = backend.sent_messages[0]
    assert sent["to"] == "bob@example.com"
    assert sent["subject"] == "Hello"
    assert sent["body"] == "World"


def test_stub_email_send_returns_none():
    backend = StubEmailBackend()
    result = backend.send(to="x@example.com", subject="S", body="B")
    assert result is None


# ---------------------------------------------------------------------------
# CalendarBackend ABC
# ---------------------------------------------------------------------------


def test_calendar_backend_is_abstract():
    assert inspect.isabstract(CalendarBackend)


def test_calendar_backend_abstract_methods():
    abstract = {
        name
        for name, val in inspect.getmembers(CalendarBackend)
        if getattr(val, "__isabstractmethod__", False)
    }
    assert "get_upcoming" in abstract
    assert "create_event" in abstract


def test_calendar_backend_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        CalendarBackend()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# StubCalendarBackend
# ---------------------------------------------------------------------------


def test_stub_calendar_backend_implements_interface():
    assert issubclass(StubCalendarBackend, CalendarBackend)


def test_stub_calendar_get_upcoming_returns_list_of_dicts():
    backend = StubCalendarBackend()
    results = backend.get_upcoming(days=7)
    assert isinstance(results, list)
    for item in results:
        assert isinstance(item, dict)
        assert "id" in item
        assert "title" in item
        assert "start" in item
        assert "end" in item


def test_stub_calendar_create_event_returns_dict_with_id():
    backend = StubCalendarBackend()
    event = backend.create_event(
        title="Planning meeting",
        start="2026-04-01T10:00:00+00:00",
        end="2026-04-01T11:00:00+00:00",
    )
    assert isinstance(event, dict)
    assert "id" in event
    assert event["title"] == "Planning meeting"
    assert event["start"] == "2026-04-01T10:00:00+00:00"
    assert event["end"] == "2026-04-01T11:00:00+00:00"


def test_stub_calendar_create_event_appears_in_upcoming():
    backend = StubCalendarBackend(events=[])
    backend.create_event("Solo event", "2026-04-01T09:00:00+00:00", "2026-04-01T10:00:00+00:00")
    upcoming = backend.get_upcoming()
    assert len(upcoming) == 1
    assert upcoming[0]["title"] == "Solo event"


def test_stub_calendar_create_event_idempotent_count():
    backend = StubCalendarBackend(events=[])
    backend.create_event("A", "2026-04-01T09:00:00+00:00", "2026-04-01T10:00:00+00:00")
    backend.create_event("B", "2026-04-01T11:00:00+00:00", "2026-04-01T12:00:00+00:00")
    assert len(backend.created_events) == 2


# ---------------------------------------------------------------------------
# SMSBackend ABC
# ---------------------------------------------------------------------------


def test_sms_backend_is_abstract():
    assert inspect.isabstract(SMSBackend)


def test_sms_backend_abstract_methods():
    abstract = {
        name
        for name, val in inspect.getmembers(SMSBackend)
        if getattr(val, "__isabstractmethod__", False)
    }
    assert "send" in abstract
    assert "receive" in abstract


def test_sms_backend_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        SMSBackend()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# StubSMSBackend
# ---------------------------------------------------------------------------


def test_stub_sms_backend_implements_interface():
    assert issubclass(StubSMSBackend, SMSBackend)


def test_stub_sms_receive_returns_list_of_dicts():
    backend = StubSMSBackend()
    results = backend.receive()
    assert isinstance(results, list)
    for item in results:
        assert isinstance(item, dict)
        assert "id" in item
        assert "from" in item
        assert "body" in item
        assert "received_at" in item


def test_stub_sms_send_records_message():
    backend = StubSMSBackend()
    backend.send(to="+15550001234", body="Hello there")
    assert len(backend.sent_messages) == 1
    sent = backend.sent_messages[0]
    assert sent["to"] == "+15550001234"
    assert sent["body"] == "Hello there"


def test_stub_sms_send_returns_none():
    backend = StubSMSBackend()
    result = backend.send(to="+15550001234", body="test")
    assert result is None
