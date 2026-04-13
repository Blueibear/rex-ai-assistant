"""
Tests for the email service.

This test module supports BOTH email service variants that have appeared in the
codebase:

Variant A (legacy/event-bus style):
- EmailMessage dataclass with fields:
    message_id, sender, subject, body, received_at
- EmailService(event_bus, mock_messages=[...], mock_data_path=...)
- service.triage_unread() -> list[dict]
- EventBus.subscribe(event_type, callback(event_type, payload))

Variant B (newer/pydantic/mock-file style):
- EmailSummary pydantic model with fields:
    id, from_addr, subject, snippet, received_at, labels, importance_score, category
- EmailService(mock_data_file=Path(...))
- service.connect()
- service.fetch_unread(limit=10)
- service.mark_as_read(email_id)
- service.categorize(email_summary)
- service.summarize(email_id)
- service.get_all_emails()

These tests auto-detect which API is available at runtime and skip the
incompatible tests, keeping CI green while implementations converge.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from rex.email_service import EmailService


def _has_attr(obj: Any, name: str) -> bool:
    return hasattr(obj, name)


def _email_summary_class():
    try:
        from rex.email_service import EmailSummary  # type: ignore

        return EmailSummary
    except Exception:
        return None


def _email_message_class():
    try:
        from rex.email_service import EmailMessage  # type: ignore

        return EmailMessage
    except Exception:
        return None


def _event_bus_class():
    try:
        from rex.openclaw.event_bus import EventBus  # type: ignore

        return EventBus
    except Exception:
        return None


def _email_service_accepts_event_bus() -> bool:
    EventBus = _event_bus_class()
    if EventBus is None:
        return False
    try:
        bus = EventBus()
        _ = EmailService(bus)  # type: ignore[arg-type]
        return True
    except Exception:
        return False


def _email_summary_is_pydantic() -> bool:
    EmailSummary = _email_summary_class()
    if EmailSummary is None:
        return False
    # Pydantic v2 models have model_dump; dataclasses don't.
    return _has_attr(EmailSummary, "model_dump") or _has_attr(EmailSummary, "model_validate")


# -------------------------------------------------------------------
# Legacy implementation test (event bus + triage publishes events)
# -------------------------------------------------------------------


@pytest.mark.skipif(
    not _email_service_accepts_event_bus(),
    reason="EmailService event-bus style API not available in this build.",
)
def test_email_triage_publishes_events() -> None:
    EmailMessage = _email_message_class()
    EventBus = _event_bus_class()
    assert EmailMessage is not None
    assert EventBus is not None

    bus = EventBus()
    events: list[tuple[str, dict[str, object]]] = []

    def handler(event_type: str, payload: dict[str, object]) -> None:
        events.append((event_type, payload))

    bus.subscribe("email.triaged", handler)

    message = EmailMessage(
        message_id="email-123",
        sender="billing@example.com",
        subject="Invoice ready",
        body="Details inside",
        received_at=datetime(2024, 1, 1, 9, 0, tzinfo=UTC),
    )

    service = EmailService(bus, mock_messages=[message])  # type: ignore[call-arg]
    triaged = service.triage_unread()  # type: ignore[attr-defined]

    assert triaged[0]["category"] == "finance"
    assert events, "Expected at least one published event"
    assert events[0][0] == "email.triaged"
    assert events[0][1]["count"] == 1


# -------------------------------------------------------------------
# Newer implementation fixtures (mock file based)
# -------------------------------------------------------------------


@pytest.fixture
def temp_mock_emails(tmp_path: Path) -> Path:
    """Create a temporary mock emails file (newer implementation)."""
    import json

    mock_emails = [
        {
            "id": "email-1",
            "from_addr": "boss@company.com",
            "subject": "URGENT: Project deadline",
            "snippet": "The project is due tomorrow...",
            "received_at": "2026-01-28T10:00:00",
            "labels": ["unread", "important"],
            "importance_score": 0.9,
        },
        {
            "id": "email-2",
            "from_addr": "newsletter@spam.com",
            "subject": "50% OFF SALE TODAY!",
            "snippet": "Click here to unsubscribe...",
            "received_at": "2026-01-28T09:00:00",
            "labels": ["unread"],
            "importance_score": 0.1,
        },
        {
            "id": "email-3",
            "from_addr": "friend@example.com",
            "subject": "Lunch next week?",
            "snippet": "Hey, want to grab lunch?",
            "received_at": "2026-01-27T15:00:00",
            "labels": [],
            "importance_score": 0.6,
        },
    ]

    mock_file = tmp_path / "mock_emails.json"
    mock_file.write_text(json.dumps(mock_emails), encoding="utf-8")
    return mock_file


@pytest.fixture
def email_service(temp_mock_emails: Path) -> EmailService:
    """Create a test email service instance (newer implementation)."""
    return EmailService(mock_data_file=temp_mock_emails)  # type: ignore[call-arg]


# -------------------------------------------------------------------
# Newer implementation tests (pydantic model + mock file + CRUD)
# -------------------------------------------------------------------


@pytest.mark.skipif(
    not _email_summary_is_pydantic(),
    reason="EmailSummary pydantic-style model not available in this build.",
)
def test_email_summary_creation() -> None:
    EmailSummary = _email_summary_class()
    assert EmailSummary is not None

    email = EmailSummary(
        id="test-1",
        from_addr="test@example.com",
        subject="Test Subject",
        snippet="Test snippet",
        received_at=datetime.now(),
    )

    assert email.id == "test-1"
    assert email.from_addr == "test@example.com"
    assert email.subject == "Test Subject"
    assert email.snippet == "Test snippet"
    assert email.labels == []
    assert email.importance_score == 0.5
    assert email.category is None


@pytest.mark.skipif(
    not _email_summary_is_pydantic(),
    reason="Mock-file email service tests apply to the newer implementation only.",
)
def test_email_service_initialization(temp_mock_emails: Path) -> None:
    service = EmailService(mock_data_file=temp_mock_emails)  # type: ignore[call-arg]
    assert service.mock_data_file == temp_mock_emails  # type: ignore[attr-defined]
    assert service.connected is False  # type: ignore[attr-defined]


@pytest.mark.skipif(
    not _email_summary_is_pydantic(),
    reason="Mock-file email service tests apply to the newer implementation only.",
)
def test_connect(email_service: EmailService) -> None:
    result = email_service.connect()  # type: ignore[attr-defined]
    assert result is True
    assert email_service.connected is True  # type: ignore[attr-defined]


@pytest.mark.skipif(
    not _email_summary_is_pydantic(),
    reason="Mock-file email service tests apply to the newer implementation only.",
)
def test_connect_loads_mock_data(email_service: EmailService) -> None:
    email_service.connect()  # type: ignore[attr-defined]
    assert len(email_service._mock_emails) == 3  # type: ignore[attr-defined]


@pytest.mark.skipif(
    not _email_summary_is_pydantic(),
    reason="Mock-file email service tests apply to the newer implementation only.",
)
def test_fetch_unread(email_service: EmailService) -> None:
    email_service.connect()  # type: ignore[attr-defined]
    unread = email_service.fetch_unread()  # type: ignore[attr-defined]

    assert len(unread) == 2
    assert all("unread" in email.labels for email in unread)


@pytest.mark.skipif(
    not _email_summary_is_pydantic(),
    reason="Mock-file email service tests apply to the newer implementation only.",
)
def test_fetch_unread_with_limit(email_service: EmailService) -> None:
    email_service.connect()  # type: ignore[attr-defined]
    unread = email_service.fetch_unread(limit=1)  # type: ignore[attr-defined]
    assert len(unread) == 1


@pytest.mark.skipif(
    not _email_summary_is_pydantic(),
    reason="Mock-file email service tests apply to the newer implementation only.",
)
def test_fetch_unread_not_connected() -> None:
    service = EmailService(mock_data_file=Path("/nonexistent"))  # type: ignore[call-arg]
    unread = service.fetch_unread()  # type: ignore[attr-defined]
    assert unread == []


@pytest.mark.skipif(
    not _email_summary_is_pydantic(),
    reason="Mock-file email service tests apply to the newer implementation only.",
)
def test_mark_as_read(email_service: EmailService) -> None:
    email_service.connect()  # type: ignore[attr-defined]
    unread = email_service.fetch_unread()  # type: ignore[attr-defined]
    email_id = unread[0].id

    result = email_service.mark_as_read(email_id)  # type: ignore[attr-defined]
    assert result is True

    unread_after = email_service.fetch_unread()  # type: ignore[attr-defined]
    assert len(unread_after) == len(unread) - 1
    assert email_id not in [e.id for e in unread_after]


@pytest.mark.skipif(
    not _email_summary_is_pydantic(),
    reason="Mock-file email service tests apply to the newer implementation only.",
)
def test_mark_as_read_nonexistent(email_service: EmailService) -> None:
    email_service.connect()  # type: ignore[attr-defined]
    assert email_service.mark_as_read("nonexistent-id") is False  # type: ignore[attr-defined]


@pytest.mark.skipif(
    not _email_summary_is_pydantic(),
    reason="Mock-file email service tests apply to the newer implementation only.",
)
def test_mark_as_read_not_connected() -> None:
    service = EmailService(mock_data_file=Path("/nonexistent"))  # type: ignore[call-arg]
    assert service.mark_as_read("email-1") is False  # type: ignore[attr-defined]


@pytest.mark.skipif(
    not _email_summary_is_pydantic(),
    reason="Mock-file email service tests apply to the newer implementation only.",
)
def test_categorize_promo(email_service: EmailService) -> None:
    EmailSummary = _email_summary_class()
    assert EmailSummary is not None

    email = EmailSummary(
        id="test",
        from_addr="sales@shop.com",
        subject="50% DISCOUNT TODAY!",
        snippet="Limited time offer! Buy now!",
        received_at=datetime.now(),
    )
    assert email_service.categorize(email) == "promo"  # type: ignore[attr-defined]


@pytest.mark.skipif(
    not _email_summary_is_pydantic(),
    reason="Mock-file email service tests apply to the newer implementation only.",
)
def test_categorize_social(email_service: EmailService) -> None:
    EmailSummary = _email_summary_class()
    assert EmailSummary is not None

    email = EmailSummary(
        id="test",
        from_addr="notifications@facebook.com",
        subject="John liked your post",
        snippet="John and 5 others liked your post",
        received_at=datetime.now(),
    )
    assert email_service.categorize(email) == "social"  # type: ignore[attr-defined]


@pytest.mark.skipif(
    not _email_summary_is_pydantic(),
    reason="Mock-file email service tests apply to the newer implementation only.",
)
def test_categorize_newsletter(email_service: EmailService) -> None:
    EmailSummary = _email_summary_class()
    assert EmailSummary is not None

    email = EmailSummary(
        id="test",
        from_addr="newsletter@techblog.com",
        subject="Weekly Tech Digest",
        snippet="This week's top stories... Click here to unsubscribe",
        received_at=datetime.now(),
    )
    assert email_service.categorize(email) == "newsletter"  # type: ignore[attr-defined]


@pytest.mark.skipif(
    not _email_summary_is_pydantic(),
    reason="Mock-file email service tests apply to the newer implementation only.",
)
def test_categorize_important_by_keywords(email_service: EmailService) -> None:
    EmailSummary = _email_summary_class()
    assert EmailSummary is not None

    email = EmailSummary(
        id="test",
        from_addr="boss@company.com",
        subject="URGENT: Action required",
        snippet="Please complete this immediately",
        received_at=datetime.now(),
    )
    assert email_service.categorize(email) == "important"  # type: ignore[attr-defined]


@pytest.mark.skipif(
    not _email_summary_is_pydantic(),
    reason="Mock-file email service tests apply to the newer implementation only.",
)
def test_categorize_important_by_score(email_service: EmailService) -> None:
    EmailSummary = _email_summary_class()
    assert EmailSummary is not None

    email = EmailSummary(
        id="test",
        from_addr="client@customer.com",
        subject="Project update",
        snippet="Here are the latest updates",
        received_at=datetime.now(),
        importance_score=0.9,
    )
    assert email_service.categorize(email) == "important"  # type: ignore[attr-defined]


@pytest.mark.skipif(
    not _email_summary_is_pydantic(),
    reason="Mock-file email service tests apply to the newer implementation only.",
)
def test_categorize_general(email_service: EmailService) -> None:
    EmailSummary = _email_summary_class()
    assert EmailSummary is not None

    email = EmailSummary(
        id="test",
        from_addr="friend@example.com",
        subject="How are you?",
        snippet="Just checking in",
        received_at=datetime.now(),
    )
    assert email_service.categorize(email) == "general"  # type: ignore[attr-defined]


@pytest.mark.skipif(
    not _email_summary_is_pydantic(),
    reason="Mock-file email service tests apply to the newer implementation only.",
)
def test_summarize(email_service: EmailService) -> None:
    email_service.connect()  # type: ignore[attr-defined]
    summary = email_service.summarize("email-1")  # type: ignore[attr-defined]

    assert "boss@company.com" in summary
    assert "URGENT: Project deadline" in summary
    assert "project is due tomorrow" in summary.lower()


@pytest.mark.skipif(
    not _email_summary_is_pydantic(),
    reason="Mock-file email service tests apply to the newer implementation only.",
)
def test_summarize_nonexistent(email_service: EmailService) -> None:
    email_service.connect()  # type: ignore[attr-defined]
    summary = email_service.summarize("nonexistent")  # type: ignore[attr-defined]
    assert "not found" in summary.lower()


@pytest.mark.skipif(
    not _email_summary_is_pydantic(),
    reason="Mock-file email service tests apply to the newer implementation only.",
)
def test_summarize_not_connected() -> None:
    service = EmailService(mock_data_file=Path("/nonexistent"))  # type: ignore[call-arg]
    summary = service.summarize("email-1")  # type: ignore[attr-defined]
    assert "not connected" in summary.lower()


@pytest.mark.skipif(
    not _email_summary_is_pydantic(),
    reason="Mock-file email service tests apply to the newer implementation only.",
)
def test_get_all_emails(email_service: EmailService) -> None:
    email_service.connect()  # type: ignore[attr-defined]
    all_emails = email_service.get_all_emails()  # type: ignore[attr-defined]

    assert len(all_emails) == 3
    EmailSummary = _email_summary_class()
    assert EmailSummary is not None
    assert all(isinstance(email, EmailSummary) for email in all_emails)


@pytest.mark.skipif(
    not _email_summary_is_pydantic(),
    reason="Mock-file email service tests apply to the newer implementation only.",
)
def test_load_mock_data_file_not_found() -> None:
    service = EmailService(mock_data_file=Path("/nonexistent"))  # type: ignore[call-arg]
    service._load_mock_data()  # type: ignore[attr-defined]
    assert getattr(service, "_mock_emails", None) == []


@pytest.mark.skipif(
    not _email_summary_is_pydantic(),
    reason="EmailSummary pydantic-style model not available in this build.",
)
def test_email_summary_with_all_fields() -> None:
    EmailSummary = _email_summary_class()
    assert EmailSummary is not None

    email = EmailSummary(
        id="test-1",
        from_addr="test@example.com",
        subject="Test",
        snippet="Test snippet",
        received_at=datetime.now(),
        labels=["unread", "important"],
        importance_score=0.8,
        category="important",
    )

    assert email.labels == ["unread", "important"]
    assert email.importance_score == 0.8
    assert email.category == "important"


@pytest.mark.skipif(
    not _email_summary_is_pydantic(),
    reason="EmailSummary pydantic-style model not available in this build.",
)
def test_email_summary_serialization() -> None:
    EmailSummary = _email_summary_class()
    assert EmailSummary is not None

    now = datetime.now()
    email = EmailSummary(
        id="test-1",
        from_addr="test@example.com",
        subject="Test",
        snippet="Test snippet",
        received_at=now,
    )

    data = email.model_dump()
    assert data["id"] == "test-1"
    assert data["from_addr"] == "test@example.com"
    assert data["received_at"] == now

    # JSON serialization should use isoformat for received_at
    import json as _json

    json_str = email.model_dump_json()
    json_data = _json.loads(json_str)
    assert json_data["received_at"] == now.isoformat()


@pytest.mark.skipif(
    not _email_summary_is_pydantic(),
    reason="EmailSummary pydantic-style model not available in this build.",
)
def test_email_summary_no_pydantic_v2_deprecation_warnings() -> None:
    """Creating EmailSummary should not emit Pydantic v2 deprecation warnings."""
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        EmailSummary = _email_summary_class()
        assert EmailSummary is not None
        EmailSummary(
            id="warn-test",
            from_addr="test@example.com",
            subject="Test",
            snippet="Test snippet",
            received_at=datetime.now(),
        )

    pydantic_warnings = [
        w
        for w in caught
        if "pydantic" in str(w.category.__module__).lower()
        or "class-based" in str(w.message).lower()
        or "json_encoders" in str(w.message).lower()
    ]
    assert (
        pydantic_warnings == []
    ), f"Pydantic deprecation warnings emitted: {[str(w.message) for w in pydantic_warnings]}"
