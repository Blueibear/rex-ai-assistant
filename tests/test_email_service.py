"""Tests for email service module."""

from datetime import datetime
from pathlib import Path

import pytest

from rex.email_service import EmailService, EmailSummary


@pytest.fixture
def temp_mock_emails(tmp_path):
    """Create a temporary mock emails file."""
    import json

    mock_emails = [
        {
            "id": "email-1",
            "from_addr": "boss@company.com",
            "subject": "URGENT: Project deadline",
            "snippet": "The project is due tomorrow...",
            "received_at": "2026-01-28T10:00:00",
            "labels": ["unread", "important"],
            "importance_score": 0.9
        },
        {
            "id": "email-2",
            "from_addr": "newsletter@spam.com",
            "subject": "50% OFF SALE TODAY!",
            "snippet": "Click here to unsubscribe...",
            "received_at": "2026-01-28T09:00:00",
            "labels": ["unread"],
            "importance_score": 0.1
        },
        {
            "id": "email-3",
            "from_addr": "friend@example.com",
            "subject": "Lunch next week?",
            "snippet": "Hey, want to grab lunch?",
            "received_at": "2026-01-27T15:00:00",
            "labels": [],
            "importance_score": 0.6
        }
    ]

    mock_file = tmp_path / "mock_emails.json"
    with open(mock_file, 'w') as f:
        json.dump(mock_emails, f)

    return mock_file


@pytest.fixture
def email_service(temp_mock_emails):
    """Create a test email service instance."""
    return EmailService(mock_data_file=temp_mock_emails)


def test_email_summary_creation():
    """Test creating an EmailSummary."""
    email = EmailSummary(
        id="test-1",
        from_addr="test@example.com",
        subject="Test Subject",
        snippet="Test snippet",
        received_at=datetime.now()
    )

    assert email.id == "test-1"
    assert email.from_addr == "test@example.com"
    assert email.subject == "Test Subject"
    assert email.snippet == "Test snippet"
    assert email.labels == []
    assert email.importance_score == 0.5
    assert email.category is None


def test_email_service_initialization(temp_mock_emails):
    """Test email service initializes correctly."""
    service = EmailService(mock_data_file=temp_mock_emails)
    assert service.mock_data_file == temp_mock_emails
    assert service.connected is False


def test_connect(email_service):
    """Test connecting to email service."""
    result = email_service.connect()
    assert result is True
    assert email_service.connected is True


def test_connect_loads_mock_data(email_service):
    """Test that connect loads mock data."""
    email_service.connect()
    assert len(email_service._mock_emails) == 3


def test_fetch_unread(email_service):
    """Test fetching unread emails."""
    email_service.connect()
    unread = email_service.fetch_unread()

    assert len(unread) == 2
    assert all("unread" in email.labels for email in unread)


def test_fetch_unread_with_limit(email_service):
    """Test fetching unread emails with limit."""
    email_service.connect()
    unread = email_service.fetch_unread(limit=1)

    assert len(unread) == 1


def test_fetch_unread_not_connected():
    """Test fetching unread when not connected."""
    service = EmailService(mock_data_file=Path("/nonexistent"))
    unread = service.fetch_unread()

    assert unread == []


def test_mark_as_read(email_service):
    """Test marking an email as read."""
    email_service.connect()

    # Get an unread email
    unread = email_service.fetch_unread()
    email_id = unread[0].id

    # Mark as read
    result = email_service.mark_as_read(email_id)
    assert result is True

    # Verify it's no longer unread
    unread_after = email_service.fetch_unread()
    assert len(unread_after) == len(unread) - 1
    assert email_id not in [e.id for e in unread_after]


def test_mark_as_read_nonexistent(email_service):
    """Test marking a non-existent email as read."""
    email_service.connect()
    result = email_service.mark_as_read("nonexistent-id")
    assert result is False


def test_mark_as_read_not_connected():
    """Test marking as read when not connected."""
    service = EmailService(mock_data_file=Path("/nonexistent"))
    result = service.mark_as_read("email-1")
    assert result is False


def test_categorize_promo(email_service):
    """Test categorizing promotional emails."""
    email = EmailSummary(
        id="test",
        from_addr="sales@shop.com",
        subject="50% DISCOUNT TODAY!",
        snippet="Limited time offer! Buy now!",
        received_at=datetime.now()
    )

    category = email_service.categorize(email)
    assert category == "promo"


def test_categorize_social(email_service):
    """Test categorizing social emails."""
    email = EmailSummary(
        id="test",
        from_addr="notifications@facebook.com",
        subject="John liked your post",
        snippet="John and 5 others liked your post",
        received_at=datetime.now()
    )

    category = email_service.categorize(email)
    assert category == "social"


def test_categorize_newsletter(email_service):
    """Test categorizing newsletters."""
    email = EmailSummary(
        id="test",
        from_addr="newsletter@techblog.com",
        subject="Weekly Tech Digest",
        snippet="This week's top stories... Click here to unsubscribe",
        received_at=datetime.now()
    )

    category = email_service.categorize(email)
    assert category == "newsletter"


def test_categorize_important_by_keywords(email_service):
    """Test categorizing important emails by keywords."""
    email = EmailSummary(
        id="test",
        from_addr="boss@company.com",
        subject="URGENT: Action required",
        snippet="Please complete this immediately",
        received_at=datetime.now()
    )

    category = email_service.categorize(email)
    assert category == "important"


def test_categorize_important_by_score(email_service):
    """Test categorizing important emails by score."""
    email = EmailSummary(
        id="test",
        from_addr="client@customer.com",
        subject="Project update",
        snippet="Here are the latest updates",
        received_at=datetime.now(),
        importance_score=0.9
    )

    category = email_service.categorize(email)
    assert category == "important"


def test_categorize_general(email_service):
    """Test categorizing general emails."""
    email = EmailSummary(
        id="test",
        from_addr="friend@example.com",
        subject="How are you?",
        snippet="Just checking in",
        received_at=datetime.now()
    )

    category = email_service.categorize(email)
    assert category == "general"


def test_summarize(email_service):
    """Test summarizing an email."""
    email_service.connect()
    summary = email_service.summarize("email-1")

    assert "boss@company.com" in summary
    assert "URGENT: Project deadline" in summary
    assert "project is due tomorrow" in summary


def test_summarize_nonexistent(email_service):
    """Test summarizing a non-existent email."""
    email_service.connect()
    summary = email_service.summarize("nonexistent")

    assert "not found" in summary.lower()


def test_summarize_not_connected():
    """Test summarizing when not connected."""
    service = EmailService(mock_data_file=Path("/nonexistent"))
    summary = service.summarize("email-1")

    assert "not connected" in summary.lower()


def test_get_all_emails(email_service):
    """Test getting all emails."""
    email_service.connect()
    all_emails = email_service.get_all_emails()

    assert len(all_emails) == 3
    assert all(isinstance(email, EmailSummary) for email in all_emails)


def test_load_mock_data_file_not_found():
    """Test loading mock data when file doesn't exist."""
    service = EmailService(mock_data_file=Path("/nonexistent"))
    service._load_mock_data()

    assert service._mock_emails == []


def test_email_summary_with_all_fields():
    """Test EmailSummary with all fields."""
    email = EmailSummary(
        id="test-1",
        from_addr="test@example.com",
        subject="Test",
        snippet="Test snippet",
        received_at=datetime.now(),
        labels=["unread", "important"],
        importance_score=0.8,
        category="important"
    )

    assert email.labels == ["unread", "important"]
    assert email.importance_score == 0.8
    assert email.category == "important"


def test_email_summary_serialization():
    """Test EmailSummary JSON serialization."""
    now = datetime.now()
    email = EmailSummary(
        id="test-1",
        from_addr="test@example.com",
        subject="Test",
        snippet="Test snippet",
        received_at=now
    )

    data = email.model_dump()
    assert data['id'] == "test-1"
    assert data['from_addr'] == "test@example.com"
    assert data['received_at'] == now
