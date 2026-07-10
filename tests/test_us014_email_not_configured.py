"""Tests for US-014: Remove mock email data and connect real backend.

Acceptance criteria:
- No hardcoded fake email data remains in rex/email_service.py or rex/email_backends/
- If IMAP/SMTP credentials are absent, get_email_service() raises IntegrationNotConfiguredError
- If credentials are present, a real backend is returned
- Test covers both configured and not-configured paths
- Typecheck passes
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from rex.assistant_errors import IntegrationNotConfiguredError
from rex.email_backends.inbox_stub import EmailInboxStub
from rex.email_service import EmailService, _create_configured_email_service

# ---------------------------------------------------------------------------
# AC1: No hardcoded fake emails in inbox_stub
# ---------------------------------------------------------------------------


def test_inbox_stub_starts_empty_by_default() -> None:
    """EmailInboxStub has no built-in mock data when no emails injected."""
    stub = EmailInboxStub()
    assert stub.all_emails == []


def test_inbox_stub_accepts_injected_emails() -> None:
    """EmailInboxStub returns injected emails."""
    from datetime import UTC, datetime

    from rex.email_backends.base import EmailEnvelope

    env = EmailEnvelope(
        message_id="injected-001",
        from_addr="a@example.com",
        subject="Test",
        snippet="snippet",
        received_at=datetime(2026, 1, 1, tzinfo=UTC),
        to_addrs=["b@example.com"],
        labels=["unread"],
    )
    stub = EmailInboxStub(emails=[env])
    assert len(stub.all_emails) == 1
    assert stub.all_emails[0].message_id == "injected-001"


def test_email_service_has_no_hardcoded_mock_data() -> None:
    """EmailService with no backend loads no hardcoded emails."""
    service = EmailService(mock_data_file=None)
    # _mock_emails starts empty; only populated when _load_mock_data() is called
    assert service._mock_emails == []


# ---------------------------------------------------------------------------
# AC2: Not-configured path raises IntegrationNotConfiguredError
# ---------------------------------------------------------------------------


def test_get_email_service_raises_when_not_configured() -> None:
    """get_email_service() raises IntegrationNotConfiguredError when no accounts."""
    import rex.config_manager as config_manager
    import rex.email_service as svc_mod

    original = svc_mod._email_service
    try:
        svc_mod._email_service = None
        with patch.object(config_manager, "load_config", return_value={}):
            with pytest.raises(IntegrationNotConfiguredError):
                svc_mod.get_email_service()
    finally:
        svc_mod._email_service = original


def test_create_configured_raises_when_no_backend() -> None:
    """_create_configured_email_service() raises when no accounts configured."""
    import rex.config_manager as config_manager

    with patch.object(config_manager, "load_config", return_value={}):
        with pytest.raises(IntegrationNotConfiguredError):
            _create_configured_email_service()


def test_create_configured_error_message_mentions_email() -> None:
    """IntegrationNotConfiguredError message references email."""
    import rex.config_manager as config_manager

    with patch.object(config_manager, "load_config", return_value={}):
        with pytest.raises(IntegrationNotConfiguredError, match="Email"):
            _create_configured_email_service()


# ---------------------------------------------------------------------------
# AC3: Configured path returns a real backend
# ---------------------------------------------------------------------------


def test_get_email_service_returns_service_when_configured() -> None:
    """get_email_service() returns an owner-enforcing EmailService when configured."""
    import rex.config_manager as config_manager
    import rex.email_service as svc_mod

    raw_config = {
        "email": {
            "default_account_id": "personal",
            "accounts": [
                {
                    "id": "personal",
                    "address": "you@example.com",
                    "imap": {"host": "imap.example.com"},
                    "smtp": {"host": "smtp.example.com"},
                    "credential_ref": "email:personal",
                }
            ],
        }
    }
    original = svc_mod._email_service
    try:
        svc_mod._email_service = None
        with patch.object(config_manager, "load_config", return_value=raw_config):
            service = svc_mod.get_email_service()
        assert service is not None
        assert isinstance(service, EmailService)
        # Backends are resolved lazily per user; nothing is pre-bound.
        assert service.active_backend is None
    finally:
        svc_mod._email_service = original


def test_configured_service_uses_real_backend_for_fetch() -> None:
    """EmailService delegates fetch_unread to the bound backend for its owner."""
    from datetime import UTC, datetime

    from rex.email_backends.base import EmailEnvelope

    env = EmailEnvelope(
        message_id="real-001",
        from_addr="boss@corp.example.com",
        subject="Q1 Report",
        snippet="Please review the attached Q1 report.",
        received_at=datetime(2026, 4, 1, tzinfo=UTC),
        to_addrs=["user@corp.example.com"],
        labels=["unread"],
    )
    mock_backend = MagicMock()
    mock_backend.connect.return_value = True
    mock_backend.fetch_unread.return_value = [env]

    service = EmailService(backend=mock_backend)
    service.connected = True

    results = service.fetch_unread(limit=5, user_id="default")
    assert len(results) == 1
    assert results[0].id == "real-001"
    mock_backend.fetch_unread.assert_called_once_with(limit=5)
