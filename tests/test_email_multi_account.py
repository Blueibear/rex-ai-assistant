"""Tests for US-208: multi-account email config and routing.

Covers:
- AppConfig parses email.accounts[] and email.default_account_id from JSON
- EmailService routes fetch_unread/send to the correct backend by account_id
- Falls back to default_account_id when account_id is omitted
- Raises ValueError with an actionable message for unknown account_id
- Backward-compatible: single legacy-style account config works
- No live network calls (backend_factory DI)
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from rex.config import EmailAccountConfig, build_app_config
from rex.integrations.email.service import EmailService

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_stub_backend(sent: list | None = None, unread: list | None = None):
    """Return a MagicMock that behaves like an EmailBackend."""
    backend = MagicMock()
    backend.fetch_unread.return_value = unread or [
        {
            "id": "msg-1",
            "from": "sender@example.com",
            "subject": "Hello",
            "snippet": "Hello",
            "received_at": "2026-03-30T00:00:00+00:00",
        }
    ]
    backend.send.return_value = None
    if sent is not None:
        backend._sent = sent
    return backend


def _make_accounts():
    """Return two EmailAccountConfig fixtures (work + personal)."""
    work = EmailAccountConfig(
        id="work",
        address="james@work.example.com",
        imap_host="imap.work.example.com",
        smtp_host="smtp.work.example.com",
    )
    personal = EmailAccountConfig(
        id="personal",
        address="james@personal.example.com",
        imap_host="imap.personal.example.com",
        smtp_host="smtp.personal.example.com",
    )
    return work, personal


# ---------------------------------------------------------------------------
# AppConfig parsing
# ---------------------------------------------------------------------------


class TestAppConfigEmailAccounts:
    def test_parses_accounts_list(self):
        cfg = build_app_config(
            {
                "email": {
                    "accounts": [
                        {
                            "id": "work",
                            "address": "james@work.example.com",
                            "imap_host": "imap.work.example.com",
                            "smtp_host": "smtp.work.example.com",
                        }
                    ],
                    "default_account_id": "work",
                }
            }
        )
        assert len(cfg.email_accounts) == 1
        acc = cfg.email_accounts[0]
        assert acc.id == "work"
        assert acc.address == "james@work.example.com"
        assert acc.imap_host == "imap.work.example.com"
        assert acc.smtp_host == "smtp.work.example.com"
        assert cfg.email_default_account_id == "work"

    def test_parses_nested_imap_smtp_dicts(self):
        """Support the nested imap/smtp sub-dict form."""
        cfg = build_app_config(
            {
                "email": {
                    "accounts": [
                        {
                            "id": "personal",
                            "address": "me@personal.com",
                            "imap": {"host": "imap.personal.com", "port": 993},
                            "smtp": {"host": "smtp.personal.com", "port": 465},
                            "use_starttls": False,
                        }
                    ]
                }
            }
        )
        acc = cfg.email_accounts[0]
        assert acc.imap_host == "imap.personal.com"
        assert acc.imap_port == 993
        assert acc.smtp_host == "smtp.personal.com"
        assert acc.smtp_port == 465
        assert acc.use_starttls is False

    def test_parses_multiple_accounts(self):
        cfg = build_app_config(
            {
                "email": {
                    "accounts": [
                        {"id": "a", "address": "a@example.com", "imap_host": "imap.a.com"},
                        {"id": "b", "address": "b@example.com", "imap_host": "imap.b.com"},
                    ],
                    "default_account_id": "a",
                }
            }
        )
        assert len(cfg.email_accounts) == 2
        assert cfg.email_accounts[0].id == "a"
        assert cfg.email_accounts[1].id == "b"

    def test_skips_malformed_entries(self):
        """Missing required 'id' field is skipped without raising."""
        cfg = build_app_config(
            {
                "email": {
                    "accounts": [
                        {"address": "no-id@example.com"},  # missing 'id'
                        {"id": "ok", "address": "ok@example.com", "imap_host": "imap.ok.com"},
                    ]
                }
            }
        )
        assert len(cfg.email_accounts) == 1
        assert cfg.email_accounts[0].id == "ok"

    def test_empty_accounts_is_backward_compatible(self):
        """Config with no email section uses defaults."""
        cfg = build_app_config({})
        assert cfg.email_accounts == []
        assert cfg.email_default_account_id == ""


# ---------------------------------------------------------------------------
# EmailService routing
# ---------------------------------------------------------------------------


class TestEmailServiceRouting:
    def _make_service(self, accounts=None, default_id="work"):
        work, personal = _make_accounts()
        all_accounts = accounts if accounts is not None else [work, personal]
        backends: dict[str, MagicMock] = {
            "work": _make_stub_backend(),
            "personal": _make_stub_backend(),
        }

        def factory(account: EmailAccountConfig) -> MagicMock:
            return backends[account.id]

        svc = EmailService(all_accounts, default_account_id=default_id, backend_factory=factory)
        return svc, backends

    def test_fetch_unread_uses_default_when_no_account_id(self):
        svc, backends = self._make_service(default_id="work")
        svc.fetch_unread(limit=5)
        backends["work"].fetch_unread.assert_called_once_with(limit=5)
        backends["personal"].fetch_unread.assert_not_called()

    def test_fetch_unread_routes_to_specified_account(self):
        svc, backends = self._make_service(default_id="work")
        svc.fetch_unread(limit=3, account_id="personal")
        backends["personal"].fetch_unread.assert_called_once_with(limit=3)
        backends["work"].fetch_unread.assert_not_called()

    def test_send_uses_default_when_no_account_id(self):
        svc, backends = self._make_service(default_id="personal")
        svc.send("to@example.com", "Subject", "Body")
        backends["personal"].send.assert_called_once_with(
            to="to@example.com", subject="Subject", body="Body"
        )
        backends["work"].send.assert_not_called()

    def test_send_routes_to_specified_account(self):
        svc, backends = self._make_service(default_id="personal")
        svc.send("to@example.com", "Subject", "Body", account_id="work")
        backends["work"].send.assert_called_once_with(
            to="to@example.com", subject="Subject", body="Body"
        )
        backends["personal"].send.assert_not_called()

    def test_invalid_account_id_raises_value_error_fetch(self):
        svc, _ = self._make_service()
        with pytest.raises(ValueError, match="Unknown email account_id 'bogus'"):
            svc.fetch_unread(account_id="bogus")

    def test_invalid_account_id_raises_value_error_send(self):
        svc, _ = self._make_service()
        with pytest.raises(ValueError, match="Unknown email account_id 'nope'"):
            svc.send("x@y.com", "S", "B", account_id="nope")

    def test_error_message_lists_known_accounts(self):
        svc, _ = self._make_service()
        with pytest.raises(ValueError) as exc_info:
            svc.fetch_unread(account_id="missing")
        assert "work" in str(exc_info.value)
        assert "personal" in str(exc_info.value)

    def test_backend_constructed_lazily(self):
        call_log: list[str] = []
        work, personal = _make_accounts()

        def factory(account: EmailAccountConfig):
            call_log.append(account.id)
            return _make_stub_backend()

        svc = EmailService([work, personal], default_account_id="work", backend_factory=factory)
        assert call_log == []
        svc.fetch_unread()
        assert call_log == ["work"]
        svc.fetch_unread()
        assert call_log == ["work"]  # not constructed again

    def test_account_ids_property(self):
        svc, _ = self._make_service()
        assert sorted(svc.account_ids) == ["personal", "work"]


# ---------------------------------------------------------------------------
# No-accounts edge cases
# ---------------------------------------------------------------------------


class TestEmailServiceNoAccounts:
    def test_fetch_unread_returns_empty_list_when_no_accounts(self):
        svc = EmailService([])
        result = svc.fetch_unread()
        assert result == []

    def test_send_raises_value_error_when_no_accounts(self):
        svc = EmailService([])
        with pytest.raises(ValueError, match="no accounts are configured"):
            svc.send("to@example.com", "S", "B")


# ---------------------------------------------------------------------------
# Default account fallback — first account wins when default_id is empty
# ---------------------------------------------------------------------------


class TestEmailServiceDefaultFallback:
    def test_first_account_is_default_when_default_id_not_set(self):
        work, personal = _make_accounts()
        backends: dict[str, MagicMock] = {
            "work": _make_stub_backend(),
            "personal": _make_stub_backend(),
        }

        def factory(account: EmailAccountConfig) -> MagicMock:
            return backends[account.id]

        svc = EmailService([work, personal], default_account_id="", backend_factory=factory)
        svc.fetch_unread()
        backends["work"].fetch_unread.assert_called_once()
        backends["personal"].fetch_unread.assert_not_called()
