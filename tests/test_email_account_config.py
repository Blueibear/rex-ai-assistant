"""Tests for multi-account email configuration parsing and routing.

Covers:
- Config model validation
- Account selection logic (explicit, default, fallback)
- ``load_email_config`` from raw dicts
- Edge cases (empty config, unknown IDs, etc.)
"""

from __future__ import annotations

import pytest

from rex.email_backends.account_config import (
    EmailAccountConfig,
    EmailConfig,
    ImapServerConfig,
    SmtpServerConfig,
    load_email_config,
)


def _sample_account(account_id: str = "personal", address: str = "me@gmail.com") -> dict:
    return {
        "id": account_id,
        "label": f"{account_id} account",
        "address": address,
        "imap": {"host": "imap.gmail.com", "port": 993, "ssl": True},
        "smtp": {"host": "smtp.gmail.com", "port": 587, "starttls": True},
        "credential_ref": f"email:{account_id}",
    }


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------


class TestImapServerConfig:
    def test_defaults(self):
        cfg = ImapServerConfig(host="imap.example.com")
        assert cfg.port == 993
        assert cfg.ssl is True

    def test_custom_port(self):
        cfg = ImapServerConfig(host="imap.example.com", port=143, ssl=False)
        assert cfg.port == 143
        assert cfg.ssl is False


class TestSmtpServerConfig:
    def test_defaults(self):
        cfg = SmtpServerConfig(host="smtp.example.com")
        assert cfg.port == 587
        assert cfg.starttls is True

    def test_smtps(self):
        cfg = SmtpServerConfig(host="smtp.example.com", port=465, starttls=False)
        assert cfg.port == 465
        assert cfg.starttls is False


class TestEmailAccountConfig:
    def test_valid_construction(self):
        acct = EmailAccountConfig(**_sample_account())
        assert acct.id == "personal"
        assert acct.address == "me@gmail.com"
        assert acct.imap.host == "imap.gmail.com"
        assert acct.smtp.host == "smtp.gmail.com"

    def test_missing_required_field(self):
        with pytest.raises((TypeError, ValueError)):
            EmailAccountConfig(id="bad", address="a@b.com")  # missing imap/smtp


# ---------------------------------------------------------------------------
# EmailConfig account selection
# ---------------------------------------------------------------------------


class TestEmailConfig:
    def test_empty_config(self):
        cfg = EmailConfig()
        assert cfg.accounts == []
        assert cfg.get_account() is None
        assert cfg.list_account_ids() == []

    def test_explicit_account_id(self):
        cfg = EmailConfig(
            accounts=[
                EmailAccountConfig(**_sample_account("personal")),
                EmailAccountConfig(**_sample_account("work", "work@corp.com")),
            ],
        )
        acct = cfg.get_account("work")
        assert acct is not None
        assert acct.id == "work"

    def test_default_account_id(self):
        cfg = EmailConfig(
            default_account_id="work",
            accounts=[
                EmailAccountConfig(**_sample_account("personal")),
                EmailAccountConfig(**_sample_account("work", "work@corp.com")),
            ],
        )
        acct = cfg.get_account()  # no explicit
        assert acct is not None
        assert acct.id == "work"

    def test_fallback_to_first_account(self):
        cfg = EmailConfig(
            accounts=[
                EmailAccountConfig(**_sample_account("personal")),
                EmailAccountConfig(**_sample_account("work", "work@corp.com")),
            ],
        )
        acct = cfg.get_account()  # no explicit, no default
        assert acct is not None
        assert acct.id == "personal"  # first in list

    def test_explicit_overrides_default(self):
        cfg = EmailConfig(
            default_account_id="work",
            accounts=[
                EmailAccountConfig(**_sample_account("personal")),
                EmailAccountConfig(**_sample_account("work", "work@corp.com")),
            ],
        )
        acct = cfg.get_account("personal")  # explicit overrides default
        assert acct is not None
        assert acct.id == "personal"

    def test_unknown_explicit_returns_none(self):
        cfg = EmailConfig(
            accounts=[EmailAccountConfig(**_sample_account("personal"))],
        )
        assert cfg.get_account("nonexistent") is None

    def test_unknown_default_falls_back(self):
        cfg = EmailConfig(
            default_account_id="deleted",
            accounts=[EmailAccountConfig(**_sample_account("personal"))],
        )
        acct = cfg.get_account()
        assert acct is not None
        assert acct.id == "personal"  # falls back to first

    def test_list_account_ids(self):
        cfg = EmailConfig(
            accounts=[
                EmailAccountConfig(**_sample_account("a")),
                EmailAccountConfig(**_sample_account("b", "b@b.com")),
            ],
        )
        assert cfg.list_account_ids() == ["a", "b"]


# ---------------------------------------------------------------------------
# load_email_config from raw dict
# ---------------------------------------------------------------------------


class TestLoadEmailConfig:
    def test_empty_raw_config(self):
        cfg = load_email_config({})
        assert cfg.accounts == []

    def test_no_email_key(self):
        cfg = load_email_config({"runtime": {}})
        assert cfg.accounts == []

    def test_full_config(self):
        raw = {
            "email": {
                "default_account_id": "work",
                "accounts": [
                    _sample_account("personal"),
                    _sample_account("work", "work@corp.com"),
                ],
            }
        }
        cfg = load_email_config(raw)
        assert len(cfg.accounts) == 2
        assert cfg.default_account_id == "work"
        assert cfg.get_account().id == "work"

    def test_invalid_account_raises(self):
        raw = {
            "email": {
                "accounts": [{"id": "bad"}],  # missing required fields
            }
        }
        with pytest.raises((TypeError, ValueError)):
            load_email_config(raw)
