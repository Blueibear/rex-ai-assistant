"""Tests for the email account router (backend selection logic).

Covers:
- Stub fallback when no accounts configured
- Real backend creation when credentials available
- Stub fallback when credentials missing
- Credential parsing (username:password vs password-only)
"""

from __future__ import annotations

from rex.email_backends.account_config import (
    EmailAccountConfig,
    EmailConfig,
    ImapServerConfig,
    SmtpServerConfig,
)
from rex.email_backends.account_router import resolve_backend
from rex.email_backends.stub import StubEmailBackend


def _make_account(account_id: str = "test") -> EmailAccountConfig:
    return EmailAccountConfig(
        id=account_id,
        label="Test Account",
        address="user@example.com",
        imap=ImapServerConfig(host="imap.example.com", port=993, ssl=True),
        smtp=SmtpServerConfig(host="smtp.example.com", port=587, starttls=True),
        credential_ref=f"email:{account_id}",
    )


class TestResolveBackend:
    def test_no_accounts_returns_stub(self):
        config = EmailConfig()
        backend, acct = resolve_backend(config)
        assert isinstance(backend, StubEmailBackend)
        assert acct is None

    def test_no_credentials_returns_stub(self):
        config = EmailConfig(accounts=[_make_account()])
        backend, acct = resolve_backend(
            config,
            credential_getter=lambda ref: None,
        )
        assert isinstance(backend, StubEmailBackend)
        assert acct is None

    def test_empty_credential_returns_stub(self):
        config = EmailConfig(accounts=[_make_account()])
        backend, acct = resolve_backend(
            config,
            credential_getter=lambda ref: "",
        )
        assert isinstance(backend, StubEmailBackend)

    def test_valid_credential_returns_imap_smtp(self):
        from rex.email_backends.imap_smtp import ImapSmtpEmailBackend

        config = EmailConfig(accounts=[_make_account()])
        backend, acct = resolve_backend(
            config,
            credential_getter=lambda ref: "user@example.com:app_password_123",
        )
        assert isinstance(backend, ImapSmtpEmailBackend)
        assert acct is not None
        assert acct.id == "test"

    def test_password_only_credential(self):
        from rex.email_backends.imap_smtp import ImapSmtpEmailBackend

        config = EmailConfig(accounts=[_make_account()])
        backend, acct = resolve_backend(
            config,
            credential_getter=lambda ref: "app_password_only",
        )
        assert isinstance(backend, ImapSmtpEmailBackend)
        assert acct is not None

    def test_explicit_account_id(self):
        from rex.email_backends.imap_smtp import ImapSmtpEmailBackend

        config = EmailConfig(
            default_account_id="other",
            accounts=[_make_account("test"), _make_account("other")],
        )
        backend, acct = resolve_backend(
            config,
            account_id="test",
            credential_getter=lambda ref: "u:p",
        )
        assert isinstance(backend, ImapSmtpEmailBackend)
        assert acct is not None
        assert acct.id == "test"

    def test_stub_fixture_path_passed_through(self, tmp_path):
        config = EmailConfig()
        fixture = tmp_path / "custom.json"
        backend, _ = resolve_backend(config, stub_fixture=fixture)
        assert isinstance(backend, StubEmailBackend)
        assert backend._fixture_path == fixture

    def test_credential_getter_exception_returns_stub(self):
        config = EmailConfig(accounts=[_make_account()])

        def bad_getter(ref):
            raise RuntimeError("boom")

        backend, acct = resolve_backend(config, credential_getter=bad_getter)
        assert isinstance(backend, StubEmailBackend)
