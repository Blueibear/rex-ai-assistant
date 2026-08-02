"""Tests for email account management CLI commands.

Covers: rex email accounts list, set-active, send, test-connection, unread —
all user-scoped (issue #303).  Every subcommand resolves one validated user,
fails closed without identity, and reveals only that user's accounts.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from rex.cli import cmd_email
from rex.email_accounts import EmailAccountResolver
from rex.email_backends.account_config import EmailConfig


def _make_resolver(accounts: list[dict], users: dict, defaults: dict | None = None):
    """Build a real EmailAccountResolver from plain dicts."""
    from rex.config import _parse_user_email_accounts

    email_config = EmailConfig.model_validate({"accounts": accounts})
    user_accounts = _parse_user_email_accounts(users, accounts)
    return EmailAccountResolver(email_config, user_accounts, defaults or {})


def _account(account_id: str, stem: str) -> dict:
    return {
        "id": account_id,
        "label": f"{account_id} label",
        "address": f"{account_id}@{stem}.example.com",
        "imap": {"host": f"imap.{stem}.example.com"},
        "smtp": {"host": f"smtp.{stem}.example.com"},
        "credential_ref": f"email:{account_id}",
    }


TWO_USER_ACCOUNTS = [_account("alice-work", "alice"), _account("bob-personal", "bob")]
TWO_USER_USERS = {
    "alice": {"email_accounts": [{"account_id": "alice-work"}]},
    "bob": {"email_accounts": [{"account_id": "bob-personal"}]},
}


@pytest.fixture
def mock_email_service():
    """Mock email service instance."""
    with patch("rex.cli.get_email_service") as mock:
        service = MagicMock()
        service.connected = False
        mock.return_value = service
        yield service


@pytest.fixture
def _no_email_resolver():
    """Patch _load_email_resolver_safe to return None (no config)."""
    with patch("rex.cli._load_email_resolver_safe", return_value=None):
        yield


@pytest.fixture
def _two_user_resolver():
    """Patch _load_email_resolver_safe with a two-user resolver."""
    resolver = _make_resolver(TWO_USER_ACCOUNTS, TWO_USER_USERS)
    with patch("rex.cli._load_email_resolver_safe", return_value=resolver):
        yield resolver


def _args(**kwargs) -> MagicMock:
    kwargs.setdefault("user", "alice")
    return MagicMock(**kwargs)


class TestEmailRequiresUser:
    def test_missing_identity_fails_closed(self, mock_email_service, capsys):
        with patch("rex.cli._resolve_cli_user", return_value=None):
            args = _args(email_command="unread", user=None)
            result = cmd_email(args)
        assert result == 1
        out = capsys.readouterr().out
        assert "No active user for email" in out
        assert "rex identify" in out

    def test_invalid_identity_fails_closed(self, mock_email_service, capsys):
        args = _args(email_command="unread", user="../evil")
        result = cmd_email(args)
        assert result == 1
        out = capsys.readouterr().out
        assert "Error" in out


class TestEmailAccountsList:
    def test_no_accounts_configured(self, mock_email_service, _no_email_resolver, capsys):
        args = _args(email_command="accounts", accounts_command="list")
        result = cmd_email(args)
        assert result == 0
        out = capsys.readouterr().out
        assert "No email accounts configured for user 'alice'" in out

    def test_list_shows_only_own_accounts(self, mock_email_service, _two_user_resolver, capsys):
        args = _args(email_command="accounts", accounts_command="list", user="alice")
        result = cmd_email(args)
        assert result == 0
        out = capsys.readouterr().out
        assert "alice-work" in out
        assert "alice-work@alice.example.com" in out
        # Bob's account, address, and label are never revealed to Alice.
        assert "bob-personal" not in out
        assert "bob.example.com" not in out
        assert "bob-personal label" not in out

    def test_user_with_no_assignments_sees_nothing(
        self, mock_email_service, _two_user_resolver, capsys
    ):
        args = _args(email_command="accounts", accounts_command="list", user="charlie")
        result = cmd_email(args)
        assert result == 0
        out = capsys.readouterr().out
        assert "No email accounts configured for user 'charlie'" in out
        assert "alice-work" not in out
        assert "bob-personal" not in out


class TestEmailAccountsSetActive:
    def test_set_active_no_config(self, mock_email_service, _no_email_resolver, capsys):
        args = _args(email_command="accounts", accounts_command="set-active", account_id="work")
        result = cmd_email(args)
        assert result == 1
        out = capsys.readouterr().out
        assert "Error" in out

    def test_set_active_foreign_account_rejected(
        self, mock_email_service, _two_user_resolver, capsys
    ):
        args = _args(
            email_command="accounts",
            accounts_command="set-active",
            account_id="bob-personal",
            user="alice",
        )
        result = cmd_email(args)
        assert result == 1
        out = capsys.readouterr().out
        assert "not available for user 'alice'" in out
        # Only Alice's own accounts may be suggested.
        assert "bob-personal" not in out.replace("Account 'bob-personal'", "")

    def test_set_active_nonexistent_matches_foreign(
        self, mock_email_service, _two_user_resolver, capsys
    ):
        args = _args(
            email_command="accounts",
            accounts_command="set-active",
            account_id="no-such",
            user="alice",
        )
        result = cmd_email(args)
        assert result == 1
        out = capsys.readouterr().out
        assert "not available for user 'alice'" in out

    def test_set_active_updates_only_selected_users_default(
        self, mock_email_service, _two_user_resolver, tmp_path, monkeypatch, capsys
    ):
        monkeypatch.chdir(tmp_path)
        config_path = tmp_path / "config" / "rex_config.json"
        config_path.parent.mkdir()
        config_path.write_text(
            json.dumps({"users": {"bob": {"default_email_account_id": "bob-personal"}}}),
            encoding="utf-8",
        )

        args = _args(
            email_command="accounts",
            accounts_command="set-active",
            account_id="alice-work",
            user="alice",
        )
        result = cmd_email(args)
        assert result == 0
        out = capsys.readouterr().out
        assert "for user 'alice'" in out

        written = json.loads(config_path.read_text(encoding="utf-8"))
        assert written["users"]["alice"]["default_email_account_id"] == "alice-work"
        # Bob's default and the legacy global default are untouched.
        assert written["users"]["bob"]["default_email_account_id"] == "bob-personal"
        assert "email" not in written or "default_account_id" not in written.get("email", {})


class TestEmailSend:
    def test_send_missing_args(self, mock_email_service, capsys):
        args = _args(email_command="send", to=None, subject=None, body=None, account_id=None)
        result = cmd_email(args)
        assert result == 1
        out = capsys.readouterr().out
        assert "required" in out

    def test_send_passes_resolved_user(self, mock_email_service, capsys):
        mock_email_service.connected = True
        mock_email_service.send.return_value = {"ok": True, "message_id": "test-123"}
        args = _args(
            email_command="send",
            to="recipient@example.com",
            subject="Test",
            body="Hello",
            account_id=None,
            user="alice",
        )
        result = cmd_email(args)
        assert result == 0
        out = capsys.readouterr().out
        assert "sent successfully" in out
        kwargs = mock_email_service.send.call_args.kwargs
        assert kwargs["user_id"] == "alice"
        assert kwargs["account_id"] is None

    def test_send_foreign_account_generic_error(self, mock_email_service, capsys):
        mock_email_service.connected = True
        mock_email_service.send.side_effect = PermissionError(
            "Email account 'bob-personal' is not available for user 'alice'"
        )
        args = _args(
            email_command="send",
            to="x@example.com",
            subject="s",
            body="b",
            account_id="bob-personal",
            user="alice",
        )
        result = cmd_email(args)
        assert result == 1
        out = capsys.readouterr().out
        assert "not available" in out

    def test_send_connection_failure(self, mock_email_service, capsys):
        mock_email_service.connected = False
        mock_email_service.connect.return_value = False
        args = _args(
            email_command="send",
            to="recipient@example.com",
            subject="Test",
            body="Hello",
            account_id=None,
        )
        result = cmd_email(args)
        assert result == 1
        out = capsys.readouterr().out
        assert "Failed to connect" in out


class TestEmailUnread:
    def test_unread_passes_resolved_user(self, mock_email_service, capsys):
        mock_email_service.connected = True
        mock_email_service.fetch_unread.return_value = []
        args = _args(email_command="unread", limit=5, verbose=False, user="alice")
        result = cmd_email(args)
        assert result == 0
        kwargs = mock_email_service.fetch_unread.call_args.kwargs
        assert kwargs["user_id"] == "alice"

    def test_unread_connects_as_resolved_user(self, mock_email_service, capsys):
        mock_email_service.connected = False
        mock_email_service.connect.return_value = True
        mock_email_service.fetch_unread.return_value = []
        args = _args(email_command="unread", limit=5, verbose=False, user="alice")
        result = cmd_email(args)
        assert result == 0
        mock_email_service.connect.assert_called_once_with("alice")


class TestEmailTestConnection:
    def test_no_accounts_stub(self, mock_email_service, _no_email_resolver, capsys):
        args = _args(email_command="test-connection", account_id=None)
        result = cmd_email(args)
        assert result == 0
        out = capsys.readouterr().out
        assert "stub" in out.lower()

    def test_foreign_account_not_revealed(self, mock_email_service, _two_user_resolver, capsys):
        args = _args(email_command="test-connection", account_id="bob-personal", user="alice")
        result = cmd_email(args)
        assert result == 1
        out = capsys.readouterr().out
        assert "Available for user 'alice': alice-work" in out
        # Bob's address/host details are never revealed.
        assert "bob.example.com" not in out

    def test_user_without_accounts_gets_generic_result(
        self, mock_email_service, _two_user_resolver, capsys
    ):
        args = _args(email_command="test-connection", account_id=None, user="charlie")
        result = cmd_email(args)
        assert result == 1
        out = capsys.readouterr().out
        assert "No email accounts configured for user 'charlie'" in out
        assert "alice-work" not in out

    def test_own_account_is_tested_with_own_credential_ref(
        self, mock_email_service, _two_user_resolver, capsys
    ):
        lookups: list[str] = []

        class FakeCM:
            def get_token(self, ref, **context):
                lookups.append(ref)
                assert context == {
                    "integration": "email",
                    "account": "alice-work",
                    "slot": "password",
                }
                return "user:pass"

        fake_backend = MagicMock()
        fake_backend.connect.return_value = True

        with patch("rex.credentials.CredentialManager", return_value=FakeCM()) as manager:
            with patch(
                "rex.email_backends.account_router.build_backend_for_account",
                return_value=fake_backend,
            ) as build:
                args = _args(email_command="test-connection", account_id=None, user="alice")
                result = cmd_email(args)

        assert result == 0
        manager.assert_called_once_with(scope="user", user_id="alice")
        out = capsys.readouterr().out
        assert "alice-work" in out
        assert lookups == ["email:alice-work"]
        assert build.call_args.args[0].id == "alice-work"
