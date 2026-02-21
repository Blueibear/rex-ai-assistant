"""Tests for email account management CLI commands (BL-011 gap fill).

Covers: rex email accounts list, set-active, send, test-connection.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from rex.cli import cmd_email


@pytest.fixture
def mock_email_service():
    """Mock email service instance."""
    with patch("rex.cli.get_email_service") as mock:
        service = MagicMock()
        service.connected = False
        mock.return_value = service
        yield service


@pytest.fixture
def _no_email_config():
    """Patch _load_email_config_safe to return None (no config)."""
    with patch("rex.cli._load_email_config_safe", return_value=None):
        yield


@pytest.fixture
def _stub_email_config():
    """Patch _load_email_config_safe to return a mock config with accounts."""
    config = MagicMock()
    acct = MagicMock()
    acct.id = "personal"
    acct.label = "Personal"
    acct.address = "user@example.com"
    acct.imap.host = "imap.example.com"
    acct.imap.port = 993
    acct.imap.ssl = True
    acct.smtp.host = "smtp.example.com"
    acct.smtp.port = 587
    acct.smtp.starttls = True
    acct.credential_ref = "email:personal"
    config.accounts = [acct]
    config.default_account_id = "personal"
    config.list_account_ids.return_value = ["personal"]
    config.get_account.return_value = acct
    with patch("rex.cli._load_email_config_safe", return_value=config):
        yield config


class TestEmailAccountsList:
    def test_no_accounts_configured(self, mock_email_service, _no_email_config, capsys):
        args = MagicMock(email_command="accounts", accounts_command="list", user=None)
        result = cmd_email(args)
        assert result == 0
        out = capsys.readouterr().out
        assert "No email accounts configured" in out

    def test_list_with_accounts(self, mock_email_service, _stub_email_config, capsys):
        args = MagicMock(email_command="accounts", accounts_command="list", user=None)
        result = cmd_email(args)
        assert result == 0
        out = capsys.readouterr().out
        assert "personal" in out
        assert "user@example.com" in out


class TestEmailAccountsSetActive:
    def test_set_active_no_config(self, mock_email_service, _no_email_config, capsys):
        args = MagicMock(
            email_command="accounts",
            accounts_command="set-active",
            account_id="work",
            user=None,
        )
        result = cmd_email(args)
        assert result == 1
        out = capsys.readouterr().out
        assert "Error" in out

    def test_set_active_invalid_id(self, mock_email_service, _stub_email_config, capsys):
        args = MagicMock(
            email_command="accounts",
            accounts_command="set-active",
            account_id="nonexistent",
            user=None,
        )
        result = cmd_email(args)
        assert result == 1
        out = capsys.readouterr().out
        assert "not found" in out


class TestEmailSend:
    def test_send_missing_args(self, mock_email_service, capsys):
        args = MagicMock(
            email_command="send",
            to=None,
            subject=None,
            body=None,
            account_id=None,
            user=None,
        )
        result = cmd_email(args)
        assert result == 1
        out = capsys.readouterr().out
        assert "required" in out

    def test_send_success(self, mock_email_service, capsys):
        mock_email_service.connected = True
        mock_email_service.send.return_value = {"ok": True, "message_id": "test-123"}
        args = MagicMock(
            email_command="send",
            to="recipient@example.com",
            subject="Test",
            body="Hello",
            account_id=None,
            user=None,
        )
        result = cmd_email(args)
        assert result == 0
        out = capsys.readouterr().out
        assert "sent successfully" in out

    def test_send_connection_failure(self, mock_email_service, capsys):
        mock_email_service.connected = False
        mock_email_service.connect.return_value = False
        args = MagicMock(
            email_command="send",
            to="recipient@example.com",
            subject="Test",
            body="Hello",
            account_id=None,
            user=None,
        )
        result = cmd_email(args)
        assert result == 1
        out = capsys.readouterr().out
        assert "Failed to connect" in out


class TestEmailTestConnection:
    def test_no_accounts_stub(self, mock_email_service, _no_email_config, capsys):
        args = MagicMock(
            email_command="test-connection",
            account_id=None,
            user=None,
        )
        result = cmd_email(args)
        assert result == 0
        out = capsys.readouterr().out
        assert "stub" in out.lower()
