"""Tests for calendar account management CLI commands (issue #303).

Covers: rex calendar accounts list, set-active, test-connection, upcoming —
all user-scoped.  Every subcommand resolves one validated user, fails closed
without identity, and reveals only that user's accounts.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from rex.calendar_accounts import CalendarAccountResolver
from rex.cli import cmd_calendar

TWO_USER_RAW = {
    "calendar": {
        "accounts": [
            {
                "id": "alice-cal",
                "label": "Alice calendar",
                "provider": "ics",
                "ics": {"source": "https://example.com/alice.ics"},
            },
            {
                "id": "bob-cal",
                "label": "Bob calendar",
                "provider": "google",
                "credential_ref": "GOOGLE_CALENDAR_TOKEN_BOB",
            },
        ],
    },
    "users": {
        "alice": {"calendar_accounts": [{"account_id": "alice-cal"}]},
        "bob": {"calendar_accounts": [{"account_id": "bob-cal"}]},
    },
}


@pytest.fixture
def mock_calendar_service():
    with patch("rex.cli.get_calendar_service") as mock:
        service = MagicMock()
        service.connected = False
        mock.return_value = service
        yield service


@pytest.fixture
def _no_calendar_resolver():
    with patch("rex.cli._load_calendar_resolver_safe", return_value=None):
        yield


@pytest.fixture
def _two_user_resolver():
    resolver = CalendarAccountResolver.from_raw_config(TWO_USER_RAW)
    with patch("rex.cli._load_calendar_resolver_safe", return_value=resolver):
        yield resolver


def _args(**kwargs) -> MagicMock:
    kwargs.setdefault("user", "alice")
    return MagicMock(**kwargs)


class TestCalendarRequiresUser:
    def test_missing_identity_fails_closed(self, mock_calendar_service, capsys):
        with patch("rex.cli._resolve_cli_user", return_value=None):
            args = _args(calendar_command="upcoming", user=None)
            result = cmd_calendar(args)
        assert result == 1
        out = capsys.readouterr().out
        assert "No active user for calendar" in out
        assert "rex identify" in out

    def test_invalid_identity_fails_closed(self, mock_calendar_service, capsys):
        args = _args(calendar_command="upcoming", user="../evil")
        result = cmd_calendar(args)
        assert result == 1
        out = capsys.readouterr().out
        assert "Error" in out

    def test_upcoming_passes_resolved_user(self, mock_calendar_service, capsys):
        mock_calendar_service.connected = True
        mock_calendar_service.get_upcoming_events.return_value = []
        args = _args(calendar_command="upcoming", days=7, conflicts=False, verbose=False)
        result = cmd_calendar(args)
        assert result == 0
        kwargs = mock_calendar_service.get_upcoming_events.call_args.kwargs
        assert kwargs["user_id"] == "alice"

    def test_upcoming_connects_as_resolved_user(self, mock_calendar_service, capsys):
        mock_calendar_service.connected = False
        mock_calendar_service.connect.return_value = True
        mock_calendar_service.get_upcoming_events.return_value = []
        args = _args(calendar_command="upcoming", days=7, conflicts=False, verbose=False)
        result = cmd_calendar(args)
        assert result == 0
        mock_calendar_service.connect.assert_called_once_with("alice")


class TestCalendarAccountsList:
    def test_no_accounts_configured(self, _no_calendar_resolver, capsys):
        args = _args(calendar_command="accounts", accounts_command="list")
        result = cmd_calendar(args)
        assert result == 0
        out = capsys.readouterr().out
        assert "No calendar accounts configured for user 'alice'" in out

    def test_list_shows_only_own_accounts(self, _two_user_resolver, capsys):
        args = _args(calendar_command="accounts", accounts_command="list", user="alice")
        result = cmd_calendar(args)
        assert result == 0
        out = capsys.readouterr().out
        assert "alice-cal" in out
        # Bob's account and label are never revealed to Alice.
        assert "bob-cal" not in out
        assert "Bob calendar" not in out

    def test_list_never_prints_credential_references(self, _two_user_resolver, capsys):
        args = _args(calendar_command="accounts", accounts_command="list", user="bob")
        result = cmd_calendar(args)
        assert result == 0
        out = capsys.readouterr().out
        assert "bob-cal" in out
        assert "GOOGLE_CALENDAR_TOKEN_BOB" not in out

    def test_user_with_no_assignments_sees_nothing(self, _two_user_resolver, capsys):
        args = _args(calendar_command="accounts", accounts_command="list", user="charlie")
        result = cmd_calendar(args)
        assert result == 0
        out = capsys.readouterr().out
        assert "No calendar accounts configured for user 'charlie'" in out
        assert "alice-cal" not in out
        assert "bob-cal" not in out


class TestCalendarAccountsSetActive:
    def test_set_active_no_config(self, _no_calendar_resolver, capsys):
        args = _args(
            calendar_command="accounts", accounts_command="set-active", account_id="alice-cal"
        )
        result = cmd_calendar(args)
        assert result == 1
        out = capsys.readouterr().out
        assert "Error" in out

    def test_set_active_foreign_account_rejected(self, _two_user_resolver, capsys):
        args = _args(
            calendar_command="accounts",
            accounts_command="set-active",
            account_id="bob-cal",
            user="alice",
        )
        result = cmd_calendar(args)
        assert result == 1
        out = capsys.readouterr().out
        assert "not available for user 'alice'" in out

    def test_set_active_nonexistent_matches_foreign(self, _two_user_resolver, capsys):
        args = _args(
            calendar_command="accounts",
            accounts_command="set-active",
            account_id="no-such",
            user="alice",
        )
        result = cmd_calendar(args)
        assert result == 1
        out = capsys.readouterr().out
        assert "not available for user 'alice'" in out

    def test_set_active_updates_only_selected_users_default(
        self, _two_user_resolver, tmp_path, monkeypatch, capsys
    ):
        monkeypatch.chdir(tmp_path)
        config_path = tmp_path / "config" / "rex_config.json"
        config_path.parent.mkdir()
        config_path.write_text(
            json.dumps({"users": {"bob": {"default_calendar_account_id": "bob-cal"}}}),
            encoding="utf-8",
        )

        args = _args(
            calendar_command="accounts",
            accounts_command="set-active",
            account_id="alice-cal",
            user="alice",
        )
        result = cmd_calendar(args)
        assert result == 0
        out = capsys.readouterr().out
        assert "for user 'alice'" in out

        written = json.loads(config_path.read_text(encoding="utf-8"))
        assert written["users"]["alice"]["default_calendar_account_id"] == "alice-cal"
        # Bob's default and the legacy global default are untouched.
        assert written["users"]["bob"]["default_calendar_account_id"] == "bob-cal"
        assert "default_account_id" not in written.get("calendar", {})


class TestCalendarTestConnection:
    def test_no_accounts_stub(self, _no_calendar_resolver, capsys):
        args = _args(calendar_command="test-connection", account_id=None)
        result = cmd_calendar(args)
        assert result == 0
        out = capsys.readouterr().out
        assert "stub" in out.lower()

    def test_foreign_account_not_revealed(self, _two_user_resolver, capsys):
        args = _args(calendar_command="test-connection", account_id="bob-cal", user="alice")
        result = cmd_calendar(args)
        assert result == 1
        out = capsys.readouterr().out
        assert "Available for user 'alice': alice-cal" in out
        assert "GOOGLE_CALENDAR_TOKEN_BOB" not in out

    def test_user_without_accounts_gets_generic_result(self, _two_user_resolver, capsys):
        args = _args(calendar_command="test-connection", account_id=None, user="charlie")
        result = cmd_calendar(args)
        assert result == 1
        out = capsys.readouterr().out
        assert "No calendar accounts configured for user 'charlie'" in out
        assert "alice-cal" not in out

    def test_own_google_account_checks_own_credential_without_printing_ref(
        self, _two_user_resolver, monkeypatch, capsys
    ):
        monkeypatch.setenv("GOOGLE_CALENDAR_TOKEN_BOB", "bob-token-value")
        args = _args(calendar_command="test-connection", account_id=None, user="bob")
        result = cmd_calendar(args)
        assert result == 0
        out = capsys.readouterr().out
        assert "Connection test passed" in out
        # Neither the credential reference nor the token value is echoed.
        assert "GOOGLE_CALENDAR_TOKEN_BOB" not in out
        assert "bob-token-value" not in out

    def test_missing_credential_fails_without_leaking_ref(
        self, _two_user_resolver, monkeypatch, capsys
    ):
        monkeypatch.delenv("GOOGLE_CALENDAR_TOKEN_BOB", raising=False)
        args = _args(calendar_command="test-connection", account_id=None, user="bob")
        result = cmd_calendar(args)
        assert result == 1
        out = capsys.readouterr().out
        assert "NOT FOUND" in out
        assert "GOOGLE_CALENDAR_TOKEN_BOB" not in out
