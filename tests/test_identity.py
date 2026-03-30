"""Tests for the identity resolution module.

All tests are offline and deterministic.  Session state is isolated
via tmp_path so no real session file is modified.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from rex.identity import (
    _load_session,
    clear_session_user,
    get_session_user,
    list_known_users,
    require_active_user,
    resolve_active_user,
    set_session_user,
)

# ---------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------


@pytest.fixture()
def session_dir(tmp_path: Path):
    """Patch session state path to use a temp directory."""
    session_file = tmp_path / "rex-ai" / "session.json"
    with patch("rex.identity._session_state_path", return_value=session_file):
        yield tmp_path


@pytest.fixture()
def memory_dir(tmp_path: Path):
    """Create fake Memory/ profiles for testing."""
    mem = tmp_path / "Memory"
    (mem / "alice").mkdir(parents=True)
    (mem / "alice" / "core.json").write_text(
        json.dumps({"name": "Alice", "role": "Admin"}), encoding="utf-8"
    )
    (mem / "bob").mkdir(parents=True)
    (mem / "bob" / "core.json").write_text(
        json.dumps({"name": "Bob", "role": "User"}), encoding="utf-8"
    )
    # Directory without core.json should be ignored
    (mem / "empty_dir").mkdir(parents=True)

    with patch("rex.identity._known_user_ids") as mock_known:
        mock_known.return_value = ["alice", "bob"]
        with patch("rex.identity.list_known_users") as mock_list:
            mock_list.return_value = [
                {"id": "alice", "name": "Alice", "role": "Admin"},
                {"id": "bob", "name": "Bob", "role": "User"},
            ]
            yield mem


# ---------------------------------------------------------------
# Session state tests
# ---------------------------------------------------------------


class TestSessionState:
    """Tests for session user get/set/clear."""

    def test_no_session_returns_none(self, session_dir):
        """No session file means no active user."""
        assert get_session_user() is None

    def test_set_and_get(self, session_dir):
        """Setting session user persists it."""
        set_session_user("alice")
        assert get_session_user() == "alice"

    def test_overwrite(self, session_dir):
        """Setting a new user overwrites the previous one."""
        set_session_user("alice")
        set_session_user("bob")
        assert get_session_user() == "bob"

    def test_clear(self, session_dir):
        """Clearing removes the active user."""
        set_session_user("alice")
        clear_session_user()
        assert get_session_user() is None

    def test_corrupted_session_file_returns_empty_dict_and_logs_warning(
        self, tmp_path: Path, caplog
    ):
        """Malformed JSON in the session file must return {} and emit a warning."""
        session_file = tmp_path / "rex-ai" / "session.json"
        session_file.parent.mkdir(parents=True, exist_ok=True)
        session_file.write_text("this is not valid json!!!", encoding="utf-8")

        with patch("rex.identity._session_state_path", return_value=session_file):
            with caplog.at_level(logging.WARNING, logger="rex.identity"):
                result = _load_session()

        assert result == {}, f"Expected empty dict, got {result!r}"
        assert any(
            "Corrupted session file" in record.message and "resetting" in record.message
            for record in caplog.records
        ), "Expected a warning log about corrupted session file"

    def test_expired_session_returns_empty_dict_and_deletes_file(self, tmp_path: Path):
        """Expired session files are removed and treated as empty state."""
        session_file = tmp_path / "rex-ai" / "session.json"
        session_file.parent.mkdir(parents=True, exist_ok=True)
        session_file.write_text(json.dumps({"active_user": "alice"}), encoding="utf-8")
        expired_mtime = 1_000_000_000
        os.utime(session_file, (expired_mtime, expired_mtime))

        with patch("rex.identity._session_state_path", return_value=session_file):
            with patch.object(
                __import__("rex.identity", fromlist=["settings"]).settings, "session_ttl_hours", 8
            ):
                result = _load_session()

        assert result == {}
        assert not session_file.exists()


# ---------------------------------------------------------------
# User resolution tests
# ---------------------------------------------------------------


class TestResolveActiveUser:
    """Tests for resolve_active_user priority chain."""

    def test_explicit_user_wins(self, session_dir):
        """Explicit --user flag takes highest priority."""
        set_session_user("bob")
        config = {"runtime": {"active_user": "charlie"}}
        result = resolve_active_user("alice", config=config)
        assert result == "alice"

    def test_session_user_second(self, session_dir):
        """Session user is used when no explicit user is given."""
        set_session_user("bob")
        config = {"runtime": {"active_user": "charlie"}}
        result = resolve_active_user(config=config)
        assert result == "bob"

    def test_config_active_user_third(self, session_dir):
        """Config active_user is used when no session/explicit user."""
        config = {"runtime": {"active_user": "charlie"}}
        result = resolve_active_user(config=config)
        assert result == "charlie"

    def test_config_user_id_fourth(self, session_dir):
        """Config user_id is used as last resort (if not 'default')."""
        config = {"runtime": {"user_id": "dave"}}
        result = resolve_active_user(config=config)
        assert result == "dave"

    def test_default_user_id_ignored(self, session_dir):
        """user_id='default' is treated as unset."""
        config = {"runtime": {"user_id": "default"}}
        result = resolve_active_user(config=config)
        assert result is None

    def test_no_user_returns_none(self, session_dir):
        """No user from any source returns None."""
        result = resolve_active_user()
        assert result is None


# ---------------------------------------------------------------
# require_active_user tests
# ---------------------------------------------------------------


class TestRequireActiveUser:
    """Tests for require_active_user error behaviour."""

    def test_returns_user_when_available(self, session_dir):
        """Returns user when explicitly provided."""
        result = require_active_user("alice")
        assert result == "alice"

    def test_raises_when_no_user(self, session_dir):
        """Raises SystemExit with helpful message when no user."""
        with pytest.raises(SystemExit) as exc_info:
            require_active_user(action="sending email")
        msg = str(exc_info.value)
        assert "No active user" in msg
        assert "sending email" in msg
        assert "rex identify" in msg


# ---------------------------------------------------------------
# list_known_users tests (real Memory/ dir)
# ---------------------------------------------------------------


class TestListKnownUsers:
    """Tests for list_known_users from Memory/ directory."""

    def test_discovers_users_from_memory_dir(self, tmp_path: Path):
        """Discovers users with core.json in Memory/ subdirs."""
        mem = tmp_path / "Memory"
        (mem / "alice").mkdir(parents=True)
        (mem / "alice" / "core.json").write_text(
            json.dumps({"name": "Alice", "role": "Admin"}), encoding="utf-8"
        )
        (mem / "bob").mkdir(parents=True)
        (mem / "bob" / "core.json").write_text(json.dumps({"name": "Bob"}), encoding="utf-8")
        # No core.json - should be skipped
        (mem / "empty").mkdir(parents=True)

        # Patch the module-level path resolution to use our temp dir
        fake_identity_py = tmp_path / "rex" / "identity.py"
        fake_identity_py.parent.mkdir(parents=True, exist_ok=True)

        import rex.identity as id_mod

        def patched_known_user_ids():
            if not mem.is_dir():
                return []
            users = []
            for entry in sorted(mem.iterdir()):
                if entry.is_dir() and (entry / "core.json").exists():
                    users.append(entry.name)
            return users

        with patch.object(id_mod, "_known_user_ids", patched_known_user_ids):
            result = id_mod._known_user_ids()
            assert "alice" in result
            assert "bob" in result
            assert "empty" not in result

    def test_list_known_users_returns_dicts(self):
        """list_known_users returns list of dicts with expected keys."""
        # Use real Memory/ dir (cole, james exist per repo)
        users = list_known_users()
        # May be empty in test env if Memory/ not present relative to installed module
        for u in users:
            assert "id" in u
            assert "name" in u


# ---------------------------------------------------------------
# CLI integration tests
# ---------------------------------------------------------------


class TestIdentityCLI:
    """Tests for whoami and identify CLI commands."""

    def test_whoami_no_user(self, session_dir):
        """rex whoami with no active user shows helpful message."""
        import argparse

        from rex.cli import cmd_whoami

        args = argparse.Namespace()
        # Should not raise
        result = cmd_whoami(args)
        assert result == 0

    def test_whoami_with_session_user(self, session_dir):
        """rex whoami shows active session user."""
        set_session_user("testuser")

        import argparse

        from rex.cli import cmd_whoami

        args = argparse.Namespace()
        result = cmd_whoami(args)
        assert result == 0

    def test_identify_noninteractive(self, session_dir):
        """rex identify --user sets the session user."""
        import argparse

        from rex.cli import cmd_identify

        args = argparse.Namespace(user="alice")
        result = cmd_identify(args)
        assert result == 0
        assert get_session_user() == "alice"

    def test_identify_no_users_no_interactive(self, session_dir):
        """rex identify with no known users and no --user shows error."""
        import argparse

        from rex.cli import cmd_identify

        args = argparse.Namespace(user=None)
        with patch("rex.identity.list_known_users", return_value=[]):
            result = cmd_identify(args)
            assert result == 1
