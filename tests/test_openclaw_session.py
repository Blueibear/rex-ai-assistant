"""Tests for rex.openclaw.session — OpenClaw session bridge."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from rex.openclaw.session import build_session_context


def test_build_session_context_returns_required_keys():
    """build_session_context always returns the required key set."""
    with patch("rex.openclaw.session.resolve_active_user", return_value=None):
        with patch("rex.openclaw.session.list_known_users", return_value=[]):
            result = build_session_context()

    assert "user_id" in result
    assert "session_started_at" in result
    assert "rex_known_users" in result
    assert "rex_user_profile" in result


def test_build_session_context_explicit_user():
    """Explicit user_id is reflected in the session context."""
    with patch("rex.openclaw.session.resolve_active_user", return_value="alice") as mock_resolve:
        with patch("rex.openclaw.session.list_known_users", return_value=[{"id": "alice"}]):
            with patch("rex.openclaw.session.get_user_profile", return_value={"name": "Alice"}):
                result = build_session_context(explicit_user="alice")

    assert result["user_id"] == "alice"
    # resolve_active_user was called with the explicit arg
    mock_resolve.assert_called_once_with("alice", config=None)


def test_build_session_context_no_user():
    """When no user is resolved, user_id is None and profile is None."""
    with patch("rex.openclaw.session.resolve_active_user", return_value=None):
        with patch("rex.openclaw.session.list_known_users", return_value=[]):
            result = build_session_context()

    assert result["user_id"] is None
    assert result["rex_user_profile"] is None


def test_build_session_context_metadata_merged():
    """Extra metadata keys are included in the session dict."""
    with patch("rex.openclaw.session.resolve_active_user", return_value=None):
        with patch("rex.openclaw.session.list_known_users", return_value=[]):
            result = build_session_context(metadata={"channel": "voice", "request_id": "abc"})

    assert result["channel"] == "voice"
    assert result["request_id"] == "abc"


def test_build_session_context_timestamp_is_iso8601():
    """session_started_at is an ISO 8601 UTC timestamp string."""
    with patch("rex.openclaw.session.resolve_active_user", return_value=None):
        with patch("rex.openclaw.session.list_known_users", return_value=[]):
            result = build_session_context()

    ts = result["session_started_at"]
    assert isinstance(ts, str)
    # Must contain a UTC offset indicator (Z or +00:00)
    assert "+" in ts or ts.endswith("Z")


def test_build_session_context_known_users_list():
    """rex_known_users contains the list from list_known_users."""
    known = [{"id": "alice", "name": "Alice", "role": ""}, {"id": "bob", "name": "Bob", "role": ""}]
    with patch("rex.openclaw.session.resolve_active_user", return_value=None):
        with patch("rex.openclaw.session.list_known_users", return_value=known):
            result = build_session_context()

    assert result["rex_known_users"] == known


def test_build_session_context_profile_load_failure_does_not_raise():
    """If get_user_profile raises, rex_user_profile is None (not an exception)."""
    with patch("rex.openclaw.session.resolve_active_user", return_value="broken"):
        with patch("rex.openclaw.session.list_known_users", return_value=[]):
            with patch(
                "rex.openclaw.session.get_user_profile", side_effect=RuntimeError("disk error")
            ):
                result = build_session_context()

    assert result["user_id"] == "broken"
    assert result["rex_user_profile"] is None
