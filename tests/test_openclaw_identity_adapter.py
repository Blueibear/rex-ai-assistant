"""Tests for rex.openclaw.identity_adapter — US-P3-014.

Verifies that IdentityAdapter correctly bridges Rex's identity system:
- set_user / get_user propagation through session state
- resolve_user priority chain
- build_session produces the expected keys and reflects set_user
- clear_user resets session state
- register() returns None when openclaw is not installed
"""

from __future__ import annotations

from typing import Any
from unittest.mock import call, patch

import pytest

from rex.openclaw.identity_adapter import IdentityAdapter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _adapter(**kwargs: Any) -> IdentityAdapter:
    """Return a fresh IdentityAdapter with no shared state."""
    return IdentityAdapter(**kwargs)


# ---------------------------------------------------------------------------
# Instantiation
# ---------------------------------------------------------------------------


class TestIdentityAdapterInstantiation:
    def test_imports_without_error(self) -> None:
        from rex.openclaw import identity_adapter  # noqa: F401

    def test_instantiates_with_no_args(self) -> None:
        adapter = IdentityAdapter()
        assert adapter is not None

    def test_instantiates_with_config_dict(self) -> None:
        adapter = IdentityAdapter(config={"runtime": {"active_user": "alice"}})
        assert adapter is not None

    def test_openclaw_available_flag_is_bool(self) -> None:
        from rex.openclaw.identity_adapter import OPENCLAW_AVAILABLE

        assert isinstance(OPENCLAW_AVAILABLE, bool)


# ---------------------------------------------------------------------------
# set_user / get_user
# ---------------------------------------------------------------------------


class TestSetAndGetUser:
    def test_set_user_then_get_user_returns_same_id(self) -> None:
        adapter = _adapter()
        with patch("rex.openclaw.identity_adapter.set_session_user") as mock_set:
            with patch(
                "rex.openclaw.identity_adapter.get_session_user", return_value="james"
            ):
                adapter.set_user("james")
                result = adapter.get_user()

        mock_set.assert_called_once_with("james")
        assert result == "james"

    def test_get_user_returns_none_when_no_session(self) -> None:
        adapter = _adapter()
        with patch(
            "rex.openclaw.identity_adapter.get_session_user", return_value=None
        ):
            result = adapter.get_user()

        assert result is None

    def test_set_user_delegates_to_set_session_user(self) -> None:
        adapter = _adapter()
        with patch("rex.openclaw.identity_adapter.set_session_user") as mock_set:
            adapter.set_user("alice")

        mock_set.assert_called_once_with("alice")

    def test_get_user_delegates_to_get_session_user(self) -> None:
        adapter = _adapter()
        with patch(
            "rex.openclaw.identity_adapter.get_session_user", return_value="bob"
        ) as mock_get:
            result = adapter.get_user()

        mock_get.assert_called_once()
        assert result == "bob"


# ---------------------------------------------------------------------------
# clear_user
# ---------------------------------------------------------------------------


class TestClearUser:
    def test_clear_user_delegates_to_clear_session_user(self) -> None:
        adapter = _adapter()
        with patch(
            "rex.openclaw.identity_adapter.clear_session_user"
        ) as mock_clear:
            adapter.clear_user()

        mock_clear.assert_called_once()

    def test_clear_user_then_get_user_returns_none(self) -> None:
        adapter = _adapter()
        with patch("rex.openclaw.identity_adapter.clear_session_user"):
            with patch(
                "rex.openclaw.identity_adapter.get_session_user", return_value=None
            ):
                adapter.clear_user()
                result = adapter.get_user()

        assert result is None


# ---------------------------------------------------------------------------
# resolve_user
# ---------------------------------------------------------------------------


class TestResolveUser:
    def test_resolve_user_explicit_takes_priority(self) -> None:
        adapter = _adapter()
        with patch(
            "rex.openclaw.identity_adapter.resolve_active_user", return_value="explicit_user"
        ) as mock_resolve:
            result = adapter.resolve_user(explicit_user="explicit_user")

        mock_resolve.assert_called_once_with("explicit_user", config={})
        assert result == "explicit_user"

    def test_resolve_user_no_explicit_uses_session(self) -> None:
        adapter = _adapter()
        with patch(
            "rex.openclaw.identity_adapter.resolve_active_user", return_value="session_user"
        ) as mock_resolve:
            result = adapter.resolve_user()

        mock_resolve.assert_called_once_with(None, config={})
        assert result == "session_user"

    def test_resolve_user_returns_none_when_no_user(self) -> None:
        adapter = _adapter()
        with patch(
            "rex.openclaw.identity_adapter.resolve_active_user", return_value=None
        ):
            result = adapter.resolve_user()

        assert result is None

    def test_resolve_user_passes_config_to_chain(self) -> None:
        cfg = {"runtime": {"active_user": "config_user"}}
        adapter = IdentityAdapter(config=cfg)
        with patch(
            "rex.openclaw.identity_adapter.resolve_active_user", return_value="config_user"
        ) as mock_resolve:
            adapter.resolve_user()

        mock_resolve.assert_called_once_with(None, config=cfg)


# ---------------------------------------------------------------------------
# build_session — propagation from set_user
# ---------------------------------------------------------------------------


class TestBuildSession:
    def test_build_session_contains_required_keys(self) -> None:
        adapter = _adapter()
        with patch("rex.openclaw.identity_adapter.set_session_user"):
            with patch(
                "rex.openclaw.identity_adapter.build_session_context",
                return_value={
                    "user_id": "james",
                    "session_started_at": "2026-03-22T00:00:00+00:00",
                    "rex_known_users": [],
                    "rex_user_profile": None,
                },
            ):
                adapter.set_user("james")
                session = adapter.build_session()

        assert "user_id" in session
        assert "session_started_at" in session
        assert "rex_known_users" in session
        assert "rex_user_profile" in session

    def test_set_user_then_build_session_reflects_user(self) -> None:
        """After set_user, build_session should carry the set user_id."""
        adapter = _adapter()

        with patch("rex.openclaw.identity_adapter.set_session_user"):
            adapter.set_user("james")

        with patch(
            "rex.openclaw.identity_adapter.build_session_context",
            return_value={
                "user_id": "james",
                "session_started_at": "2026-03-22T00:00:00+00:00",
                "rex_known_users": [],
                "rex_user_profile": None,
            },
        ):
            session = adapter.build_session(explicit_user="james")

        assert session["user_id"] == "james"

    def test_build_session_delegates_to_build_session_context(self) -> None:
        adapter = _adapter()
        expected = {
            "user_id": "alice",
            "session_started_at": "2026-03-22T00:00:00+00:00",
            "rex_known_users": [],
            "rex_user_profile": {"name": "Alice"},
        }
        with patch(
            "rex.openclaw.identity_adapter.build_session_context",
            return_value=expected,
        ) as mock_ctx:
            result = adapter.build_session(explicit_user="alice", metadata={"channel": "voice"})

        mock_ctx.assert_called_once_with("alice", config=None, metadata={"channel": "voice"})
        assert result == expected

    def test_build_session_metadata_passed_through(self) -> None:
        adapter = _adapter()
        with patch(
            "rex.openclaw.identity_adapter.build_session_context",
            return_value={
                "user_id": None,
                "session_started_at": "2026-03-22T00:00:00+00:00",
                "rex_known_users": [],
                "rex_user_profile": None,
                "channel": "voice",
            },
        ):
            result = adapter.build_session(metadata={"channel": "voice"})

        assert result.get("channel") == "voice"

    def test_build_session_no_user_returns_none_user_id(self) -> None:
        adapter = _adapter()
        with patch(
            "rex.openclaw.identity_adapter.build_session_context",
            return_value={
                "user_id": None,
                "session_started_at": "2026-03-22T00:00:00+00:00",
                "rex_known_users": [],
                "rex_user_profile": None,
            },
        ):
            result = adapter.build_session()

        assert result["user_id"] is None


# ---------------------------------------------------------------------------
# register
# ---------------------------------------------------------------------------


class TestRegister:
    def test_register_returns_none_when_openclaw_unavailable(self) -> None:
        adapter = _adapter()
        with patch("rex.openclaw.identity_adapter.OPENCLAW_AVAILABLE", False):
            result = adapter.register()

        assert result is None

    def test_register_with_agent_arg_returns_none_when_unavailable(self) -> None:
        adapter = _adapter()
        with patch("rex.openclaw.identity_adapter.OPENCLAW_AVAILABLE", False):
            result = adapter.register(agent=object())

        assert result is None
