"""Tests for rex.openclaw.tools.plex_tool — US-P5-017 and US-P5-018.

US-P5-017 acceptance criteria:
  - plex_search, plex_play, plex_pause, plex_stop exist and are callable
  - Each returns a dict with at least {"ok": bool}
  - ToolBridge.register_plex_tools() exists and returns a dict keyed by tool names
  - register() returns None (or dict of Nones) when openclaw not installed

US-P5-018 acceptance criteria:
  - plex_search delegates to PlexClient.search()
  - plex_play delegates to PlexClient.play()
  - plex_pause delegates to PlexClient.pause()
  - plex_stop delegates to PlexClient.stop()
  - No-client guard: returns ok=False when client is None
  - Not-enabled guard: returns ok=False when client.enabled is False
  - Failure path: returns ok=False when underlying call returns False
  - Exception path: returns ok=False with error string
  - ToolBridge.register_plex_tools() wires through to plex_tool.register()
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import rex.plex_client as _plex_module
from rex.openclaw.tool_bridge import ToolBridge
from rex.openclaw.tools.plex_tool import (
    TOOL_NAMES,
    plex_pause,
    plex_play,
    plex_search,
    plex_stop,
    register,
)
from rex.plex_client import PlexMediaItem


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_client(*, enabled: bool = True) -> MagicMock:
    client = MagicMock()
    client.enabled = enabled
    return client


def _make_media_item(
    rating_key: str = "42",
    title: str = "Breaking Bad",
    media_type: str = "show",
    year: int | None = 2008,
    summary: str = "A chemistry teacher...",
    duration_ms: int | None = 3600000,
) -> PlexMediaItem:
    return PlexMediaItem(
        rating_key=rating_key,
        title=title,
        media_type=media_type,
        year=year,
        summary=summary,
        duration_ms=duration_ms,
        metadata={},
    )


# ---------------------------------------------------------------------------
# US-P5-017: Tool structure / registration
# ---------------------------------------------------------------------------


class TestToolStructure:
    def test_plex_search_callable(self):
        assert callable(plex_search)

    def test_plex_play_callable(self):
        assert callable(plex_play)

    def test_plex_pause_callable(self):
        assert callable(plex_pause)

    def test_plex_stop_callable(self):
        assert callable(plex_stop)

    def test_tool_names_tuple(self):
        assert isinstance(TOOL_NAMES, tuple)
        assert set(TOOL_NAMES) == {"plex_search", "plex_play", "plex_pause", "plex_stop"}

    def test_register_returns_dict(self):
        result = register()
        assert isinstance(result, dict)

    def test_register_keys_match_tool_names(self):
        result = register()
        assert set(result.keys()) == set(TOOL_NAMES)

    def test_register_values_none_without_openclaw(self):
        result = register()
        # openclaw is not installed in test env
        assert all(v is None for v in result.values())

    def test_tool_bridge_has_register_plex_tools(self):
        bridge = ToolBridge()
        assert callable(bridge.register_plex_tools)

    def test_tool_bridge_register_plex_tools_returns_dict(self):
        bridge = ToolBridge()
        result = bridge.register_plex_tools()
        assert isinstance(result, dict)
        assert set(result.keys()) == set(TOOL_NAMES)


# ---------------------------------------------------------------------------
# US-P5-018: plex_search delegation
# ---------------------------------------------------------------------------


class TestPlexSearch:
    def test_search_delegates_to_client(self):
        item = _make_media_item()
        client = _make_client()
        client.search.return_value = [item]

        with patch.object(_plex_module, "_client", client):
            result = plex_search("Breaking Bad")

        client.search.assert_called_once_with("Breaking Bad", limit=20)
        assert result["ok"] is True
        assert len(result["results"]) == 1
        assert result["results"][0]["rating_key"] == "42"
        assert result["results"][0]["title"] == "Breaking Bad"

    def test_search_custom_limit(self):
        client = _make_client()
        client.search.return_value = []

        with patch.object(_plex_module, "_client", client):
            plex_search("Inception", limit=5)

        client.search.assert_called_once_with("Inception", limit=5)

    def test_search_maps_all_fields(self):
        item = _make_media_item(
            rating_key="99",
            title="Inception",
            media_type="movie",
            year=2010,
            summary="Dreams within dreams",
            duration_ms=8880000,
        )
        client = _make_client()
        client.search.return_value = [item]

        with patch.object(_plex_module, "_client", client):
            result = plex_search("Inception")

        r = result["results"][0]
        assert r["rating_key"] == "99"
        assert r["title"] == "Inception"
        assert r["media_type"] == "movie"
        assert r["year"] == 2010
        assert r["summary"] == "Dreams within dreams"
        assert r["duration_ms"] == 8880000

    def test_search_returns_empty_list_on_no_results(self):
        client = _make_client()
        client.search.return_value = []

        with patch.object(_plex_module, "_client", client):
            result = plex_search("nothing")

        assert result["ok"] is True
        assert result["results"] == []

    def test_search_no_client_guard(self):
        with patch.object(_plex_module, "_client", None):
            result = plex_search("anything")

        assert result["ok"] is False
        assert "not configured" in result["error"].lower()

    def test_search_not_enabled_guard(self):
        client = _make_client(enabled=False)
        with patch.object(_plex_module, "_client", client):
            result = plex_search("anything")

        assert result["ok"] is False
        assert "not enabled" in result["error"].lower()

    def test_search_exception_returns_error(self):
        client = _make_client()
        client.search.side_effect = RuntimeError("network timeout")

        with patch.object(_plex_module, "_client", client):
            result = plex_search("anything")

        assert result["ok"] is False
        assert "network timeout" in result["error"]


# ---------------------------------------------------------------------------
# US-P5-018: plex_play delegation
# ---------------------------------------------------------------------------


class TestPlexPlay:
    def test_play_delegates_to_client(self):
        client = _make_client()
        client.play.return_value = True

        with patch.object(_plex_module, "_client", client):
            result = plex_play("my-client", rating_key="42")

        client.play.assert_called_once_with("my-client", rating_key="42")
        assert result["ok"] is True
        assert result["error"] is None

    def test_play_without_rating_key(self):
        client = _make_client()
        client.play.return_value = True

        with patch.object(_plex_module, "_client", client):
            plex_play("my-client")

        client.play.assert_called_once_with("my-client", rating_key=None)

    def test_play_failure_returns_ok_false(self):
        client = _make_client()
        client.play.return_value = False

        with patch.object(_plex_module, "_client", client):
            result = plex_play("my-client")

        assert result["ok"] is False
        assert result["error"] is not None

    def test_play_no_client_guard(self):
        with patch.object(_plex_module, "_client", None):
            result = plex_play("my-client")

        assert result["ok"] is False

    def test_play_not_enabled_guard(self):
        client = _make_client(enabled=False)
        with patch.object(_plex_module, "_client", client):
            result = plex_play("my-client")

        assert result["ok"] is False

    def test_play_exception_returns_error(self):
        client = _make_client()
        client.play.side_effect = ConnectionError("refused")

        with patch.object(_plex_module, "_client", client):
            result = plex_play("my-client")

        assert result["ok"] is False
        assert "refused" in result["error"]


# ---------------------------------------------------------------------------
# US-P5-018: plex_pause delegation
# ---------------------------------------------------------------------------


class TestPlexPause:
    def test_pause_delegates_to_client(self):
        client = _make_client()
        client.pause.return_value = True

        with patch.object(_plex_module, "_client", client):
            result = plex_pause("my-client")

        client.pause.assert_called_once_with("my-client")
        assert result["ok"] is True

    def test_pause_failure_returns_ok_false(self):
        client = _make_client()
        client.pause.return_value = False

        with patch.object(_plex_module, "_client", client):
            result = plex_pause("my-client")

        assert result["ok"] is False

    def test_pause_no_client_guard(self):
        with patch.object(_plex_module, "_client", None):
            result = plex_pause("my-client")

        assert result["ok"] is False

    def test_pause_not_enabled_guard(self):
        client = _make_client(enabled=False)
        with patch.object(_plex_module, "_client", client):
            result = plex_pause("my-client")

        assert result["ok"] is False

    def test_pause_exception_returns_error(self):
        client = _make_client()
        client.pause.side_effect = RuntimeError("timeout")

        with patch.object(_plex_module, "_client", client):
            result = plex_pause("my-client")

        assert result["ok"] is False
        assert "timeout" in result["error"]


# ---------------------------------------------------------------------------
# US-P5-018: plex_stop delegation
# ---------------------------------------------------------------------------


class TestPlexStop:
    def test_stop_delegates_to_client(self):
        client = _make_client()
        client.stop.return_value = True

        with patch.object(_plex_module, "_client", client):
            result = plex_stop("my-client")

        client.stop.assert_called_once_with("my-client")
        assert result["ok"] is True

    def test_stop_failure_returns_ok_false(self):
        client = _make_client()
        client.stop.return_value = False

        with patch.object(_plex_module, "_client", client):
            result = plex_stop("my-client")

        assert result["ok"] is False

    def test_stop_no_client_guard(self):
        with patch.object(_plex_module, "_client", None):
            result = plex_stop("my-client")

        assert result["ok"] is False

    def test_stop_not_enabled_guard(self):
        client = _make_client(enabled=False)
        with patch.object(_plex_module, "_client", client):
            result = plex_stop("my-client")

        assert result["ok"] is False

    def test_stop_exception_returns_error(self):
        client = _make_client()
        client.stop.side_effect = RuntimeError("disconnect")

        with patch.object(_plex_module, "_client", client):
            result = plex_stop("my-client")

        assert result["ok"] is False
        assert "disconnect" in result["error"]


# ---------------------------------------------------------------------------
# US-P5-017: ToolBridge wiring
# ---------------------------------------------------------------------------


class TestToolBridgeWiring:
    def test_register_plex_tools_delegates_to_plex_register(self):
        bridge = ToolBridge()
        fake_handles = {name: MagicMock() for name in TOOL_NAMES}

        with patch("rex.openclaw.tool_bridge._register_plex_tools", return_value=fake_handles) as mock_reg:
            result = bridge.register_plex_tools(agent=None)

        mock_reg.assert_called_once_with(agent=None)
        assert result is fake_handles

    def test_register_plex_tools_passes_agent(self):
        bridge = ToolBridge()
        agent = MagicMock()

        with patch("rex.openclaw.tool_bridge._register_plex_tools", return_value={}) as mock_reg:
            bridge.register_plex_tools(agent=agent)

        mock_reg.assert_called_once_with(agent=agent)
