"""Tests for US-026: Add device discovery via Home Assistant API."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

import rex.ha.discovery as disc

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FAKE_STATES = [
    {
        "entity_id": "light.living_room",
        "state": "on",
        "attributes": {"friendly_name": "Living Room Light"},
    },
    {
        "entity_id": "switch.kitchen_fan",
        "state": "off",
        "attributes": {"friendly_name": "Kitchen Fan"},
    },
    {
        "entity_id": "sensor.temperature",
        "state": "21.5",
        "attributes": {},
    },
]


def _mock_urlopen(raw_list: list[dict]):
    """Return a context-manager mock whose .read() yields *raw_list* as JSON."""
    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=cm)
    cm.__exit__ = MagicMock(return_value=False)
    cm.read = MagicMock(return_value=json.dumps(raw_list).encode())
    return cm


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_cache():
    """Clear module cache and restore TTL before/after each test."""
    disc.clear_cache()
    original_ttl = disc._cache_ttl
    yield
    disc.clear_cache()
    disc._cache_ttl = original_ttl


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_discover_returns_entity_list():
    """discover_devices() returns correct list shape when HA responds."""
    urlopen_mock = _mock_urlopen(_FAKE_STATES)
    with patch("urllib.request.urlopen", return_value=urlopen_mock):
        results = disc.discover_devices(
            base_url="http://ha.local:8123",
            token="tok",
            cache_ttl=0,
        )

    assert len(results) == 3
    light = next(r for r in results if r["entity_id"] == "light.living_room")
    assert light["friendly_name"] == "Living Room Light"
    assert light["domain"] == "light"
    assert light["state"] == "on"


def test_discover_parses_domain_from_entity_id():
    """domain field is derived from the prefix of entity_id."""
    urlopen_mock = _mock_urlopen(_FAKE_STATES)
    with patch("urllib.request.urlopen", return_value=urlopen_mock):
        results = disc.discover_devices(
            base_url="http://ha.local:8123",
            token="tok",
            cache_ttl=0,
        )

    domains = {r["domain"] for r in results}
    assert "light" in domains
    assert "switch" in domains
    assert "sensor" in domains


def test_discover_falls_back_to_entity_id_when_no_friendly_name():
    """friendly_name falls back to entity_id when attribute is absent."""
    urlopen_mock = _mock_urlopen(_FAKE_STATES)
    with patch("urllib.request.urlopen", return_value=urlopen_mock):
        results = disc.discover_devices(
            base_url="http://ha.local:8123",
            token="tok",
            cache_ttl=0,
        )

    sensor = next(r for r in results if r["entity_id"] == "sensor.temperature")
    assert sensor["friendly_name"] == "sensor.temperature"


def test_not_configured_returns_empty_and_warns(caplog):
    """Returns [] and logs a warning when base_url or token is missing."""
    import logging

    with caplog.at_level(logging.WARNING, logger="rex.ha.discovery"):
        result = disc.discover_devices(base_url=None, token=None)

    assert result == []
    assert any("not configured" in msg for msg in caplog.messages)


def test_not_configured_missing_token_returns_empty():
    """Returns [] when token is absent even if base_url is set."""
    result = disc.discover_devices(base_url="http://ha.local:8123", token=None)
    assert result == []


def test_cache_hit_avoids_second_request():
    """Second call within TTL returns cached data without hitting HA."""
    urlopen_mock = _mock_urlopen(_FAKE_STATES)
    with patch("urllib.request.urlopen", return_value=urlopen_mock) as mock_open:
        disc.discover_devices(base_url="http://ha.local:8123", token="tok", cache_ttl=300)
        disc.discover_devices(base_url="http://ha.local:8123", token="tok", cache_ttl=300)

    # urlopen should have been called exactly once (cache hit on second call)
    assert mock_open.call_count == 1


def test_cache_miss_after_clear():
    """After clear_cache() the next call re-fetches from HA."""
    urlopen_mock = _mock_urlopen(_FAKE_STATES)
    with patch("urllib.request.urlopen", return_value=urlopen_mock) as mock_open:
        disc.discover_devices(base_url="http://ha.local:8123", token="tok", cache_ttl=300)
        disc.clear_cache()
        disc.discover_devices(base_url="http://ha.local:8123", token="tok", cache_ttl=300)

    assert mock_open.call_count == 2


def test_http_error_returns_empty_and_logs(caplog):
    """Returns [] and logs an error when urlopen raises."""
    import logging
    import urllib.error

    with patch(
        "urllib.request.urlopen",
        side_effect=urllib.error.URLError("connection refused"),
    ):
        with caplog.at_level(logging.ERROR, logger="rex.ha.discovery"):
            result = disc.discover_devices(
                base_url="http://ha.local:8123",
                token="tok",
                cache_ttl=0,
            )

    assert result == []
    assert any("failed to fetch" in msg for msg in caplog.messages)
