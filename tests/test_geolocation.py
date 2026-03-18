"""Unit tests for rex.geolocation module."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import MagicMock, patch

import pytest

import rex.geolocation as geo


def _make_response(payload: dict) -> MagicMock:
    """Build a mock urllib response context manager."""
    encoded = json.dumps(payload).encode("utf-8")
    mock_resp = MagicMock()
    mock_resp.read.return_value = encoded
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


@pytest.fixture(autouse=True)
def clear_geo_cache():
    geo.clear_cache()
    yield
    geo.clear_cache()


def test_detect_location_success():
    """Returns parsed location dict on successful API response."""
    payload = {
        "status": "success",
        "city": "Dallas",
        "timezone": "America/Chicago",
        "lat": 32.7767,
        "lon": -96.797,
    }
    with patch("urllib.request.urlopen", return_value=_make_response(payload)):
        result = asyncio.run(geo.detect_location())

    assert result is not None
    assert result["city"] == "Dallas"
    assert result["timezone"] == "America/Chicago"
    assert abs(result["lat"] - 32.7767) < 0.001
    assert abs(result["lon"] - (-96.797)) < 0.001


def test_detect_location_cached():
    """Second call returns cached result without re-fetching."""
    payload = {
        "status": "success",
        "city": "London",
        "timezone": "Europe/London",
        "lat": 51.5074,
        "lon": -0.1278,
    }
    with patch("urllib.request.urlopen", return_value=_make_response(payload)) as mock_open:
        asyncio.run(geo.detect_location())
        asyncio.run(geo.detect_location())
        assert mock_open.call_count == 1


def test_detect_location_api_failure():
    """Returns None when api returns non-success status."""
    payload = {"status": "fail", "message": "private range"}
    with patch("urllib.request.urlopen", return_value=_make_response(payload)):
        result = asyncio.run(geo.detect_location())
    assert result is None


def test_detect_location_network_error():
    """Returns None on network exception."""
    with patch("urllib.request.urlopen", side_effect=OSError("network down")):
        result = asyncio.run(geo.detect_location())
    assert result is None


def test_detect_location_timeout():
    """Returns None when the request times out."""
    import time

    def slow_fetch():
        time.sleep(10)

    with patch.object(geo, "_fetch_location", side_effect=lambda: time.sleep(10)):
        with patch("asyncio.wait_for", side_effect=asyncio.TimeoutError):
            result = asyncio.run(geo.detect_location())
    assert result is None
