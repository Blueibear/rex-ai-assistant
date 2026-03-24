"""Unit tests for rex/weather.py."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import MagicMock, patch

import pytest

from rex.weather import get_weather


def _make_mock_response(payload: dict) -> MagicMock:
    """Build a context-manager-shaped mock for urllib.request.urlopen."""
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(payload).encode()
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


_OWM_SUCCESS = {
    "name": "Dallas",
    "main": {"temp": 25.0, "humidity": 55},
    "weather": [{"description": "clear sky"}],
    "wind": {"speed": 5.0},
}


def _run(coro):
    return asyncio.run(coro)


def test_get_weather_success():
    with patch("urllib.request.urlopen", return_value=_make_mock_response(_OWM_SUCCESS)):
        result = _run(get_weather("Dallas", "fake_key"))

    assert result["city"] == "Dallas"
    assert result["temp_c"] == 25.0
    assert result["temp_f"] == pytest.approx(77.0, abs=0.2)
    assert result["description"] == "clear sky"
    assert result["humidity"] == 55
    assert result["wind_mph"] == pytest.approx(11.2, abs=0.2)


def test_get_weather_missing_location():
    result = _run(get_weather("", "fake_key"))
    assert "error" in result


def test_get_weather_missing_api_key():
    result = _run(get_weather("Dallas", ""))
    assert "error" in result


def test_get_weather_api_http_error():
    import urllib.error

    mock_exc = urllib.error.HTTPError(
        url="http://example.com",
        code=401,
        msg="Unauthorized",
        hdrs=None,  # type: ignore[arg-type]
        fp=None,  # type: ignore[arg-type]
    )
    mock_exc.read = lambda: b'{"cod": 401, "message": "Invalid API key"}'
    with patch("urllib.request.urlopen", side_effect=mock_exc):
        result = _run(get_weather("Dallas", "bad_key"))
    assert "error" in result
    assert "401" in result["error"] or "Invalid" in result["error"]


def test_get_weather_network_error():
    with patch("urllib.request.urlopen", side_effect=OSError("network unreachable")):
        result = _run(get_weather("Dallas", "fake_key"))
    assert "error" in result


def test_get_weather_malformed_response():
    malformed = {"name": "Dallas", "main": {}, "weather": [], "wind": {"speed": 0}}
    with patch("urllib.request.urlopen", return_value=_make_mock_response(malformed)):
        result = _run(get_weather("Dallas", "fake_key"))
    assert "error" in result
