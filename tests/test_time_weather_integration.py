"""Integration tests for time, date, and weather accuracy (US-012)."""

from __future__ import annotations

import re
from unittest.mock import patch

from rex.openclaw.tool_executor import execute_tool

# ---------------------------------------------------------------------------
# time_now – London
# ---------------------------------------------------------------------------


def test_time_now_london_returns_europe_london_timezone():
    result = execute_tool({"tool": "time_now", "args": {"location": "London"}}, {})
    assert "error" not in result, f"Unexpected error: {result}"
    assert result["timezone"] == "Europe/London"
    assert re.match(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}", result["local_time"])
    assert re.match(r"\d{4}-\d{2}-\d{2}", result["date"])


# ---------------------------------------------------------------------------
# time_now – Dallas direct location variants
# ---------------------------------------------------------------------------


def test_time_now_dallas_texas_returns_america_chicago():
    """time_now with 'Dallas, Texas' resolves to America/Chicago."""
    result = execute_tool({"tool": "time_now", "args": {"location": "Dallas, Texas"}}, {})
    assert "error" not in result, f"Unexpected error: {result}"
    assert result["timezone"] == "America/Chicago"
    assert re.match(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}", result["local_time"])


def test_time_now_dallas_local_time_differs_from_utc():
    """Dallas local time should differ from UTC by the CDT/CST offset."""
    from datetime import datetime
    from datetime import timezone as _utc_tz

    result = execute_tool({"tool": "time_now", "args": {"location": "Dallas"}}, {})
    assert "error" not in result, f"Unexpected error: {result}"
    assert result["timezone"] == "America/Chicago"

    # Verify the tool actually returns a different hour than UTC
    utc_now = datetime.now(tz=_utc_tz.utc)
    dallas_hour = int(result["local_time"].split(" ")[1].split(":")[0])
    # CDT = UTC-5, CST = UTC-6 — at least one of date or hour must differ
    assert dallas_hour != utc_now.hour or result["date"] != utc_now.strftime(
        "%Y-%m-%d"
    ), "Dallas time should differ from UTC"
    assert result["timezone"] != "UTC", "Dallas must not fall back to UTC"


# ---------------------------------------------------------------------------
# time_now – Dallas context fallback -> America/Chicago
# ---------------------------------------------------------------------------


def test_time_now_no_location_arg_falls_back_to_context_location():
    """When no location arg is given, default_context["location"] is used."""
    from rex import geolocation

    geolocation.clear_cache()
    result = execute_tool(
        {"tool": "time_now", "args": {}},
        {"location": "Dallas"},
    )
    assert "error" not in result, f"Unexpected error: {result}"
    assert result["timezone"] == "America/Chicago"
    assert re.match(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}", result["local_time"])


# ---------------------------------------------------------------------------
# weather_now – New York (mocked API)
# ---------------------------------------------------------------------------


_MOCK_OWM_RESPONSE = {
    "name": "New York",
    "main": {"temp": 20.0, "humidity": 55},
    "weather": [{"description": "clear sky"}],
    "wind": {"speed": 3.5},
}


def test_weather_now_new_york_returns_temperature_and_description(monkeypatch):
    """weather_now returns temp and description for New York using mocked OWM API."""
    monkeypatch.setenv("OPENWEATHERMAP_API_KEY", "test_key")

    with patch("rex.weather._fetch_weather", return_value=_MOCK_OWM_RESPONSE):
        result = execute_tool(
            {"tool": "weather_now", "args": {"location": "New York"}},
            {},
            skip_credential_check=True,
        )

    assert "error" not in result, f"Unexpected error: {result}"
    assert "temp_f" in result
    assert "temp_c" in result
    assert "description" in result
    assert isinstance(result["temp_f"], float)
    assert isinstance(result["description"], str)
    assert result["description"] == "clear sky"
    assert result["city"] == "New York"


# ---------------------------------------------------------------------------
# System prompt contains current date
# ---------------------------------------------------------------------------


def test_system_prompt_contains_current_date(monkeypatch, tmp_path):
    """_build_system_context() includes the current date in YYYY-MM-DD format."""
    import rex.assistant as assistant_module

    class DummyLanguageModel:
        def __init__(self, *args, **kwargs):
            pass

        def generate(self, prompt, config=None):
            return "ok"

    monkeypatch.setattr(assistant_module, "LanguageModel", DummyLanguageModel)

    assistant = assistant_module.Assistant(transcripts_dir=tmp_path)
    context = assistant._build_system_context()

    assert "Current date and time:" in context
    # Context contains a date in YYYY-MM-DD format
    assert re.search(
        r"\d{4}-\d{2}-\d{2}", context
    ), f"Expected YYYY-MM-DD date in system context, got: {context!r}"
