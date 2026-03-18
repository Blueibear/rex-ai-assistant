from __future__ import annotations

import json
import re

from rex.tool_router import (
    execute_tool,
    format_tool_result,
    parse_tool_request,
    route_if_tool_request,
)


def test_parse_tool_request_returns_none_for_normal_text():
    assert parse_tool_request("hello") is None


def test_parse_tool_request_parses_valid_request():
    payload = 'TOOL_REQUEST: {"tool":"time_now","args":{"location":"Dallas, TX"}}'
    assert parse_tool_request(payload) == {
        "tool": "time_now",
        "args": {"location": "Dallas, TX"},
    }


def test_execute_tool_time_now_uses_chicago_timezone_for_dallas():
    result = execute_tool({"tool": "time_now", "args": {"location": "Dallas, TX"}}, {})
    assert "local_time" in result
    assert "timezone" in result
    assert result["timezone"] == "America/Chicago"
    assert re.match(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}", result["local_time"])


def test_route_if_tool_request_calls_model_twice_on_tool_request():
    calls: list[dict[str, str] | None] = []

    def model_call(message: dict[str, str] | None = None) -> str:
        calls.append(message)
        if message is None:
            return 'TOOL_REQUEST: {"tool":"time_now","args":{"location":"Dallas, TX"}}'
        return "final response"

    llm_text = model_call()
    result = route_if_tool_request(llm_text, {}, model_call)

    assert result == "final response"
    assert len(calls) == 2
    tool_message = calls[1]
    assert isinstance(tool_message, dict)
    assert tool_message["role"] == "tool"
    assert tool_message["content"].startswith("TOOL_RESULT: ")
    payload = json.loads(tool_message["content"].split("TOOL_RESULT: ", 1)[1])
    assert payload["tool"] == "time_now"
    assert payload["result"]["timezone"] == "America/Chicago"


def test_route_if_tool_request_does_not_call_model_for_normal_text():
    calls: list[dict[str, str]] = []

    def model_call(message: dict[str, str]) -> str:
        calls.append(message)
        return "unused"

    assert route_if_tool_request("hello", {}, model_call) == "hello"
    assert calls == []


def test_execute_tool_unknown_tool_returns_error():
    # Skip policy check to test the tool routing logic directly
    result = execute_tool({"tool": "unknown", "args": {}}, {}, skip_policy_check=True)
    assert "error" in result


def test_execute_tool_weather_now_returns_not_implemented_error_payload():
    # Skip credential check; no API key configured so should return config error
    result = execute_tool(
        {"tool": "weather_now", "args": {"location": "Dallas, TX"}},
        {},
        skip_credential_check=True,
    )
    assert "error" in result
    assert "OPENWEATHERMAP_API_KEY" in result["error"]["message"]


def test_execute_tool_web_search_returns_no_results_when_unconfigured():
    # Skip credential check; no search providers configured so returns no-results
    result = execute_tool(
        {"tool": "web_search", "args": {"query": "hello"}},
        {},
        skip_credential_check=True,
    )
    # web_search now executes via plugins; with no providers configured it returns a no-results payload
    assert "query" in result
    assert result["query"] == "hello"
    assert "result" in result


def test_route_if_tool_request_returns_credential_error_for_weather_now():
    """Test that weather_now returns credential missing message when not configured."""
    calls: list[dict[str, str] | None] = []

    def model_call(message: dict[str, str] | None = None) -> str:
        calls.append(message)
        if message is None:
            return 'TOOL_REQUEST: {"tool":"weather_now","args":{"location":"Dallas, TX"}}'
        return "weather response"

    result = route_if_tool_request(model_call(), {}, model_call)

    # Should return credential error since weather_now requires openweathermap credential
    assert "Missing credentials" in result
    assert "openweathermap" in result


def test_route_if_tool_request_web_search_executes_without_credentials():
    """Test that web_search executes without required credentials (no required creds)."""
    calls: list[dict[str, str] | None] = []

    def model_call(message: dict[str, str] | None = None) -> str:
        calls.append(message)
        if message is None:
            return 'TOOL_REQUEST: {"tool":"web_search","args":{"query":"hello"}}'
        return "search response"

    result = route_if_tool_request(model_call(), {}, model_call)

    # web_search has no required credentials so executes; model called with TOOL_RESULT
    assert result == "search response"
    payload = json.loads(calls[1]["content"].split("TOOL_RESULT: ", 1)[1])
    assert payload["tool"] == "web_search"
    # Result contains query field (not an error about "not implemented")
    assert "query" in payload["result"]
    assert payload["result"]["query"] == "hello"


def test_parse_tool_request_rejects_multiline_payload():
    payload = 'TOOL_REQUEST: {"tool":"time_now","args":{"location":"Dallas, TX"}}\n'
    assert parse_tool_request(payload) is None


def test_route_if_tool_request_returns_original_on_multiline_payload():
    payload = 'TOOL_REQUEST: {"tool":"time_now","args":{"location":"Dallas, TX"}}\n'
    assert route_if_tool_request(payload, {}, lambda _: "unused") == payload


def test_parse_tool_request_rejects_invalid_json():
    assert parse_tool_request('TOOL_REQUEST: {"tool":') is None


def test_parse_tool_request_rejects_non_dict_json():
    assert parse_tool_request('TOOL_REQUEST: ["tool"]') is None


def test_parse_tool_request_rejects_missing_tool_name():
    assert parse_tool_request('TOOL_REQUEST: {"args":{}}') is None


def test_parse_tool_request_rejects_non_string_tool_name():
    assert parse_tool_request('TOOL_REQUEST: {"tool":123,"args":{}}') is None


def test_parse_tool_request_rejects_non_dict_args():
    assert parse_tool_request('TOOL_REQUEST: {"tool":"time_now","args":"bad"}') is None


def test_unknown_tool_returns_tool_result_line():
    calls: list[dict[str, str] | None] = []

    def model_call(message: dict[str, str] | None = None) -> str:
        calls.append(message)
        if message is None:
            return 'TOOL_REQUEST: {"tool":"unknown_tool","args":{"value":1}}'
        return "unknown response"

    # Skip policy check to test the tool routing logic directly
    result = route_if_tool_request(model_call(), {}, model_call, skip_policy_check=True)

    assert result == "unknown response"
    payload = json.loads(calls[1]["content"].split("TOOL_RESULT: ", 1)[1])
    assert payload["tool"] == "unknown_tool"
    assert payload["result"]["error"]["message"] == "Unknown tool unknown_tool"


def test_format_tool_result_outputs_single_line():
    result = format_tool_result("time_now", {"location": "Dallas, TX"}, {"local_time": "ok"})
    assert result.startswith("TOOL_RESULT: ")
    assert "\n" not in result


# --- US-003: expanded city coverage and date key ---


def test_execute_tool_time_now_returns_date_key():
    result = execute_tool({"tool": "time_now", "args": {"location": "Dallas, TX"}}, {})
    assert "date" in result
    assert re.match(r"\d{4}-\d{2}-\d{2}", result["date"])


def test_execute_tool_time_now_london():
    result = execute_tool({"tool": "time_now", "args": {"location": "London"}}, {})
    assert result.get("timezone") == "Europe/London"
    assert re.match(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}", result["local_time"])
    assert re.match(r"\d{4}-\d{2}-\d{2}", result["date"])


def test_execute_tool_time_now_tokyo():
    result = execute_tool({"tool": "time_now", "args": {"location": "Tokyo"}}, {})
    assert result.get("timezone") == "Asia/Tokyo"


def test_execute_tool_time_now_sydney():
    result = execute_tool({"tool": "time_now", "args": {"location": "Sydney"}}, {})
    assert result.get("timezone") == "Australia/Sydney"


def test_execute_tool_time_now_new_york():
    result = execute_tool({"tool": "time_now", "args": {"location": "New York"}}, {})
    assert result.get("timezone") == "America/New_York"


def test_execute_tool_time_now_mumbai():
    result = execute_tool({"tool": "time_now", "args": {"location": "Mumbai"}}, {})
    assert result.get("timezone") == "Asia/Kolkata"


def test_resolve_timezone_falls_back_to_default_timezone_from_context():
    result = execute_tool(
        {"tool": "time_now", "args": {"location": "UnknownCityXYZ"}},
        {"timezone": "Europe/Paris"},
    )
    assert result.get("timezone") == "Europe/Paris"


def test_resolve_timezone_falls_back_to_utc_when_no_context_or_cache():
    from rex import geolocation

    geolocation.clear_cache()
    result = execute_tool(
        {"tool": "time_now", "args": {"location": "UnknownCityXYZ"}},
        {},
    )
    assert result.get("timezone") == "UTC"


def test_resolve_timezone_falls_back_to_geolocation_cache():
    from rex import geolocation

    geolocation._location_cache = {
        "city": "Paris",
        "timezone": "Europe/Paris",
        "lat": 48.8,
        "lon": 2.3,
    }
    try:
        result = execute_tool(
            {"tool": "time_now", "args": {"location": "UnknownCityXYZ"}},
            {},
        )
        assert result.get("timezone") == "Europe/Paris"
    finally:
        geolocation.clear_cache()
