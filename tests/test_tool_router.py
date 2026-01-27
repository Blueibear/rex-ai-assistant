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
    result = execute_tool({"tool": "weather_now", "args": {"location": "Dallas, TX"}}, {})
    assert result == {
        "error": {
            "message": "Tool weather_now is not implemented",
            "tool": "weather_now",
            "args": {"location": "Dallas, TX"},
        }
    }


def test_execute_tool_web_search_returns_not_implemented_error_payload():
    result = execute_tool({"tool": "web_search", "args": {"query": "hello"}}, {})
    assert result == {
        "error": {
            "message": "Tool web_search is not implemented",
            "tool": "web_search",
            "args": {"query": "hello"},
        }
    }


def test_route_if_tool_request_returns_tool_result_for_weather_now():
    calls: list[dict[str, str] | None] = []

    def model_call(message: dict[str, str] | None = None) -> str:
        calls.append(message)
        if message is None:
            return 'TOOL_REQUEST: {"tool":"weather_now","args":{"location":"Dallas, TX"}}'
        return "weather response"

    result = route_if_tool_request(model_call(), {}, model_call)

    assert result == "weather response"
    payload = json.loads(calls[1]["content"].split("TOOL_RESULT: ", 1)[1])
    assert payload["tool"] == "weather_now"
    assert payload["result"]["error"]["message"] == "Tool weather_now is not implemented"


def test_route_if_tool_request_returns_tool_result_for_web_search():
    calls: list[dict[str, str] | None] = []

    def model_call(message: dict[str, str] | None = None) -> str:
        calls.append(message)
        if message is None:
            return 'TOOL_REQUEST: {"tool":"web_search","args":{"query":"hello"}}'
        return "search response"

    result = route_if_tool_request(model_call(), {}, model_call)

    assert result == "search response"
    payload = json.loads(calls[1]["content"].split("TOOL_RESULT: ", 1)[1])
    assert payload["tool"] == "web_search"
    assert payload["result"]["error"]["message"] == "Tool web_search is not implemented"


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
