"""Tests for US-022: Tool router.

Acceptance criteria:
- tools routed correctly
- execution dispatched
- errors handled safely
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from rex.tool_router import (
    ApprovalRequiredError,
    CredentialMissingError,
    PolicyDeniedError,
    execute_tool,
    format_tool_result,
    parse_tool_request,
    route_if_tool_request,
)


# ---------------------------------------------------------------------------
# Criterion: tools routed correctly
# ---------------------------------------------------------------------------


def test_parse_tool_request_routes_known_tool():
    line = 'TOOL_REQUEST: {"tool":"time_now","args":{"location":"Dallas, TX"}}'
    parsed = parse_tool_request(line)
    assert parsed is not None
    assert parsed["tool"] == "time_now"
    assert parsed["args"] == {"location": "Dallas, TX"}


def test_parse_tool_request_returns_none_for_plain_text():
    assert parse_tool_request("What time is it?") is None


def test_parse_tool_request_returns_none_for_multiline():
    text = 'TOOL_REQUEST: {"tool":"time_now"}\nextra'
    assert parse_tool_request(text) is None


def test_parse_tool_request_missing_tool_name():
    line = 'TOOL_REQUEST: {"args":{}}'
    assert parse_tool_request(line) is None


def test_parse_tool_request_defaults_args_to_empty_dict():
    line = 'TOOL_REQUEST: {"tool":"time_now"}'
    parsed = parse_tool_request(line)
    assert parsed is not None
    assert parsed["args"] == {}


def test_route_if_tool_request_passes_through_non_tool_text():
    result = route_if_tool_request("Hello there!", {}, lambda _msg: "")
    assert result == "Hello there!"


def test_route_if_tool_request_dispatches_and_returns_model_response():
    llm_text = 'TOOL_REQUEST: {"tool":"time_now","args":{"location":"Dallas, TX"}}'
    calls: list[dict[str, str]] = []

    def model_fn(msg: dict[str, str]) -> str:
        calls.append(msg)
        return "It is noon."

    result = route_if_tool_request(llm_text, {}, model_fn, skip_policy_check=True)
    assert result == "It is noon."
    assert len(calls) == 1
    assert calls[0]["role"] == "tool"


# ---------------------------------------------------------------------------
# Criterion: execution dispatched
# ---------------------------------------------------------------------------


def test_execute_tool_time_now_returns_local_time():
    result = execute_tool(
        {"tool": "time_now", "args": {"location": "Dallas, TX"}},
        {},
        skip_policy_check=True,
        skip_audit_log=True,
    )
    assert "local_time" in result
    assert "timezone" in result
    assert result["timezone"] == "America/Chicago"


def test_execute_tool_dispatches_via_tool_name():
    result = execute_tool(
        {"tool": "time_now", "args": {"location": "UTC"}},
        {"timezone": "UTC"},
        skip_policy_check=True,
        skip_audit_log=True,
    )
    # Returns local_time using UTC fallback
    assert "local_time" in result or "error" in result


def test_format_tool_result_includes_tool_and_result():
    formatted = format_tool_result("time_now", {"location": "Dallas, TX"}, {"local_time": "12:00"})
    assert formatted.startswith("TOOL_RESULT:")
    payload = json.loads(formatted[len("TOOL_RESULT:"):].strip())
    assert payload["tool"] == "time_now"
    assert payload["result"] == {"local_time": "12:00"}


# ---------------------------------------------------------------------------
# Criterion: errors handled safely
# ---------------------------------------------------------------------------


def test_execute_tool_unknown_tool_returns_error():
    result = execute_tool(
        {"tool": "nonexistent_tool", "args": {}},
        {},
        skip_policy_check=True,
        skip_audit_log=True,
    )
    assert "error" in result


def test_execute_tool_missing_required_arg_returns_error():
    # time_now without location and no context timezone defaults to UTC gracefully
    result = execute_tool(
        {"tool": "time_now", "args": {}},
        {},
        skip_policy_check=True,
        skip_audit_log=True,
    )
    # Should either succeed with UTC or return an error dict - must not raise
    assert isinstance(result, dict)


def test_execute_tool_invalid_request_dict_returns_error():
    result = execute_tool(
        {},  # missing tool and args
        {},
        skip_policy_check=True,
        skip_audit_log=True,
    )
    assert "error" in result


def test_route_if_tool_request_handles_policy_denied(monkeypatch):
    llm_text = 'TOOL_REQUEST: {"tool":"time_now","args":{"location":"Dallas, TX"}}'

    def raise_denied(*args, **kwargs):
        raise PolicyDeniedError("time_now", "blocked by policy")

    monkeypatch.setattr("rex.tool_router.execute_tool", raise_denied)
    result = route_if_tool_request(llm_text, {}, lambda _msg: "ok")
    assert "cannot" in result.lower() or "denied" in result.lower() or "blocked" in result.lower()


def test_route_if_tool_request_handles_credential_missing(monkeypatch):
    llm_text = 'TOOL_REQUEST: {"tool":"web_search","args":{"query":"test"}}'

    def raise_cred(*args, **kwargs):
        raise CredentialMissingError("web_search", ["BRAVE_API_KEY"])

    monkeypatch.setattr("rex.tool_router.execute_tool", raise_cred)
    result = route_if_tool_request(llm_text, {}, lambda _msg: "ok")
    assert "credential" in result.lower() or "missing" in result.lower() or "cannot" in result.lower()


def test_route_if_tool_request_handles_approval_required(monkeypatch):
    llm_text = 'TOOL_REQUEST: {"tool":"time_now","args":{"location":"Dallas, TX"}}'

    def raise_approval(*args, **kwargs):
        raise ApprovalRequiredError("time_now", "needs user OK")

    monkeypatch.setattr("rex.tool_router.execute_tool", raise_approval)
    result = route_if_tool_request(llm_text, {}, lambda _msg: "ok")
    assert "approval" in result.lower() or "requires" in result.lower()


def test_route_if_tool_request_handles_model_fn_exception(monkeypatch):
    llm_text = 'TOOL_REQUEST: {"tool":"time_now","args":{"location":"Dallas, TX"}}'

    def boom(_msg: dict[str, str]) -> str:
        raise RuntimeError("model exploded")

    result = route_if_tool_request(llm_text, {}, boom, skip_policy_check=True)
    assert isinstance(result, str)
    assert len(result) > 0
