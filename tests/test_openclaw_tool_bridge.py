"""Tests for rex.openclaw.tool_bridge — US-P4-003."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from rex.openclaw.tool_bridge import OPENCLAW_AVAILABLE, ToolBridge


class TestToolBridgeInstantiation:
    def test_import(self):
        from rex.openclaw import tool_bridge  # noqa: F401

    def test_no_args(self):
        bridge = ToolBridge()
        assert bridge is not None

    def test_openclaw_available_is_bool(self):
        assert isinstance(OPENCLAW_AVAILABLE, bool)

    def test_satisfies_protocol(self):
        from rex.contracts.tool_routing import ToolRoutingProtocol

        bridge = ToolBridge()
        assert isinstance(bridge, ToolRoutingProtocol)


class TestParseToolRequest:
    def setup_method(self):
        self.bridge = ToolBridge()

    def test_valid_request_delegated(self):
        line = 'TOOL_REQUEST: {"tool": "time_now", "args": {}}'
        result = self.bridge.parse_tool_request(line)
        assert result == {"tool": "time_now", "args": {}}

    def test_non_tool_line_returns_none(self):
        assert self.bridge.parse_tool_request("Hello world") is None

    def test_empty_string_returns_none(self):
        assert self.bridge.parse_tool_request("") is None

    def test_with_args(self):
        line = 'TOOL_REQUEST: {"tool": "weather_now", "args": {"location": "London"}}'
        result = self.bridge.parse_tool_request(line)
        assert result == {"tool": "weather_now", "args": {"location": "London"}}

    def test_delegates_to_tool_router(self):
        with patch(
            "rex.openclaw.tool_bridge._parse_tool_request", return_value={"tool": "x", "args": {}}
        ) as mock_fn:
            self.bridge.parse_tool_request("TOOL_REQUEST: {}")
            mock_fn.assert_called_once_with("TOOL_REQUEST: {}")


class TestExecuteTool:
    def setup_method(self):
        self.bridge = ToolBridge()

    def test_delegates_to_tool_router(self):
        fake_result = {"status": "ok", "result": "12:00"}
        with patch(
            "rex.openclaw.tool_bridge._execute_tool", return_value=fake_result
        ) as mock_fn:
            req = {"tool": "time_now", "args": {}}
            ctx = {"timezone": "UTC"}
            result = self.bridge.execute_tool(req, ctx)
            mock_fn.assert_called_once_with(
                req,
                ctx,
                skip_policy_check=False,
                skip_credential_check=False,
                task_id=None,
                requested_by=None,
                skip_audit_log=False,
            )
            assert result == fake_result

    def test_skip_policy_check_forwarded(self):
        with patch(
            "rex.openclaw.tool_bridge._execute_tool", return_value={"status": "ok", "result": "x"}
        ) as mock_fn:
            req = {"tool": "time_now", "args": {}}
            self.bridge.execute_tool(req, {}, skip_policy_check=True)
            _, kwargs = mock_fn.call_args
            assert kwargs["skip_policy_check"] is True

    def test_skip_credential_check_forwarded(self):
        with patch(
            "rex.openclaw.tool_bridge._execute_tool", return_value={"status": "ok", "result": "x"}
        ) as mock_fn:
            req = {"tool": "time_now", "args": {}}
            self.bridge.execute_tool(req, {}, skip_credential_check=True)
            _, kwargs = mock_fn.call_args
            assert kwargs["skip_credential_check"] is True

    def test_task_id_forwarded(self):
        with patch(
            "rex.openclaw.tool_bridge._execute_tool", return_value={"status": "ok", "result": "x"}
        ) as mock_fn:
            req = {"tool": "time_now", "args": {}}
            self.bridge.execute_tool(req, {}, task_id="abc-123")
            _, kwargs = mock_fn.call_args
            assert kwargs["task_id"] == "abc-123"

    def test_requested_by_forwarded(self):
        with patch(
            "rex.openclaw.tool_bridge._execute_tool", return_value={"status": "ok", "result": "x"}
        ) as mock_fn:
            req = {"tool": "time_now", "args": {}}
            self.bridge.execute_tool(req, {}, requested_by="user-42")
            _, kwargs = mock_fn.call_args
            assert kwargs["requested_by"] == "user-42"

    def test_skip_audit_log_forwarded(self):
        with patch(
            "rex.openclaw.tool_bridge._execute_tool", return_value={"status": "ok", "result": "x"}
        ) as mock_fn:
            req = {"tool": "time_now", "args": {}}
            self.bridge.execute_tool(req, {}, skip_audit_log=True)
            _, kwargs = mock_fn.call_args
            assert kwargs["skip_audit_log"] is True


class TestRouteIfToolRequest:
    def setup_method(self):
        self.bridge = ToolBridge()

    def test_non_tool_text_returned_unchanged(self):
        result = self.bridge.route_if_tool_request("Hello world", {}, lambda msg: "reply")
        assert result == "Hello world"

    def test_delegates_to_tool_router(self):
        model_fn = MagicMock(return_value="It is noon.")
        with patch(
            "rex.openclaw.tool_bridge._route_if_tool_request", return_value="It is noon."
        ) as mock_fn:
            llm_text = 'TOOL_REQUEST: {"tool": "time_now", "args": {}}'
            result = self.bridge.route_if_tool_request(llm_text, {}, model_fn)
            mock_fn.assert_called_once_with(llm_text, {}, model_fn, skip_policy_check=False)
            assert result == "It is noon."

    def test_skip_policy_check_forwarded(self):
        model_fn = MagicMock(return_value="done")
        with patch(
            "rex.openclaw.tool_bridge._route_if_tool_request", return_value="done"
        ) as mock_fn:
            self.bridge.route_if_tool_request("text", {}, model_fn, skip_policy_check=True)
            _, kwargs = mock_fn.call_args
            assert kwargs["skip_policy_check"] is True


class TestRegister:
    def test_register_returns_none_without_openclaw(self):
        bridge = ToolBridge()
        # If openclaw is not installed, register() should return None
        if not OPENCLAW_AVAILABLE:
            assert bridge.register() is None

    def test_register_accepts_agent_arg(self):
        bridge = ToolBridge()
        agent = MagicMock()
        if not OPENCLAW_AVAILABLE:
            assert bridge.register(agent=agent) is None
