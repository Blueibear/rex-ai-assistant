"""Tests for ToolBridge.execute_tool() dual-mode dispatch — US-009."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from rex.openclaw.errors import OpenClawAPIError, OpenClawAuthError, OpenClawConnectionError
from rex.openclaw.tool_bridge import ToolBridge
from rex.openclaw.tool_executor import PolicyDeniedError


def _make_config(use_openclaw_tools: bool = True, gateway_url: str = "http://127.0.0.1:18789"):
    """Return a minimal AppConfig-like mock."""
    cfg = MagicMock()
    cfg.use_openclaw_tools = use_openclaw_tools
    cfg.openclaw_gateway_url = gateway_url
    cfg.openclaw_gateway_timeout = 30
    cfg.openclaw_gateway_max_retries = 3
    cfg.openclaw_gateway_token = "test-token"
    return cfg


def _make_client(post_return: dict | None = None, post_side_effect=None):
    """Return a mock OpenClawClient."""
    client = MagicMock()
    if post_side_effect is not None:
        client.post.side_effect = post_side_effect
    else:
        client.post.return_value = post_return or {"status": "success", "result": "12:00"}
    return client


class TestExecuteToolHTTPPath:
    """execute_tool() dispatches via HTTP when use_openclaw_tools=True and client exists."""

    def test_posts_to_tools_invoke(self):
        cfg = _make_config()
        client = _make_client()
        bridge = ToolBridge(config=cfg)
        req = {"tool": "time_now", "args": {"location": "London"}}
        ctx = {"timezone": "Europe/London", "session_key": "user-42"}

        with patch("rex.openclaw.tool_bridge.get_openclaw_client", return_value=client):
            bridge.execute_tool(req, ctx)

        client.post.assert_called_once()
        call_kwargs = client.post.call_args
        assert call_kwargs.args[0] == "/tools/invoke"
        payload = call_kwargs.kwargs["json"]
        assert payload["tool"] == "time_now"
        assert payload["args"] == {"location": "London"}
        assert payload["sessionKey"] == "user-42"

    def test_session_key_defaults_to_main(self):
        cfg = _make_config()
        client = _make_client()
        bridge = ToolBridge(config=cfg)
        req = {"tool": "time_now", "args": {}}

        with patch("rex.openclaw.tool_bridge.get_openclaw_client", return_value=client):
            bridge.execute_tool(req, {})

        payload = client.post.call_args.kwargs["json"]
        assert payload["sessionKey"] == "main"

    def test_returns_http_response(self):
        cfg = _make_config()
        expected = {"status": "success", "result": {"local_time": "2026-03-24 10:00"}}
        client = _make_client(post_return=expected)
        bridge = ToolBridge(config=cfg)

        with patch("rex.openclaw.tool_bridge.get_openclaw_client", return_value=client):
            result = bridge.execute_tool({"tool": "time_now", "args": {}}, {})

        assert result == expected

    def test_403_raises_policy_denied_error(self):
        cfg = _make_config()
        client = _make_client(post_side_effect=OpenClawAPIError(403, "denied"))
        bridge = ToolBridge(config=cfg)

        with patch("rex.openclaw.tool_bridge.get_openclaw_client", return_value=client):
            with pytest.raises(PolicyDeniedError) as exc_info:
                bridge.execute_tool({"tool": "send_email", "args": {}}, {})

        assert exc_info.value.tool == "send_email"

    def test_404_falls_back_to_local(self):
        cfg = _make_config()
        client = _make_client(post_side_effect=OpenClawAPIError(404, "not found"))
        local_result = {"status": "ok", "result": "local"}
        bridge = ToolBridge(config=cfg)

        with patch("rex.openclaw.tool_bridge.get_openclaw_client", return_value=client):
            with patch(
                "rex.openclaw.tool_bridge._execute_tool", return_value=local_result
            ) as mock_local:
                result = bridge.execute_tool({"tool": "time_now", "args": {}}, {})

        mock_local.assert_called_once()
        assert result == local_result

    def test_5xx_raises_after_retries(self):
        cfg = _make_config()
        client = _make_client(post_side_effect=OpenClawAPIError(500, "server error"))
        bridge = ToolBridge(config=cfg)

        with patch("rex.openclaw.tool_bridge.get_openclaw_client", return_value=client):
            with pytest.raises(OpenClawAPIError) as exc_info:
                bridge.execute_tool({"tool": "time_now", "args": {}}, {})

        assert exc_info.value.status == 500

    def test_connection_error_falls_back_to_local(self):
        cfg = _make_config()
        client = _make_client(
            post_side_effect=OpenClawConnectionError("http://127.0.0.1:18789", Exception("refused"))
        )
        local_result = {"status": "ok", "result": "local"}
        bridge = ToolBridge(config=cfg)

        with patch("rex.openclaw.tool_bridge.get_openclaw_client", return_value=client):
            with patch(
                "rex.openclaw.tool_bridge._execute_tool", return_value=local_result
            ) as mock_local:
                result = bridge.execute_tool({"tool": "time_now", "args": {}}, {})

        mock_local.assert_called_once()
        assert result == local_result

    def test_auth_error_falls_back_to_local(self):
        cfg = _make_config()
        client = _make_client(post_side_effect=OpenClawAuthError())
        local_result = {"status": "ok", "result": "local"}
        bridge = ToolBridge(config=cfg)

        with patch("rex.openclaw.tool_bridge.get_openclaw_client", return_value=client):
            with patch(
                "rex.openclaw.tool_bridge._execute_tool", return_value=local_result
            ) as mock_local:
                result = bridge.execute_tool({"tool": "time_now", "args": {}}, {})

        mock_local.assert_called_once()
        assert result == local_result

    def test_no_http_call_when_flag_false(self):
        cfg = _make_config(use_openclaw_tools=False)
        client = _make_client()
        local_result = {"status": "ok", "result": "local"}
        bridge = ToolBridge(config=cfg)

        with patch("rex.openclaw.tool_bridge.get_openclaw_client", return_value=client):
            with patch(
                "rex.openclaw.tool_bridge._execute_tool", return_value=local_result
            ) as mock_local:
                result = bridge.execute_tool({"tool": "time_now", "args": {}}, {})

        client.post.assert_not_called()
        mock_local.assert_called_once()
        assert result == local_result

    def test_no_http_call_when_client_none(self):
        cfg = _make_config()
        local_result = {"status": "ok", "result": "local"}
        bridge = ToolBridge(config=cfg)

        with patch("rex.openclaw.tool_bridge.get_openclaw_client", return_value=None):
            with patch(
                "rex.openclaw.tool_bridge._execute_tool", return_value=local_result
            ) as mock_local:
                result = bridge.execute_tool({"tool": "time_now", "args": {}}, {})

        mock_local.assert_called_once()
        assert result == local_result


class TestExecuteToolStandaloneMode:
    """Standalone mode: no config, no gateway URL — behaves exactly as before."""

    def test_delegates_to_local_executor(self):
        """Without gateway URL, _execute_tool is called directly."""
        bridge = ToolBridge()
        fake_result = {"status": "ok", "result": "12:00"}

        with patch("rex.openclaw.tool_bridge.get_openclaw_client", return_value=None):
            with patch(
                "rex.openclaw.tool_bridge._execute_tool", return_value=fake_result
            ) as mock_fn:
                req = {"tool": "time_now", "args": {}}
                result = bridge.execute_tool(req, {"timezone": "UTC"})
                mock_fn.assert_called_once_with(
                    req,
                    {"timezone": "UTC"},
                    skip_policy_check=False,
                    skip_credential_check=False,
                    task_id=None,
                    requested_by=None,
                    skip_audit_log=False,
                )
                assert result == fake_result

    def test_kwargs_forwarded_to_local(self):
        """skip_policy_check and other kwargs are forwarded when running locally."""
        bridge = ToolBridge()

        with patch("rex.openclaw.tool_bridge.get_openclaw_client", return_value=None):
            with patch("rex.openclaw.tool_bridge._execute_tool", return_value={}) as mock_fn:
                bridge.execute_tool(
                    {"tool": "time_now", "args": {}},
                    {},
                    skip_policy_check=True,
                    task_id="t-123",
                    requested_by="bot",
                    skip_audit_log=True,
                )
                _, kwargs = mock_fn.call_args
                assert kwargs["skip_policy_check"] is True
                assert kwargs["task_id"] == "t-123"
                assert kwargs["requested_by"] == "bot"
                assert kwargs["skip_audit_log"] is True
