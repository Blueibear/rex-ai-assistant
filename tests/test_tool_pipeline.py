"""Unit tests for ToolDispatcher timeout and retry pipeline — US-TD-003.

Covers:
- Successful invocation logs name, duration, success
- Timeout returns "I couldn't reach [name] in time" and continues
- One retry on transient error (ConnectionError, HTTP 5xx, OSError)
- No retry on auth error (PermissionError, auth_error flag)
- tool_timeout_seconds read from config (default 10.0)
"""

from __future__ import annotations

import threading
from typing import Any
from unittest.mock import MagicMock, patch

from rex.tools.dispatcher import (
    ToolDispatcher,
    _is_auth_error,
    _is_transient_error,
)
from rex.tools.registry import Tool, ToolRegistry

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tool(name: str, handler: Any) -> Tool:
    return Tool(
        name=name,
        description=f"Test tool: {name}",
        capability_tags=["test"],
        requires_config=[],
        handler=handler,
    )


def _make_registry(*tools: Tool) -> ToolRegistry:
    reg = ToolRegistry()
    for t in tools:
        reg.register(t)
    return reg


def _make_dispatcher(tool: Tool, timeout: float = 5.0) -> ToolDispatcher:
    cfg = MagicMock()
    cfg.tool_timeout_seconds = timeout
    return ToolDispatcher(_make_registry(tool), config=cfg)


# ---------------------------------------------------------------------------
# Error classifier tests
# ---------------------------------------------------------------------------


class TestIsTransientError:
    def test_timeout_error(self) -> None:
        assert _is_transient_error(TimeoutError("timed out"))

    def test_connection_error(self) -> None:
        assert _is_transient_error(ConnectionError("refused"))

    def test_os_error(self) -> None:
        assert _is_transient_error(OSError("network"))

    def test_http_5xx(self) -> None:
        exc = Exception("server error")
        exc.status_code = 503  # type: ignore[attr-defined]
        assert _is_transient_error(exc)

    def test_http_4xx_not_transient(self) -> None:
        exc = Exception("not found")
        exc.status_code = 404  # type: ignore[attr-defined]
        assert not _is_transient_error(exc)

    def test_is_transient_flag(self) -> None:
        exc = Exception("custom transient")
        exc.is_transient = True  # type: ignore[attr-defined]
        assert _is_transient_error(exc)

    def test_value_error_not_transient(self) -> None:
        assert not _is_transient_error(ValueError("bad value"))


class TestIsAuthError:
    def test_permission_error(self) -> None:
        assert _is_auth_error(PermissionError("denied"))

    def test_http_401(self) -> None:
        exc = Exception("unauthorized")
        exc.status_code = 401  # type: ignore[attr-defined]
        assert _is_auth_error(exc)

    def test_http_403(self) -> None:
        exc = Exception("forbidden")
        exc.status_code = 403  # type: ignore[attr-defined]
        assert _is_auth_error(exc)

    def test_auth_error_flag(self) -> None:
        exc = Exception("custom auth")
        exc.auth_error = True  # type: ignore[attr-defined]
        assert _is_auth_error(exc)

    def test_connection_error_not_auth(self) -> None:
        assert not _is_auth_error(ConnectionError("network"))


# ---------------------------------------------------------------------------
# execute_tools pipeline tests
# ---------------------------------------------------------------------------


class TestExecuteToolsSuccessful:
    def test_result_stored(self) -> None:
        tool = _make_tool("my_tool", lambda **kw: {"ok": True})
        disp = _make_dispatcher(tool)
        results = disp.execute_tools([tool], "anything")
        assert results["my_tool"] == {"ok": True}

    def test_logged_as_ok(self) -> None:
        tool = _make_tool("my_tool", lambda **kw: "done")
        disp = _make_dispatcher(tool)
        import rex.tools.dispatcher as _mod

        with patch.object(_mod.logger, "info") as mock_info:
            disp.execute_tools([tool], "msg")
        # At least one INFO call whose args mention the tool name and "ok"
        logged_ok = any(
            any("my_tool" in str(a) for a in c[0]) and any("ok" in str(a) for a in c[0])
            for c in mock_info.call_args_list
        )
        assert logged_ok


class TestExecuteToolsTimeout:
    def test_timeout_message_returned(self) -> None:
        barrier = threading.Event()

        def slow_handler(**kw: Any) -> str:
            barrier.wait(timeout=5)
            return "done"

        tool = _make_tool("slow_tool", slow_handler)
        disp = _make_dispatcher(tool, timeout=0.05)
        results = disp.execute_tools([tool], "test")
        barrier.set()  # release the thread
        assert "slow_tool" in results
        assert "I couldn't reach slow_tool in time" == results["slow_tool"]

    def test_other_tools_continue_after_timeout(self) -> None:
        barrier = threading.Event()

        def slow_handler(**kw: Any) -> str:
            barrier.wait(timeout=5)
            return "slow"

        slow_tool = _make_tool("slow_tool", slow_handler)
        fast_tool = _make_tool("fast_tool", lambda **kw: "fast")
        reg = _make_registry(slow_tool, fast_tool)
        cfg = MagicMock()
        cfg.tool_timeout_seconds = 0.05
        disp = ToolDispatcher(reg, config=cfg)
        results = disp.execute_tools([slow_tool, fast_tool], "test")
        barrier.set()
        assert "I couldn't reach slow_tool in time" == results["slow_tool"]
        assert results["fast_tool"] == "fast"


class TestExecuteToolsRetry:
    def test_retry_on_connection_error_succeeds(self) -> None:
        call_count = 0

        def flaky_handler(**kw: Any) -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("network blip")
            return "recovered"

        tool = _make_tool("flaky_tool", flaky_handler)
        disp = _make_dispatcher(tool)
        results = disp.execute_tools([tool], "test")
        assert results["flaky_tool"] == "recovered"
        assert call_count == 2  # retried once

    def test_retry_on_http_5xx_succeeds(self) -> None:
        call_count = 0

        def server_error_then_ok(**kw: Any) -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                exc = Exception("server error")
                exc.status_code = 503  # type: ignore[attr-defined]
                raise exc
            return "ok_after_retry"

        tool = _make_tool("backend_tool", server_error_then_ok)
        disp = _make_dispatcher(tool)
        results = disp.execute_tools([tool], "test")
        assert results["backend_tool"] == "ok_after_retry"
        assert call_count == 2

    def test_no_retry_on_auth_error(self) -> None:
        call_count = 0

        def auth_failure(**kw: Any) -> str:
            nonlocal call_count
            call_count += 1
            raise PermissionError("401 unauthorized")

        tool = _make_tool("auth_tool", auth_failure)
        disp = _make_dispatcher(tool)
        results = disp.execute_tools([tool], "test")
        assert call_count == 1  # not retried
        assert "tool error" in results["auth_tool"]

    def test_no_retry_on_value_error(self) -> None:
        call_count = 0

        def bad_args(**kw: Any) -> str:
            nonlocal call_count
            call_count += 1
            raise ValueError("bad input")

        tool = _make_tool("strict_tool", bad_args)
        disp = _make_dispatcher(tool)
        disp.execute_tools([tool], "test")
        assert call_count == 1  # not retried

    def test_transient_fails_twice_stored_as_error(self) -> None:
        def always_fail(**kw: Any) -> str:
            raise ConnectionError("persistent network failure")

        tool = _make_tool("dead_tool", always_fail)
        disp = _make_dispatcher(tool)
        results = disp.execute_tools([tool], "test")
        assert "tool error" in results["dead_tool"]


class TestToolTimeoutFromConfig:
    def test_default_timeout_is_10s(self) -> None:
        reg = ToolRegistry()
        disp = ToolDispatcher(reg)
        assert disp._timeout_seconds == 10.0

    def test_config_timeout_used(self) -> None:
        cfg = MagicMock()
        cfg.tool_timeout_seconds = 42.0
        disp = ToolDispatcher(ToolRegistry(), config=cfg)
        assert disp._timeout_seconds == 42.0
