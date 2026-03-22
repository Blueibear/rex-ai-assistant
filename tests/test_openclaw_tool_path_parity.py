"""Tests for US-P4-012: confirm parity between legacy and bridge tool paths.

Acceptance criteria:
  - Both paths (use_openclaw_tools=False and True) produce identical output
    for non-tool text, for tool-request text, and when a tool is executed
  - Tests pass
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from rex.config import AppConfig
from rex.openclaw.tool_bridge import ToolBridge
from rex.tool_router import route_if_tool_request as legacy_route


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _settings(*, use_openclaw_tools: bool) -> AppConfig:
    cfg = AppConfig(llm_provider="transformers", llm_model="sshleifer/tiny-gpt2")
    object.__setattr__(cfg, "use_openclaw_tools", use_openclaw_tools)
    return cfg


def _bridge_route(llm_text: str, ctx: dict, model_fn, *, skip_policy_check: bool = False) -> str:
    return ToolBridge().route_if_tool_request(llm_text, ctx, model_fn, skip_policy_check=skip_policy_check)


# ---------------------------------------------------------------------------
# Unit-level parity: same inputs → same outputs from both callables
# ---------------------------------------------------------------------------


class TestRouterParity:
    """Legacy route_if_tool_request and ToolBridge.route_if_tool_request produce identical results."""

    def test_non_tool_text_unchanged_both_paths(self):
        """Plain text passes through unchanged on both paths."""
        text = "The sky is blue."
        model_fn = MagicMock(return_value="should not be called")
        ctx: dict = {}

        legacy_result = legacy_route(text, ctx, model_fn)
        bridge_result = _bridge_route(text, ctx, model_fn)

        assert legacy_result == text
        assert bridge_result == text
        assert legacy_result == bridge_result
        model_fn.assert_not_called()

    def test_empty_string_unchanged_both_paths(self):
        """Empty string passes through unchanged on both paths."""
        model_fn = MagicMock(return_value="nope")
        legacy_result = legacy_route("", {}, model_fn)
        bridge_result = _bridge_route("", {}, model_fn)
        assert legacy_result == bridge_result == ""

    def test_tool_request_routes_and_both_paths_agree(self):
        """Both paths route a TOOL_REQUEST line and re-call the model."""
        tool_line = 'TOOL_REQUEST: {"tool": "time_now", "args": {"location": "UTC"}}'
        fake_tool_result = {"status": "ok", "result": "12:00 UTC"}
        fake_model_reply = "The current time is 12:00 UTC."

        def _model_fn(msg: dict) -> str:
            return fake_model_reply

        with patch("rex.tool_router.execute_tool", return_value=fake_tool_result):
            legacy_result = legacy_route(tool_line, {}, _model_fn)

        with patch("rex.tool_router.execute_tool", return_value=fake_tool_result):
            bridge_result = _bridge_route(tool_line, {}, _model_fn)

        assert legacy_result == bridge_result == fake_model_reply

    def test_unknown_tool_both_paths_agree(self):
        """Both paths behave identically for unknown tool names."""
        tool_line = 'TOOL_REQUEST: {"tool": "nonexistent_tool", "args": {}}'
        model_fn = MagicMock(return_value="fallback reply")

        legacy_result = legacy_route(tool_line, {}, model_fn)
        bridge_result = _bridge_route(tool_line, {}, model_fn)

        assert legacy_result == bridge_result

    def test_skip_policy_check_both_paths_agree(self):
        """skip_policy_check=True behaves identically on both paths."""
        tool_line = 'TOOL_REQUEST: {"tool": "time_now", "args": {}}'
        fake_result = {"status": "ok", "result": "09:00"}
        model_fn = MagicMock(return_value="nine AM")

        with patch("rex.tool_router.execute_tool", return_value=fake_result):
            legacy_result = legacy_route(tool_line, {}, model_fn, skip_policy_check=True)

        with patch("rex.tool_router.execute_tool", return_value=fake_result):
            bridge_result = _bridge_route(
                tool_line, {}, model_fn, skip_policy_check=True
            )

        assert legacy_result == bridge_result


# ---------------------------------------------------------------------------
# Structural parity: ToolBridge delegates to the same underlying function
# ---------------------------------------------------------------------------


class TestBridgeDelegation:
    """ToolBridge.route_if_tool_request delegates to the same underlying fn as legacy."""

    def test_bridge_calls_same_underlying_fn(self):
        """ToolBridge uses _route_if_tool_request from tool_router internally."""
        fake_reply = "delegated reply"
        with patch(
            "rex.openclaw.tool_bridge._route_if_tool_request", return_value=fake_reply
        ) as mock_fn:
            bridge = ToolBridge()
            result = bridge.route_if_tool_request(
                'TOOL_REQUEST: {"tool": "time_now", "args": {}}',
                {"timezone": "UTC"},
                lambda msg: "model reply",
            )

        mock_fn.assert_called_once()
        assert result == fake_reply

    def test_bridge_and_legacy_share_same_underlying_module(self):
        """Both paths share the same route_if_tool_request in rex.tool_router."""
        from rex.openclaw.tool_bridge import _route_if_tool_request as bridge_fn
        from rex.tool_router import route_if_tool_request as direct_fn

        # They should be the exact same object
        assert bridge_fn is direct_fn


# ---------------------------------------------------------------------------
# Assistant-level parity: generate_reply produces same output for both flags
# ---------------------------------------------------------------------------


class TestAssistantPathParity:
    """Assistant.generate_reply produces identical output regardless of flag."""

    @pytest.mark.asyncio
    async def test_non_tool_reply_parity(self):
        """For a plain non-tool LLM reply, both flag settings produce the same result."""
        from rex.assistant import Assistant

        plain_reply = "Hello! I am Rex."

        for flag in (False, True):
            cfg = _settings(use_openclaw_tools=flag)
            a = Assistant(settings_obj=cfg)
            with patch.object(a._llm, "generate", return_value=plain_reply):
                result = await a.generate_reply("Say hello.")
            assert result == plain_reply, f"Failed for use_openclaw_tools={flag}"

    @pytest.mark.asyncio
    async def test_tool_reply_parity(self):
        """Both flag settings produce the same result when tool routing fires."""
        from rex.assistant import Assistant

        tool_line = 'TOOL_REQUEST: {"tool": "time_now", "args": {}}'
        fake_tool_result = {"status": "ok", "result": "15:00"}
        final_reply = "It is 3 PM."

        results = {}
        for flag in (False, True):
            cfg = _settings(use_openclaw_tools=flag)
            a = Assistant(settings_obj=cfg)

            def _make_model_fn(first_reply, tool_result, second_reply):
                calls = []

                def _gen(prompt=None, messages=None):
                    calls.append(1)
                    if len(calls) == 1:
                        return first_reply
                    return second_reply

                return _gen

            gen_fn = _make_model_fn(tool_line, fake_tool_result, final_reply)
            with (
                patch.object(a._llm, "generate", side_effect=gen_fn),
                patch("rex.tool_router.execute_tool", return_value=fake_tool_result),
            ):
                result = await a.generate_reply("What time is it?")

            results[flag] = result

        assert results[False] == results[True], (
            f"Path parity broken: legacy={results[False]!r}, bridge={results[True]!r}"
        )
