"""Tests for US-P4-011: assistant.py uses ToolBridge when use_openclaw_tools flag is set.

Acceptance criteria:
  - When use_openclaw_tools=False, _tool_router_fn is the legacy route_if_tool_request
  - When use_openclaw_tools=True, _tool_router_fn is ToolBridge.route_if_tool_request
  - generate_reply uses the bridge when flag is True
  - generate_reply uses old router when flag is False
  - Tests pass
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rex.assistant import Assistant
from rex.config import AppConfig


def _make_settings(*, use_openclaw_tools: bool = False) -> AppConfig:
    """Build a minimal AppConfig for testing."""
    cfg = AppConfig(llm_provider="transformers", llm_model="sshleifer/tiny-gpt2")
    object.__setattr__(cfg, "use_openclaw_tools", use_openclaw_tools)
    return cfg


class TestToolRouterFnSelection:
    """Assistant picks the right routing callable at __init__ time."""

    def test_legacy_fn_when_flag_false(self):
        """When use_openclaw_tools=False, _tool_router_fn is legacy route_if_tool_request."""
        from rex.tool_router import route_if_tool_request as legacy_fn

        s = _make_settings(use_openclaw_tools=False)
        a = Assistant(settings_obj=s)
        assert a._tool_router_fn is legacy_fn

    def test_bridge_fn_when_flag_true(self):
        """When use_openclaw_tools=True, _tool_router_fn is ToolBridge.route_if_tool_request."""
        from rex.openclaw.tool_bridge import ToolBridge

        s = _make_settings(use_openclaw_tools=True)
        a = Assistant(settings_obj=s)
        # The bound method belongs to a ToolBridge instance
        assert isinstance(a._tool_router_fn.__self__, ToolBridge)
        assert a._tool_router_fn.__func__ is ToolBridge.route_if_tool_request

    def test_different_instances_independent(self):
        """Two assistants with different flag values get different routing fns."""
        from rex.openclaw.tool_bridge import ToolBridge
        from rex.tool_router import route_if_tool_request as legacy_fn

        a_legacy = Assistant(settings_obj=_make_settings(use_openclaw_tools=False))
        a_bridge = Assistant(settings_obj=_make_settings(use_openclaw_tools=True))

        assert a_legacy._tool_router_fn is legacy_fn
        assert isinstance(a_bridge._tool_router_fn.__self__, ToolBridge)


class TestGenerateReplyRouting:
    """generate_reply calls the selected routing function."""

    def _settings_with_flag(self, flag: bool) -> AppConfig:
        cfg = _make_settings(use_openclaw_tools=flag)
        # Ensure HA bridge is not created
        object.__setattr__(cfg, "ha_base_url", None)
        object.__setattr__(cfg, "ha_token", None)
        return cfg

    @pytest.mark.asyncio
    async def test_generate_reply_calls_legacy_router_when_flag_false(self):
        """generate_reply delegates to legacy route_if_tool_request when flag is False."""
        s = self._settings_with_flag(False)
        a = Assistant(settings_obj=s)

        # Patch the routing function on the instance
        mock_router = MagicMock(return_value="the time is now")
        a._tool_router_fn = mock_router

        with patch.object(a._llm, "generate", return_value="some text"):
            result = await a.generate_reply("What time is it?")

        mock_router.assert_called_once()
        assert result == "the time is now"

    @pytest.mark.asyncio
    async def test_generate_reply_calls_bridge_router_when_flag_true(self):
        """generate_reply delegates to ToolBridge.route_if_tool_request when flag is True."""
        s = self._settings_with_flag(True)
        a = Assistant(settings_obj=s)

        mock_router = MagicMock(return_value="bridge routed result")
        a._tool_router_fn = mock_router

        with patch.object(a._llm, "generate", return_value="some llm text"):
            result = await a.generate_reply("What is the weather?")

        mock_router.assert_called_once()
        assert result == "bridge routed result"

    @pytest.mark.asyncio
    async def test_router_receives_correct_args(self):
        """_tool_router_fn is called with (completion, tool_context, model_call_fn)."""
        s = self._settings_with_flag(False)
        a = Assistant(settings_obj=s)

        captured = {}

        def capture_router(completion, ctx, model_fn):
            captured["completion"] = completion
            captured["ctx"] = ctx
            captured["model_fn"] = model_fn
            return completion

        a._tool_router_fn = capture_router

        with patch.object(a._llm, "generate", return_value="llm reply"):
            await a.generate_reply("Hello")

        assert captured["completion"] == "llm reply"
        assert isinstance(captured["ctx"], dict)
        assert callable(captured["model_fn"])
