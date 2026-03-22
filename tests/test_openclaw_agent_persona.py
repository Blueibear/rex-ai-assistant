"""Tests for US-P2-005: Rex persona injected into OpenClaw agent via config."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(
    wakeword: str = "rex",
    active_profile: str = "default",
    default_location: Optional[str] = None,
    default_timezone: Optional[str] = None,
    capabilities: Optional[List[str]] = None,
):
    """Return a minimal AppConfig-like object for testing."""
    from rex.config import AppConfig

    return AppConfig(
        wakeword=wakeword,
        active_profile=active_profile,
        default_location=default_location,
        default_timezone=default_timezone,
        capabilities=capabilities or [],
    )


# ---------------------------------------------------------------------------
# build_system_prompt tests
# ---------------------------------------------------------------------------

class TestBuildSystemPrompt:
    def test_returns_string(self):
        from rex.openclaw.config import build_system_prompt

        cfg = _make_config()
        result = build_system_prompt(cfg)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_agent_name_from_wakeword(self):
        from rex.openclaw.config import build_system_prompt

        cfg = _make_config(wakeword="rex")
        result = build_system_prompt(cfg)
        assert "Rex" in result

    def test_custom_wakeword_capitalised(self):
        from rex.openclaw.config import build_system_prompt

        cfg = _make_config(wakeword="aria")
        result = build_system_prompt(cfg)
        assert "Aria" in result

    def test_default_profile_not_mentioned(self):
        from rex.openclaw.config import build_system_prompt

        cfg = _make_config(active_profile="default")
        result = build_system_prompt(cfg)
        # "default" profile should not appear to avoid noise
        assert "profile" not in result.lower()

    def test_non_default_profile_included(self):
        from rex.openclaw.config import build_system_prompt

        cfg = _make_config(active_profile="work")
        result = build_system_prompt(cfg)
        assert "work" in result

    def test_location_included_when_set(self):
        from rex.openclaw.config import build_system_prompt

        cfg = _make_config(default_location="Edinburgh, Scotland")
        result = build_system_prompt(cfg)
        assert "Edinburgh" in result

    def test_timezone_included_when_set(self):
        from rex.openclaw.config import build_system_prompt

        cfg = _make_config(default_timezone="Europe/London")
        result = build_system_prompt(cfg)
        assert "Europe/London" in result

    def test_capabilities_included_when_set(self):
        from rex.openclaw.config import build_system_prompt

        cfg = _make_config(capabilities=["home_assistant", "web_search"])
        result = build_system_prompt(cfg)
        assert "home_assistant" in result
        assert "web_search" in result

    def test_no_location_not_mentioned(self):
        from rex.openclaw.config import build_system_prompt

        cfg = _make_config(default_location=None)
        result = build_system_prompt(cfg)
        assert "location" not in result.lower()

    def test_uses_global_config_when_none(self):
        """When config=None, build_system_prompt loads the global config."""
        from rex.openclaw.config import build_system_prompt

        fake_cfg = _make_config(wakeword="jarvis")
        with patch("rex.openclaw.config.load_config", return_value=fake_cfg):
            result = build_system_prompt(None)
        assert "Jarvis" in result


# ---------------------------------------------------------------------------
# RexAgent persona injection tests
# ---------------------------------------------------------------------------

class TestRexAgentPersona:
    def _make_agent(self, config=None, system_prompt=None, agent_name=None):
        from rex.openclaw.agent import RexAgent

        mock_llm = MagicMock()
        mock_llm.generate.return_value = "Hello from Rex"
        return RexAgent(
            llm=mock_llm,
            config=config,
            system_prompt=system_prompt,
            agent_name=agent_name,
        )

    def test_system_prompt_derived_from_config(self):
        cfg = _make_config(wakeword="rex")
        agent = self._make_agent(config=cfg)
        assert "Rex" in agent.system_prompt

    def test_explicit_system_prompt_overrides_config(self):
        cfg = _make_config(wakeword="rex")
        agent = self._make_agent(config=cfg, system_prompt="Custom persona override")
        assert agent.system_prompt == "Custom persona override"

    def test_agent_name_from_config(self):
        cfg = _make_config(wakeword="rex")
        agent = self._make_agent(config=cfg)
        assert agent.agent_name == "Rex"

    def test_explicit_agent_name_overrides_config(self):
        cfg = _make_config(wakeword="rex")
        agent = self._make_agent(config=cfg, agent_name="Jarvis")
        assert agent.agent_name == "Jarvis"

    def test_system_prompt_injected_in_respond(self):
        cfg = _make_config(wakeword="rex", default_location="London")
        agent = self._make_agent(config=cfg)

        agent.respond("Hello!")

        call_args = agent.llm.generate.call_args
        messages = call_args.kwargs.get("messages") or call_args.args[0]
        system_msg = next(m for m in messages if m["role"] == "system")
        assert "Rex" in system_msg["content"]

    def test_location_in_system_prompt_via_config(self):
        cfg = _make_config(wakeword="rex", default_location="Edinburgh, Scotland")
        agent = self._make_agent(config=cfg)
        assert "Edinburgh" in agent.system_prompt
