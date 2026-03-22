"""Tests for US-P3-016: Wire profile manager into OpenClaw agent.

Verifies that:
- apply_profile_to_config() updates capabilities and active_profile on AppConfig.
- RexAgent(profile_name=...) applies the profile so the persona reflects it.
- build_system_prompt() and build_agent_config() reflect profile-applied values.
- profile_name=None leaves config unchanged.
- Missing profile raises FileNotFoundError.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from rex.config import AppConfig
from rex.openclaw.config import (
    apply_profile_to_config,
    build_agent_config,
    build_system_prompt,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SAMPLE_PROFILE = {
    "profile_version": 1,
    "name": "testprofile",
    "description": "A test profile",
    "capabilities": ["web_search", "ha_router"],
    "overrides": {},
}

_EMPTY_CAPS_PROFILE = {
    "profile_version": 1,
    "name": "minimal",
    "description": "Minimal profile",
    "capabilities": [],
    "overrides": {},
}


def _base_config(**kwargs) -> AppConfig:
    """Return a minimal AppConfig for testing (avoids loading global config)."""
    defaults = dict(
        wakeword="rex",
        active_profile="default",
        capabilities=[],
        default_location="",
        default_timezone="",
    )
    defaults.update(kwargs)
    return AppConfig(**defaults)


# ---------------------------------------------------------------------------
# Tests for apply_profile_to_config()
# ---------------------------------------------------------------------------


class TestApplyProfileToConfig:
    def test_capabilities_replaced_from_profile(self, tmp_path):
        profile_file = tmp_path / "testprofile.json"
        profile_file.write_text(json.dumps(_SAMPLE_PROFILE))

        base = _base_config(capabilities=["old_cap"])
        result = apply_profile_to_config(base, "testprofile", profiles_dir=str(tmp_path))

        assert result.capabilities == ["web_search", "ha_router"]

    def test_active_profile_set_to_profile_name(self, tmp_path):
        profile_file = tmp_path / "testprofile.json"
        profile_file.write_text(json.dumps(_SAMPLE_PROFILE))

        base = _base_config()
        result = apply_profile_to_config(base, "testprofile", profiles_dir=str(tmp_path))

        assert result.active_profile == "testprofile"

    def test_returns_new_config_instance(self, tmp_path):
        profile_file = tmp_path / "testprofile.json"
        profile_file.write_text(json.dumps(_SAMPLE_PROFILE))

        base = _base_config()
        result = apply_profile_to_config(base, "testprofile", profiles_dir=str(tmp_path))

        assert result is not base

    def test_other_fields_preserved(self, tmp_path):
        profile_file = tmp_path / "testprofile.json"
        profile_file.write_text(json.dumps(_SAMPLE_PROFILE))

        base = _base_config(wakeword="jarvis", default_location="London")
        result = apply_profile_to_config(base, "testprofile", profiles_dir=str(tmp_path))

        assert result.wakeword == "jarvis"
        assert result.default_location == "London"

    def test_empty_capabilities_profile(self, tmp_path):
        profile_file = tmp_path / "minimal.json"
        profile_file.write_text(json.dumps(_EMPTY_CAPS_PROFILE))

        base = _base_config(capabilities=["something"])
        result = apply_profile_to_config(base, "minimal", profiles_dir=str(tmp_path))

        assert result.capabilities == []
        assert result.active_profile == "minimal"

    def test_missing_profile_raises_file_not_found(self, tmp_path):
        base = _base_config()
        with pytest.raises(FileNotFoundError):
            apply_profile_to_config(base, "nonexistent", profiles_dir=str(tmp_path))


# ---------------------------------------------------------------------------
# Tests for build_system_prompt() with profile-applied config
# ---------------------------------------------------------------------------


class TestBuildSystemPromptWithProfile:
    def test_profile_capabilities_appear_in_prompt(self, tmp_path):
        profile_file = tmp_path / "testprofile.json"
        profile_file.write_text(json.dumps(_SAMPLE_PROFILE))

        base = _base_config(capabilities=[])
        applied = apply_profile_to_config(base, "testprofile", profiles_dir=str(tmp_path))
        prompt = build_system_prompt(applied)

        assert "web_search" in prompt
        assert "ha_router" in prompt

    def test_profile_name_appears_in_prompt(self, tmp_path):
        profile_file = tmp_path / "testprofile.json"
        profile_file.write_text(json.dumps(_SAMPLE_PROFILE))

        base = _base_config()
        applied = apply_profile_to_config(base, "testprofile", profiles_dir=str(tmp_path))
        prompt = build_system_prompt(applied)

        assert "testprofile" in prompt

    def test_default_profile_omitted_from_prompt(self):
        # "default" active_profile should not appear in the system prompt (noise reduction).
        base = _base_config(active_profile="default")
        prompt = build_system_prompt(base)

        assert "default" not in prompt


# ---------------------------------------------------------------------------
# Tests for build_agent_config() with profile-applied config
# ---------------------------------------------------------------------------


class TestBuildAgentConfigWithProfile:
    def test_profile_capabilities_in_agent_config(self, tmp_path):
        profile_file = tmp_path / "testprofile.json"
        profile_file.write_text(json.dumps(_SAMPLE_PROFILE))

        base = _base_config(capabilities=[])
        applied = apply_profile_to_config(base, "testprofile", profiles_dir=str(tmp_path))
        cfg = build_agent_config(applied)

        assert "web_search" in cfg["rex_capabilities"]
        assert "ha_router" in cfg["rex_capabilities"]

    def test_active_profile_in_agent_config(self, tmp_path):
        profile_file = tmp_path / "testprofile.json"
        profile_file.write_text(json.dumps(_SAMPLE_PROFILE))

        base = _base_config()
        applied = apply_profile_to_config(base, "testprofile", profiles_dir=str(tmp_path))
        cfg = build_agent_config(applied)

        assert cfg["active_profile"] == "testprofile"


# ---------------------------------------------------------------------------
# Tests for RexAgent(profile_name=...)
# ---------------------------------------------------------------------------


class TestRexAgentProfileName:
    def _make_agent(self, profile_name, tmp_path, base_config=None):
        """Create a RexAgent with a mocked LLM and an in-test profile dir."""
        from rex.openclaw.agent import RexAgent

        if base_config is None:
            base_config = _base_config()

        mock_llm = MagicMock()
        mock_llm.generate.return_value = "ok"

        # Patch apply_profile_to_config inside agent.py's lazy import scope
        # by patching the function in rex.openclaw.config directly.
        original_apply = apply_profile_to_config

        def _patched_apply(cfg, name, profiles_dir=None):
            return original_apply(cfg, name, profiles_dir=str(tmp_path))

        with patch("rex.openclaw.config.apply_profile_to_config", side_effect=_patched_apply):
            agent = RexAgent(llm=mock_llm, profile_name=profile_name, config=base_config)
        return agent

    def test_profile_name_none_leaves_config_unchanged(self, tmp_path):
        base = _base_config(capabilities=["voice"])
        from rex.openclaw.agent import RexAgent

        mock_llm = MagicMock()
        agent = RexAgent(llm=mock_llm, config=base, profile_name=None)

        assert "voice" in agent.system_prompt

    def test_profile_name_updates_system_prompt_capabilities(self, tmp_path):
        profile_file = tmp_path / "testprofile.json"
        profile_file.write_text(json.dumps(_SAMPLE_PROFILE))

        agent = self._make_agent("testprofile", tmp_path)

        assert "web_search" in agent.system_prompt
        assert "ha_router" in agent.system_prompt

    def test_profile_name_updates_active_profile_in_prompt(self, tmp_path):
        profile_file = tmp_path / "testprofile.json"
        profile_file.write_text(json.dumps(_SAMPLE_PROFILE))

        agent = self._make_agent("testprofile", tmp_path)

        assert "testprofile" in agent.system_prompt

    def test_missing_profile_name_raises(self, tmp_path):
        from rex.openclaw.agent import RexAgent

        base = _base_config()
        mock_llm = MagicMock()
        with pytest.raises(FileNotFoundError):
            RexAgent(llm=mock_llm, config=base, profile_name="nonexistent")
