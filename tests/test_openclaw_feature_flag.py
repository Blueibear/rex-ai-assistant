"""Tests for US-P4-010: USE_OPENCLAW_TOOLS feature flag in AppConfig.

Acceptance criteria:
  - AppConfig has use_openclaw_tools field, defaults to False
  - build_app_config reads openclaw.use_tools from JSON config
  - Flag can be set to True via JSON config
  - Tests pass
"""

from __future__ import annotations

from rex.config import AppConfig, build_app_config

BASE_JSON = {
    "models": {
        "llm_provider": "transformers",
        "llm_model": "sshleifer/tiny-gpt2",
    }
}


class TestUseOpenclawToolsDefault:
    """use_openclaw_tools defaults to False when not configured."""

    def test_appconfig_has_use_openclaw_tools_field(self):
        """AppConfig dataclass exposes use_openclaw_tools attribute."""
        cfg = AppConfig(llm_provider="transformers", llm_model="sshleifer/tiny-gpt2")
        assert hasattr(cfg, "use_openclaw_tools")

    def test_appconfig_default_is_false(self):
        """use_openclaw_tools defaults to False."""
        cfg = AppConfig(llm_provider="transformers", llm_model="sshleifer/tiny-gpt2")
        assert cfg.use_openclaw_tools is False

    def test_build_app_config_default_is_false(self):
        """build_app_config returns False when openclaw section absent."""
        cfg = build_app_config(BASE_JSON)
        assert cfg.use_openclaw_tools is False

    def test_build_app_config_openclaw_section_missing(self):
        """build_app_config returns False when openclaw key not present."""
        json_cfg = dict(BASE_JSON)
        cfg = build_app_config(json_cfg)
        assert cfg.use_openclaw_tools is False


class TestUseOpenclawToolsEnabled:
    """use_openclaw_tools reads True from openclaw.use_tools in JSON config."""

    def test_build_app_config_reads_use_tools_true(self):
        """build_app_config picks up openclaw.use_tools=true from JSON."""
        json_cfg = {**BASE_JSON, "openclaw": {"use_tools": True}}
        cfg = build_app_config(json_cfg)
        assert cfg.use_openclaw_tools is True

    def test_build_app_config_reads_use_tools_false(self):
        """build_app_config picks up openclaw.use_tools=false from JSON."""
        json_cfg = {**BASE_JSON, "openclaw": {"use_tools": False}}
        cfg = build_app_config(json_cfg)
        assert cfg.use_openclaw_tools is False

    def test_use_openclaw_tools_is_bool(self):
        """use_openclaw_tools coerced to bool."""
        json_cfg = {**BASE_JSON, "openclaw": {"use_tools": 1}}
        cfg = build_app_config(json_cfg)
        assert isinstance(cfg.use_openclaw_tools, bool)
        assert cfg.use_openclaw_tools is True
