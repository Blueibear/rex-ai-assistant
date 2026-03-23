"""Tests for US-P6-006: USE_OPENCLAW_VOICE_BACKEND feature flag in AppConfig.

Acceptance criteria:
  - AppConfig has use_openclaw_voice_backend field, defaults to False
  - build_app_config reads openclaw.use_voice_backend from JSON config
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


class TestUseOpenclawVoiceBackendDefault:
    """use_openclaw_voice_backend defaults to False when not configured."""

    def test_appconfig_has_field(self):
        """AppConfig dataclass exposes use_openclaw_voice_backend attribute."""
        cfg = AppConfig(llm_provider="transformers", llm_model="sshleifer/tiny-gpt2")
        assert hasattr(cfg, "use_openclaw_voice_backend")

    def test_appconfig_default_is_false(self):
        """use_openclaw_voice_backend defaults to False."""
        cfg = AppConfig(llm_provider="transformers", llm_model="sshleifer/tiny-gpt2")
        assert cfg.use_openclaw_voice_backend is False

    def test_build_app_config_default_is_false(self):
        """build_app_config returns False when openclaw section absent."""
        cfg = build_app_config(BASE_JSON)
        assert cfg.use_openclaw_voice_backend is False

    def test_build_app_config_openclaw_section_missing(self):
        """build_app_config returns False when openclaw key not present."""
        cfg = build_app_config(dict(BASE_JSON))
        assert cfg.use_openclaw_voice_backend is False

    def test_use_tools_unaffected(self):
        """use_openclaw_tools is independent of use_openclaw_voice_backend."""
        cfg = build_app_config({**BASE_JSON, "openclaw": {"use_voice_backend": True}})
        assert cfg.use_openclaw_voice_backend is True
        assert cfg.use_openclaw_tools is False


class TestUseOpenclawVoiceBackendEnabled:
    """use_openclaw_voice_backend reads True from openclaw.use_voice_backend in JSON."""

    def test_build_app_config_reads_true(self):
        """build_app_config picks up openclaw.use_voice_backend=true from JSON."""
        json_cfg = {**BASE_JSON, "openclaw": {"use_voice_backend": True}}
        cfg = build_app_config(json_cfg)
        assert cfg.use_openclaw_voice_backend is True

    def test_build_app_config_reads_false(self):
        """build_app_config picks up openclaw.use_voice_backend=false from JSON."""
        json_cfg = {**BASE_JSON, "openclaw": {"use_voice_backend": False}}
        cfg = build_app_config(json_cfg)
        assert cfg.use_openclaw_voice_backend is False

    def test_use_voice_backend_is_bool(self):
        """use_openclaw_voice_backend coerced to bool."""
        json_cfg = {**BASE_JSON, "openclaw": {"use_voice_backend": 1}}
        cfg = build_app_config(json_cfg)
        assert isinstance(cfg.use_openclaw_voice_backend, bool)
        assert cfg.use_openclaw_voice_backend is True

    def test_both_flags_independently_settable(self):
        """Both use_tools and use_voice_backend can be set independently."""
        json_cfg = {**BASE_JSON, "openclaw": {"use_tools": True, "use_voice_backend": True}}
        cfg = build_app_config(json_cfg)
        assert cfg.use_openclaw_tools is True
        assert cfg.use_openclaw_voice_backend is True
