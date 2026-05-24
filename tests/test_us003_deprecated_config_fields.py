"""Tests for US-003: DeprecationWarning on high-traffic flat AppConfig fields.

Acceptance criteria verified:
- Accessing deprecated flat fields emits DeprecationWarning with nested-path guidance
- The deprecated value is identical to the nested sub-config value
- Existing code that reads flat fields still works (no AttributeError)
- load_config() itself does NOT emit DeprecationWarning (guarded by _deprecated_warnings_active)
"""

from __future__ import annotations

import warnings

import pytest

from rex.config import AppConfig, build_app_config


@pytest.fixture
def cfg() -> AppConfig:
    """A fully-constructed AppConfig with sub-configs built."""
    return build_app_config({})


# ---------------------------------------------------------------------------
# Deprecated flat fields (existing AppConfig fields emitting DeprecationWarning)
# ---------------------------------------------------------------------------


def test_llm_provider_emits_deprecation_warning(cfg: AppConfig) -> None:
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _ = cfg.llm_provider
    dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
    assert len(dep_warnings) >= 1
    assert "config.llm.llm_provider" in str(dep_warnings[0].message)


def test_tts_voice_emits_deprecation_warning(cfg: AppConfig) -> None:
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _ = cfg.tts_voice
    dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
    assert len(dep_warnings) >= 1
    assert "config.voice.tts_voice" in str(dep_warnings[0].message)


def test_whisper_device_emits_deprecation_warning(cfg: AppConfig) -> None:
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _ = cfg.whisper_device
    dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
    assert len(dep_warnings) >= 1
    assert "config.voice.whisper_device" in str(dep_warnings[0].message)


def test_openclaw_gateway_url_emits_deprecation_warning(cfg: AppConfig) -> None:
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _ = cfg.openclaw_gateway_url
    dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
    assert len(dep_warnings) >= 1
    assert "config.integrations.openclaw_gateway_url" in str(dep_warnings[0].message)


# ---------------------------------------------------------------------------
# Deprecated property aliases (new names pointing to sub-config values)
# ---------------------------------------------------------------------------


def test_model_name_alias_warns_and_returns_value(cfg: AppConfig) -> None:
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        val = cfg.model_name
    dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
    assert len(dep_warnings) >= 1
    assert "config.llm.model_name" in str(dep_warnings[0].message)
    assert val == cfg.llm.model_name


def test_tts_engine_alias_warns_and_returns_value(cfg: AppConfig) -> None:
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        val = cfg.tts_engine
    dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
    assert len(dep_warnings) >= 1
    assert "config.voice.tts_engine" in str(dep_warnings[0].message)
    assert val == cfg.voice.tts_engine


def test_wakeword_model_alias_warns_and_returns_value(cfg: AppConfig) -> None:
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        val = cfg.wakeword_model
    dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
    assert len(dep_warnings) >= 1
    assert "config.voice.wakeword_model" in str(dep_warnings[0].message)
    assert val == cfg.voice.wakeword_model


def test_home_assistant_base_url_alias_warns(cfg: AppConfig) -> None:
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        val = cfg.home_assistant_base_url
    dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
    assert len(dep_warnings) >= 1
    assert "config.integrations.home_assistant_base_url" in str(dep_warnings[0].message)
    assert val == cfg.integrations.home_assistant_base_url


def test_tool_timeout_alias_warns_and_returns_value(cfg: AppConfig) -> None:
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        val = cfg.tool_timeout
    dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
    assert len(dep_warnings) >= 1
    assert "config.tools.tool_timeout" in str(dep_warnings[0].message)
    assert val == cfg.tools.tool_timeout


def test_gui_port_alias_warns_and_returns_value(cfg: AppConfig) -> None:
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        val = cfg.gui_port
    dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
    assert len(dep_warnings) >= 1
    assert "config.ui.gui_port" in str(dep_warnings[0].message)
    assert val == cfg.ui.gui_port


def test_api_key_env_alias_warns_and_returns_value(cfg: AppConfig) -> None:
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        val = cfg.api_key_env
    dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
    assert len(dep_warnings) >= 1
    assert "config.security.api_key_env" in str(dep_warnings[0].message)
    assert val == cfg.security.api_key_env


def test_rate_limit_per_minute_alias_warns_and_returns_value(cfg: AppConfig) -> None:
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        val = cfg.rate_limit_per_minute
    dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
    assert len(dep_warnings) >= 1
    assert "config.security.rate_limit_per_minute" in str(dep_warnings[0].message)
    assert val == cfg.security.rate_limit_per_minute


# ---------------------------------------------------------------------------
# Values are identical to nested path values
# ---------------------------------------------------------------------------


def test_deprecated_flat_values_match_sub_config(cfg: AppConfig) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        assert cfg.llm_provider == cfg.llm.llm_provider
        assert cfg.tts_voice == cfg.voice.tts_voice
        assert cfg.whisper_device == cfg.voice.whisper_device
        assert cfg.openclaw_gateway_url == cfg.integrations.openclaw_gateway_url


# ---------------------------------------------------------------------------
# load_config() must NOT emit DeprecationWarning (existing AC from US-020)
# ---------------------------------------------------------------------------


def test_load_config_no_deprecation_warning() -> None:
    """load_config() itself must not trigger deprecated field access."""
    from rex.config import load_config  # noqa: PLC0415

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        load_config()  # must not raise DeprecationWarning


# ---------------------------------------------------------------------------
# Flat fields still work without AttributeError
# ---------------------------------------------------------------------------


def test_deprecated_flat_fields_still_return_values(cfg: AppConfig) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        assert isinstance(cfg.llm_provider, str)
        assert cfg.tts_voice is None or isinstance(cfg.tts_voice, str)
        assert isinstance(cfg.whisper_device, str)
        assert isinstance(cfg.openclaw_gateway_url, str)
