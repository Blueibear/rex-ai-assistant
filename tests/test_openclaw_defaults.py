from __future__ import annotations

from pathlib import Path

import pytest

from rex.config import AppConfig, ConfigurationError, validate_config


def test_openclaw_feature_flags_default_off() -> None:
    config = AppConfig()
    assert config.use_openclaw_tools is False
    assert config.use_openclaw_voice_backend is False


@pytest.mark.parametrize("flag", ["use_openclaw_tools", "use_openclaw_voice_backend"])
def test_enabled_openclaw_requires_gateway_url_and_token(flag: str) -> None:
    config = AppConfig()
    setattr(config, flag, True)
    with pytest.raises(ConfigurationError, match="OpenClaw.*gateway URL.*token"):
        validate_config(config)


def test_enabled_openclaw_rejects_invalid_gateway_url() -> None:
    config = AppConfig(
        use_openclaw_tools=True,
        openclaw_gateway_url="not-a-url",
        openclaw_gateway_token="test-token",
    )
    with pytest.raises(ConfigurationError, match=r"HTTP\(S\) gateway URL"):
        validate_config(config)


def test_enabled_openclaw_accepts_complete_gateway_configuration() -> None:
    config = AppConfig(
        use_openclaw_tools=True,
        openclaw_gateway_url="http://127.0.0.1:18789",
        openclaw_gateway_token="test-token",
    )
    validate_config(config)


def test_openclaw_surfaces_are_classified_experimental() -> None:
    text = Path("SURFACE-CLASSIFICATION.md").read_text(encoding="utf-8")
    assert "`experimental`" in text
    assert "`rex-tool-server` | `rex.openclaw.tool_server:main` | `experimental`" in text


def test_gui_labels_openclaw_experimental_and_off_by_default() -> None:
    source = Path("gui/src/pages/SettingsPage.tsx").read_text(encoding="utf-8")
    assert "OpenClaw" in source
    assert "Experimental - off by default" in source
