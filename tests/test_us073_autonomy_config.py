from __future__ import annotations

import pytest

from rex.assistant_errors import ConfigurationError
from rex.config import build_app_config, validate_config


def test_runtime_config_reads_canonical_autonomy_mode() -> None:
    config = build_app_config({"models": {"autonomy_mode": "supervised"}})

    assert config.autonomy_mode == "supervised"
    validate_config(config)


def test_runtime_config_defaults_autonomy_mode_to_manual() -> None:
    config = build_app_config({})

    assert config.autonomy_mode == "manual"
    validate_config(config)


def test_runtime_config_rejects_unknown_autonomy_mode() -> None:
    config = build_app_config({"models": {"autonomy_mode": "unbounded"}})

    with pytest.raises(ConfigurationError, match="autonomy_mode"):
        validate_config(config)


@pytest.mark.parametrize("value", [["full-auto"], {"mode": "full-auto"}])
def test_runtime_config_rejects_non_string_autonomy_mode(value: object) -> None:
    config = build_app_config({"models": {"autonomy_mode": value}})

    with pytest.raises(ConfigurationError, match="autonomy_mode"):
        validate_config(config)
