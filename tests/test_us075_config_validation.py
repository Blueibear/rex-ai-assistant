"""Tests for US-075: Config type validation and coercion.

Verifies that:
- Invalid (non-parseable) string values raise ConfigurationError.
- String-typed numeric values still warn and coerce successfully.
- check_config_types() in doctor returns the correct status.
"""

from __future__ import annotations

import json
import logging

import pytest

from rex.assistant_errors import ConfigurationError
from rex.config import build_app_config
from rex.doctor import Status, check_config_types


def _base_json() -> dict:
    """Minimal valid JSON config with correct numeric types."""
    return {
        "models": {
            "llm_provider": "transformers",
            "llm_model": "sshleifer/tiny-gpt2",
        }
    }


class TestInvalidValueRaisesConfigurationError:
    """_coerce_float/_coerce_int raise ConfigurationError on unparseable strings."""

    def test_invalid_float_raises(self):
        """'abc' for llm_temperature raises ConfigurationError."""
        json_cfg = _base_json()
        json_cfg["models"]["llm_temperature"] = "abc"
        with pytest.raises(ConfigurationError, match="llm_temperature"):
            build_app_config(json_cfg)

    def test_invalid_int_raises(self):
        """'abc' for llm_max_tokens raises ConfigurationError."""
        json_cfg = _base_json()
        json_cfg["models"]["llm_max_tokens"] = "abc"
        with pytest.raises(ConfigurationError, match="llm_max_tokens"):
            build_app_config(json_cfg)

    def test_invalid_float_message_contains_bad_value(self):
        """ConfigurationError message includes the bad field path and raw value."""
        json_cfg = _base_json()
        json_cfg["models"]["llm_temperature"] = "not-a-number"
        with pytest.raises(ConfigurationError) as exc_info:
            build_app_config(json_cfg)
        assert "not-a-number" in str(exc_info.value)


class TestValidStringCoercionStillWorks:
    """Parseable string-typed values still coerce and warn (AC-1, AC-2)."""

    def test_string_float_coerces(self, caplog):
        """'0.7' for llm_temperature coerces to 0.7 with a warning."""
        json_cfg = _base_json()
        json_cfg["models"]["llm_temperature"] = "0.7"
        with caplog.at_level(logging.WARNING, logger="rex.config"):
            cfg = build_app_config(json_cfg)
        assert cfg.llm_temperature == pytest.approx(0.7)
        assert any(
            "llm_temperature" in r.message for r in caplog.records if r.levelno == logging.WARNING
        )

    def test_string_int_coerces(self, caplog):
        """'128' for llm_max_tokens coerces to 128 with a warning."""
        json_cfg = _base_json()
        json_cfg["models"]["llm_max_tokens"] = "128"
        with caplog.at_level(logging.WARNING, logger="rex.config"):
            cfg = build_app_config(json_cfg)
        assert cfg.llm_max_tokens == 128
        assert any(
            "llm_max_tokens" in r.message for r in caplog.records if r.levelno == logging.WARNING
        )


class TestCheckConfigTypesDoctor:
    """check_config_types() returns correct CheckResult status."""

    def test_returns_info_when_root_is_none(self):
        result = check_config_types(None)
        assert result.status == Status.INFO

    def test_returns_ok_for_clean_config(self, tmp_path):
        """A config with correct numeric types returns OK."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        config_file = config_dir / "rex_config.json"
        config_file.write_text(
            json.dumps(
                {
                    "models": {
                        "llm_provider": "transformers",
                        "llm_model": "sshleifer/tiny-gpt2",
                        "llm_temperature": 0.7,
                        "llm_max_tokens": 120,
                    }
                }
            )
        )
        result = check_config_types(tmp_path)
        assert result.status == Status.OK

    def test_returns_warning_for_string_typed_number(self, tmp_path):
        """A config with a string-typed float returns WARNING."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        config_file = config_dir / "rex_config.json"
        config_file.write_text(
            json.dumps(
                {
                    "models": {
                        "llm_provider": "transformers",
                        "llm_model": "sshleifer/tiny-gpt2",
                        "llm_temperature": "0.6",
                    }
                }
            )
        )
        result = check_config_types(tmp_path)
        assert result.status == Status.WARNING
        assert "llm_temperature" in result.details

    def test_returns_error_for_invalid_value(self, tmp_path):
        """A config with an invalid value returns ERROR."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        config_file = config_dir / "rex_config.json"
        config_file.write_text(
            json.dumps(
                {
                    "models": {
                        "llm_provider": "transformers",
                        "llm_model": "sshleifer/tiny-gpt2",
                        "llm_temperature": "not-a-float",
                    }
                }
            )
        )
        result = check_config_types(tmp_path)
        assert result.status == Status.ERROR
