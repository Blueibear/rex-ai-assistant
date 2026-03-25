"""Tests for US-020: string-typed numeric config value coercion warnings.

Verifies that ``build_app_config()`` warns at WARNING level when a numeric
field in rex_config.json carries a string-typed value (e.g. ``"0.6"`` instead
of ``0.6``), while still coercing the value so startup succeeds.

Tests:
- String-typed float logs a warning and coerces correctly.
- String-typed int logs a warning and coerces correctly.
- Correct float type produces no warning.
- Correct int type produces no warning.
- Multiple string-typed fields each produce their own warning.
"""

from __future__ import annotations

import logging

from rex.config import build_app_config


def _base_json() -> dict:
    """Minimal valid JSON config with correct types."""
    return {
        "models": {
            "llm_provider": "transformers",
            "llm_model": "sshleifer/tiny-gpt2",
        }
    }


class TestStringFloatCoercionWarning:
    """build_app_config warns when a float field holds a string value."""

    def test_string_llm_temperature_logs_warning(self, caplog):
        """``"0.6"`` for llm_temperature triggers a WARNING."""
        json_cfg = _base_json()
        json_cfg["models"]["llm_temperature"] = "0.6"

        with caplog.at_level(logging.WARNING, logger="rex.config"):
            cfg = build_app_config(json_cfg)

        assert cfg.llm_temperature == 0.6
        assert any(
            "models.llm_temperature" in r.message and "string" in r.message.lower()
            for r in caplog.records
            if r.levelno == logging.WARNING
        )

    def test_string_llm_top_p_logs_warning(self, caplog):
        """``"0.9"`` for llm_top_p triggers a WARNING."""
        json_cfg = _base_json()
        json_cfg["models"]["llm_top_p"] = "0.9"

        with caplog.at_level(logging.WARNING, logger="rex.config"):
            cfg = build_app_config(json_cfg)

        assert cfg.llm_top_p == 0.9
        assert any(
            "models.llm_top_p" in r.message for r in caplog.records if r.levelno == logging.WARNING
        )

    def test_string_tts_speed_logs_warning(self, caplog):
        """``"1.08"`` for tts_speed triggers a WARNING."""
        json_cfg = _base_json()
        json_cfg["models"] = dict(json_cfg["models"])
        json_cfg["models"]["tts_speed"] = "1.08"

        with caplog.at_level(logging.WARNING, logger="rex.config"):
            cfg = build_app_config(json_cfg)

        assert cfg.tts_speed == 1.08
        assert any(
            "models.tts_speed" in r.message for r in caplog.records if r.levelno == logging.WARNING
        )


class TestStringIntCoercionWarning:
    """build_app_config warns when an int field holds a string value."""

    def test_string_llm_max_tokens_logs_warning(self, caplog):
        """``"120"`` for llm_max_tokens triggers a WARNING."""
        json_cfg = _base_json()
        json_cfg["models"]["llm_max_tokens"] = "120"

        with caplog.at_level(logging.WARNING, logger="rex.config"):
            cfg = build_app_config(json_cfg)

        assert cfg.llm_max_tokens == 120
        assert any(
            "models.llm_max_tokens" in r.message
            for r in caplog.records
            if r.levelno == logging.WARNING
        )

    def test_string_sample_rate_logs_warning(self, caplog):
        """``"16000"`` for sample_rate triggers a WARNING."""
        json_cfg = _base_json()
        json_cfg["audio"] = {"sample_rate": "16000"}

        with caplog.at_level(logging.WARNING, logger="rex.config"):
            cfg = build_app_config(json_cfg)

        assert cfg.sample_rate == 16000
        assert any(
            "audio.sample_rate" in r.message for r in caplog.records if r.levelno == logging.WARNING
        )


class TestCorrectTypesProduceNoWarning:
    """build_app_config does NOT warn when numeric fields have correct types."""

    def test_float_temperature_no_warning(self, caplog):
        """Correct float 0.6 produces no coercion warning."""
        json_cfg = _base_json()
        json_cfg["models"]["llm_temperature"] = 0.6

        with caplog.at_level(logging.WARNING, logger="rex.config"):
            cfg = build_app_config(json_cfg)

        assert cfg.llm_temperature == 0.6
        assert not any(
            "llm_temperature" in r.message for r in caplog.records if r.levelno == logging.WARNING
        )

    def test_int_max_tokens_no_warning(self, caplog):
        """Correct int 120 produces no coercion warning."""
        json_cfg = _base_json()
        json_cfg["models"]["llm_max_tokens"] = 120

        with caplog.at_level(logging.WARNING, logger="rex.config"):
            cfg = build_app_config(json_cfg)

        assert cfg.llm_max_tokens == 120
        assert not any(
            "llm_max_tokens" in r.message for r in caplog.records if r.levelno == logging.WARNING
        )

    def test_default_values_produce_no_warning(self, caplog):
        """When no numeric fields are specified, defaults are used without warnings."""
        json_cfg = _base_json()

        with caplog.at_level(logging.WARNING, logger="rex.config"):
            build_app_config(json_cfg)

        coercion_warnings = [
            r for r in caplog.records if r.levelno == logging.WARNING and "coercing" in r.message
        ]
        assert coercion_warnings == []


class TestMultipleStringFieldsEachWarn:
    """Each string-typed numeric field generates its own warning."""

    def test_two_string_fields_produce_two_warnings(self, caplog):
        """Setting two float fields as strings generates two separate warnings."""
        json_cfg = _base_json()
        json_cfg["models"]["llm_temperature"] = "0.6"
        json_cfg["models"]["llm_top_p"] = "0.9"

        with caplog.at_level(logging.WARNING, logger="rex.config"):
            cfg = build_app_config(json_cfg)

        assert cfg.llm_temperature == 0.6
        assert cfg.llm_top_p == 0.9

        warned_paths = {
            r.message
            for r in caplog.records
            if r.levelno == logging.WARNING and "coercing" in r.message
        }
        assert any("llm_temperature" in w for w in warned_paths)
        assert any("llm_top_p" in w for w in warned_paths)
