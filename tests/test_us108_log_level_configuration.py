"""Tests for US-108: Log level configuration per environment.

Acceptance criteria:
- log level configurable via environment variable (e.g., LOG_LEVEL=DEBUG)
- default log level is INFO when LOG_LEVEL is not set
- per-module log level overrides supported via config
- DEBUG-level logs do not appear in output when LOG_LEVEL=INFO
- Typecheck passes
"""

from __future__ import annotations

import logging
import os
from io import StringIO
from unittest.mock import patch

import pytest

from rex.logging_utils import (
    _LEVEL_NAMES,
    _env_log_level,
    _env_module_levels,
    apply_module_log_levels,
    configure_logging,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fresh_root_logger() -> logging.Logger:
    """Strip all handlers from root logger so configure_logging() runs fresh."""
    root = logging.getLogger()
    for h in root.handlers[:]:
        root.removeHandler(h)
    return root


def _make_stream_handler() -> tuple[logging.StreamHandler, StringIO]:  # type: ignore[type-arg]
    buf = StringIO()
    return logging.StreamHandler(buf), buf


# ---------------------------------------------------------------------------
# _env_log_level
# ---------------------------------------------------------------------------


class TestEnvLogLevel:
    def test_default_is_info(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("LOG_LEVEL", raising=False)
        monkeypatch.delenv("REX_LOG_LEVEL", raising=False)
        assert _env_log_level() == logging.INFO

    def test_log_level_debug(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")
        monkeypatch.delenv("REX_LOG_LEVEL", raising=False)
        assert _env_log_level() == logging.DEBUG

    def test_log_level_warning(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LOG_LEVEL", "WARNING")
        assert _env_log_level() == logging.WARNING

    def test_log_level_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LOG_LEVEL", "ERROR")
        assert _env_log_level() == logging.ERROR

    def test_log_level_critical(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LOG_LEVEL", "CRITICAL")
        assert _env_log_level() == logging.CRITICAL

    def test_log_level_case_insensitive(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LOG_LEVEL", "debug")
        assert _env_log_level() == logging.DEBUG

    def test_log_level_mixed_case(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LOG_LEVEL", "Warning")
        assert _env_log_level() == logging.WARNING

    def test_log_level_warn_alias(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LOG_LEVEL", "WARN")
        assert _env_log_level() == logging.WARNING

    def test_log_level_unknown_falls_back_to_info(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LOG_LEVEL", "VERBOSE")
        monkeypatch.delenv("REX_LOG_LEVEL", raising=False)
        assert _env_log_level() == logging.INFO

    def test_rex_log_level_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("LOG_LEVEL", raising=False)
        monkeypatch.setenv("REX_LOG_LEVEL", "DEBUG")
        assert _env_log_level() == logging.DEBUG

    def test_log_level_takes_precedence_over_rex_log_level(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LOG_LEVEL", "ERROR")
        monkeypatch.setenv("REX_LOG_LEVEL", "DEBUG")
        assert _env_log_level() == logging.ERROR

    def test_level_names_dict_contains_all_standard_levels(self) -> None:
        for name in ("DEBUG", "INFO", "WARNING", "WARN", "ERROR", "CRITICAL"):
            assert name in _LEVEL_NAMES


# ---------------------------------------------------------------------------
# configure_logging — respects LOG_LEVEL env var
# ---------------------------------------------------------------------------


class TestConfigureLoggingLevel:
    def test_default_level_is_info_when_env_not_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("LOG_LEVEL", raising=False)
        monkeypatch.delenv("REX_LOG_LEVEL", raising=False)
        root = _fresh_root_logger()
        handler, _ = _make_stream_handler()
        with patch("rex.logging_utils._json_logging_enabled", return_value=False):
            configure_logging(handlers=[handler])
        assert root.level == logging.INFO
        for h in root.handlers[:]:
            root.removeHandler(h)

    def test_level_from_log_level_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")
        root = _fresh_root_logger()
        handler, _ = _make_stream_handler()
        with patch("rex.logging_utils._json_logging_enabled", return_value=False):
            configure_logging(handlers=[handler])
        assert root.level == logging.DEBUG
        for h in root.handlers[:]:
            root.removeHandler(h)

    def test_explicit_level_overrides_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")
        root = _fresh_root_logger()
        handler, _ = _make_stream_handler()
        with patch("rex.logging_utils._json_logging_enabled", return_value=False):
            configure_logging(level=logging.ERROR, handlers=[handler])
        assert root.level == logging.ERROR
        for h in root.handlers[:]:
            root.removeHandler(h)

    def test_debug_messages_suppressed_at_info_level(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("LOG_LEVEL", raising=False)
        monkeypatch.delenv("REX_LOG_LEVEL", raising=False)
        root = _fresh_root_logger()
        handler, buf = _make_stream_handler()
        with patch("rex.logging_utils._json_logging_enabled", return_value=False):
            configure_logging(level=logging.INFO, handlers=[handler])
        logger = logging.getLogger("rex.test.suppress_debug")
        logger.debug("this should not appear")
        logger.info("this should appear")
        output = buf.getvalue()
        assert "this should not appear" not in output
        assert "this should appear" in output
        for h in root.handlers[:]:
            root.removeHandler(h)

    def test_debug_messages_visible_at_debug_level(self, monkeypatch: pytest.MonkeyPatch) -> None:
        root = _fresh_root_logger()
        handler, buf = _make_stream_handler()
        with patch("rex.logging_utils._json_logging_enabled", return_value=False):
            configure_logging(level=logging.DEBUG, handlers=[handler])
        logger = logging.getLogger("rex.test.show_debug")
        logger.debug("debug message visible")
        output = buf.getvalue()
        assert "debug message visible" in output
        for h in root.handlers[:]:
            root.removeHandler(h)

    def test_debug_suppressed_when_log_level_env_is_info(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LOG_LEVEL", "INFO")
        root = _fresh_root_logger()
        handler, buf = _make_stream_handler()
        with patch("rex.logging_utils._json_logging_enabled", return_value=False):
            configure_logging(handlers=[handler])
        logger = logging.getLogger("rex.test.env_info_suppress_debug")
        logger.debug("suppressed debug")
        logger.warning("visible warning")
        output = buf.getvalue()
        assert "suppressed debug" not in output
        assert "visible warning" in output
        for h in root.handlers[:]:
            root.removeHandler(h)


# ---------------------------------------------------------------------------
# _env_module_levels — parse LOG_LEVEL_<module> env vars
# ---------------------------------------------------------------------------


class TestEnvModuleLevels:
    def test_empty_when_no_env_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        env = {k: v for k, v in os.environ.items() if not k.startswith("LOG_LEVEL_")}
        with patch.dict(os.environ, env, clear=True):
            result = _env_module_levels()
        assert result == {}

    def test_single_module_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LOG_LEVEL_rex.http_errors", "DEBUG")
        result = _env_module_levels()
        assert result.get("rex.http_errors") == logging.DEBUG

    def test_multiple_module_overrides(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LOG_LEVEL_rex.llm_client", "WARNING")
        monkeypatch.setenv("LOG_LEVEL_rex.dashboard", "ERROR")
        result = _env_module_levels()
        assert result["rex.llm_client"] == logging.WARNING
        assert result["rex.dashboard"] == logging.ERROR

    def test_case_insensitive_level(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LOG_LEVEL_rex.foo", "debug")
        result = _env_module_levels()
        assert result.get("rex.foo") == logging.DEBUG

    def test_unknown_level_ignored(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LOG_LEVEL_rex.bar", "VERBOSE")
        result = _env_module_levels()
        assert "rex.bar" not in result


# ---------------------------------------------------------------------------
# apply_module_log_levels
# ---------------------------------------------------------------------------


class TestApplyModuleLogLevels:
    def test_sets_logger_level_by_name(self) -> None:
        apply_module_log_levels({"rex.test.module_override": logging.DEBUG})
        logger = logging.getLogger("rex.test.module_override")
        assert logger.level == logging.DEBUG
        logger.setLevel(logging.NOTSET)  # cleanup

    def test_sets_logger_level_by_string(self) -> None:
        apply_module_log_levels({"rex.test.module_override_str": "WARNING"})
        logger = logging.getLogger("rex.test.module_override_str")
        assert logger.level == logging.WARNING
        logger.setLevel(logging.NOTSET)  # cleanup

    def test_multiple_modules(self) -> None:
        apply_module_log_levels(
            {
                "rex.test.mod_a": logging.DEBUG,
                "rex.test.mod_b": logging.ERROR,
            }
        )
        assert logging.getLogger("rex.test.mod_a").level == logging.DEBUG
        assert logging.getLogger("rex.test.mod_b").level == logging.ERROR
        logging.getLogger("rex.test.mod_a").setLevel(logging.NOTSET)
        logging.getLogger("rex.test.mod_b").setLevel(logging.NOTSET)

    def test_unknown_string_level_defaults_to_info(self) -> None:
        apply_module_log_levels({"rex.test.unknown_level": "VERBOSE"})
        logger = logging.getLogger("rex.test.unknown_level")
        assert logger.level == logging.INFO
        logger.setLevel(logging.NOTSET)  # cleanup

    def test_module_level_independent_of_root(self) -> None:
        """A DEBUG module level should emit debug even if root is INFO."""
        root = _fresh_root_logger()
        handler, buf = _make_stream_handler()
        with patch("rex.logging_utils._json_logging_enabled", return_value=False):
            configure_logging(level=logging.INFO, handlers=[handler])

        apply_module_log_levels({"rex.test.debug_module": logging.DEBUG})
        logger = logging.getLogger("rex.test.debug_module")
        logger.debug("module-level debug message")
        output = buf.getvalue()
        assert "module-level debug message" in output

        # Cleanup
        logging.getLogger("rex.test.debug_module").setLevel(logging.NOTSET)
        for h in root.handlers[:]:
            root.removeHandler(h)


# ---------------------------------------------------------------------------
# configure_logging — module_levels parameter
# ---------------------------------------------------------------------------


class TestConfigureLoggingModuleLevels:
    def test_module_levels_applied_via_parameter(self) -> None:
        root = _fresh_root_logger()
        handler, buf = _make_stream_handler()
        with patch("rex.logging_utils._json_logging_enabled", return_value=False):
            configure_logging(
                level=logging.INFO,
                handlers=[handler],
                module_levels={"rex.test.param_override": logging.DEBUG},
            )
        logger = logging.getLogger("rex.test.param_override")
        assert logger.level == logging.DEBUG
        # Cleanup
        logger.setLevel(logging.NOTSET)
        for h in root.handlers[:]:
            root.removeHandler(h)

    def test_module_levels_from_env_applied_automatically(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LOG_LEVEL_rex.test.env_override", "DEBUG")
        root = _fresh_root_logger()
        handler, buf = _make_stream_handler()
        with patch("rex.logging_utils._json_logging_enabled", return_value=False):
            configure_logging(level=logging.INFO, handlers=[handler])
        logger = logging.getLogger("rex.test.env_override")
        assert logger.level == logging.DEBUG
        # Cleanup
        logger.setLevel(logging.NOTSET)
        for h in root.handlers[:]:
            root.removeHandler(h)
