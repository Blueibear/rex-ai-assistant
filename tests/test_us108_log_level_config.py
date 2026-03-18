"""Tests for US-108: Log level configuration per environment.

Verifies:
- log level configurable via LOG_LEVEL env var
- default log level is INFO when LOG_LEVEL is not set
- per-module log level overrides supported via config
- DEBUG-level logs do not appear in output when LOG_LEVEL=INFO
- Typecheck passes (verified by mypy in CI)
"""

from __future__ import annotations

import logging
import os
from io import StringIO
from unittest.mock import patch

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


def _fresh_root() -> logging.Logger:
    """Strip all handlers from root logger so configure_logging() runs fresh."""
    root = logging.getLogger()
    for h in root.handlers[:]:
        root.removeHandler(h)
    return root


def _stream_configure(level: int | None = None) -> tuple[logging.Logger, StringIO]:
    """Configure logging with a StringIO stream and return (root, stream)."""
    root = _fresh_root()
    stream = StringIO()
    configure_logging(level=level, handlers=[logging.StreamHandler(stream)])
    return root, stream


def _cleanup_root() -> None:
    root = logging.getLogger()
    for h in root.handlers[:]:
        root.removeHandler(h)


# ---------------------------------------------------------------------------
# _LEVEL_NAMES
# ---------------------------------------------------------------------------


class TestLevelNames:
    def test_debug_mapped(self) -> None:
        assert _LEVEL_NAMES["DEBUG"] == logging.DEBUG

    def test_info_mapped(self) -> None:
        assert _LEVEL_NAMES["INFO"] == logging.INFO

    def test_warning_mapped(self) -> None:
        assert _LEVEL_NAMES["WARNING"] == logging.WARNING

    def test_warn_alias(self) -> None:
        assert _LEVEL_NAMES["WARN"] == logging.WARNING

    def test_error_mapped(self) -> None:
        assert _LEVEL_NAMES["ERROR"] == logging.ERROR

    def test_critical_mapped(self) -> None:
        assert _LEVEL_NAMES["CRITICAL"] == logging.CRITICAL


# ---------------------------------------------------------------------------
# _env_log_level
# ---------------------------------------------------------------------------


class TestEnvLogLevel:
    def test_default_is_info_when_no_env_var(self) -> None:
        env = {k: v for k, v in os.environ.items() if k not in ("LOG_LEVEL", "REX_LOG_LEVEL")}
        with patch.dict(os.environ, env, clear=True):
            assert _env_log_level() == logging.INFO

    def test_log_level_debug(self) -> None:
        with patch.dict(os.environ, {"LOG_LEVEL": "DEBUG"}, clear=False):
            assert _env_log_level() == logging.DEBUG

    def test_log_level_warning(self) -> None:
        with patch.dict(os.environ, {"LOG_LEVEL": "WARNING"}, clear=False):
            assert _env_log_level() == logging.WARNING

    def test_log_level_error(self) -> None:
        with patch.dict(os.environ, {"LOG_LEVEL": "ERROR"}, clear=False):
            assert _env_log_level() == logging.ERROR

    def test_log_level_critical(self) -> None:
        with patch.dict(os.environ, {"LOG_LEVEL": "CRITICAL"}, clear=False):
            assert _env_log_level() == logging.CRITICAL

    def test_log_level_case_insensitive(self) -> None:
        with patch.dict(os.environ, {"LOG_LEVEL": "debug"}, clear=False):
            assert _env_log_level() == logging.DEBUG

    def test_log_level_mixed_case(self) -> None:
        with patch.dict(os.environ, {"LOG_LEVEL": "Warning"}, clear=False):
            assert _env_log_level() == logging.WARNING

    def test_rex_log_level_legacy_var(self) -> None:
        env = {k: v for k, v in os.environ.items() if k not in ("LOG_LEVEL", "REX_LOG_LEVEL")}
        env["REX_LOG_LEVEL"] = "DEBUG"
        with patch.dict(os.environ, env, clear=True):
            assert _env_log_level() == logging.DEBUG

    def test_log_level_takes_precedence_over_rex_log_level(self) -> None:
        with patch.dict(
            os.environ,
            {"LOG_LEVEL": "ERROR", "REX_LOG_LEVEL": "DEBUG"},
            clear=False,
        ):
            assert _env_log_level() == logging.ERROR

    def test_invalid_log_level_falls_through_to_info(self) -> None:
        env = {k: v for k, v in os.environ.items() if k not in ("LOG_LEVEL", "REX_LOG_LEVEL")}
        env["LOG_LEVEL"] = "VERBOSE"
        with patch.dict(os.environ, env, clear=True):
            assert _env_log_level() == logging.INFO

    def test_empty_log_level_falls_through_to_info(self) -> None:
        env = {k: v for k, v in os.environ.items() if k not in ("LOG_LEVEL", "REX_LOG_LEVEL")}
        env["LOG_LEVEL"] = ""
        with patch.dict(os.environ, env, clear=True):
            assert _env_log_level() == logging.INFO


# ---------------------------------------------------------------------------
# _env_module_levels
# ---------------------------------------------------------------------------


class TestEnvModuleLevels:
    def test_empty_when_no_overrides(self) -> None:
        env = {k: v for k, v in os.environ.items() if not k.startswith("LOG_LEVEL_")}
        with patch.dict(os.environ, env, clear=True):
            assert _env_module_levels() == {}

    def test_single_module_override(self) -> None:
        with patch.dict(os.environ, {"LOG_LEVEL_rex.retry": "DEBUG"}, clear=False):
            levels = _env_module_levels()
            assert levels.get("rex.retry") == logging.DEBUG

    def test_multiple_module_overrides(self) -> None:
        with patch.dict(
            os.environ,
            {
                "LOG_LEVEL_rex.llm_client": "WARNING",
                "LOG_LEVEL_rex.ha_bridge": "ERROR",
            },
            clear=False,
        ):
            levels = _env_module_levels()
            assert levels.get("rex.llm_client") == logging.WARNING
            assert levels.get("rex.ha_bridge") == logging.ERROR

    def test_invalid_level_ignored(self) -> None:
        with patch.dict(os.environ, {"LOG_LEVEL_rex.bad": "VERBOSE"}, clear=False):
            levels = _env_module_levels()
            assert "rex.bad" not in levels

    def test_level_case_insensitive(self) -> None:
        with patch.dict(os.environ, {"LOG_LEVEL_rex.test": "debug"}, clear=False):
            levels = _env_module_levels()
            assert levels.get("rex.test") == logging.DEBUG

    def test_empty_module_name_ignored(self) -> None:
        # LOG_LEVEL_ with no suffix after prefix
        with patch.dict(os.environ, {"LOG_LEVEL_": "DEBUG"}, clear=False):
            levels = _env_module_levels()
            assert "" not in levels


# ---------------------------------------------------------------------------
# apply_module_log_levels
# ---------------------------------------------------------------------------


class TestApplyModuleLogLevels:
    def test_sets_logger_level_by_int(self) -> None:
        apply_module_log_levels({"rex.test.module_a": logging.DEBUG})
        assert logging.getLogger("rex.test.module_a").level == logging.DEBUG

    def test_sets_logger_level_by_string(self) -> None:
        apply_module_log_levels({"rex.test.module_b": "WARNING"})
        assert logging.getLogger("rex.test.module_b").level == logging.WARNING

    def test_string_case_insensitive(self) -> None:
        apply_module_log_levels({"rex.test.module_c": "error"})
        assert logging.getLogger("rex.test.module_c").level == logging.ERROR

    def test_multiple_modules(self) -> None:
        apply_module_log_levels(
            {
                "rex.test.mod_d": logging.DEBUG,
                "rex.test.mod_e": logging.CRITICAL,
            }
        )
        assert logging.getLogger("rex.test.mod_d").level == logging.DEBUG
        assert logging.getLogger("rex.test.mod_e").level == logging.CRITICAL

    def test_unknown_string_defaults_to_info(self) -> None:
        apply_module_log_levels({"rex.test.module_f": "VERBOSE"})
        assert logging.getLogger("rex.test.module_f").level == logging.INFO


# ---------------------------------------------------------------------------
# configure_logging integration — level behaviour
# ---------------------------------------------------------------------------


class TestConfigureLoggingLevel:
    def teardown_method(self) -> None:
        _cleanup_root()

    def test_default_level_is_info(self) -> None:
        env = {k: v for k, v in os.environ.items() if k not in ("LOG_LEVEL", "REX_LOG_LEVEL")}
        with patch.dict(os.environ, env, clear=True):
            root, _ = _stream_configure()
        assert root.level == logging.INFO

    def test_log_level_env_sets_root_level(self) -> None:
        with patch.dict(os.environ, {"LOG_LEVEL": "DEBUG"}, clear=False):
            root, _ = _stream_configure()
        assert root.level == logging.DEBUG

    def test_explicit_level_overrides_env(self) -> None:
        with patch.dict(os.environ, {"LOG_LEVEL": "DEBUG"}, clear=False):
            root, _ = _stream_configure(level=logging.ERROR)
        assert root.level == logging.ERROR

    def test_debug_logs_suppressed_at_info(self) -> None:
        """DEBUG records must not appear when root level is INFO."""
        env = {k: v for k, v in os.environ.items() if k not in ("LOG_LEVEL", "REX_LOG_LEVEL")}
        with patch.dict(os.environ, env, clear=True):
            root, stream = _stream_configure()
        root.setLevel(logging.INFO)
        logger = logging.getLogger("rex.test.info_only")
        logger.debug("should not appear")
        logger.info("should appear")
        output = stream.getvalue()
        assert "should not appear" not in output
        assert "should appear" in output

    def test_debug_logs_visible_at_debug_level(self) -> None:
        with patch.dict(os.environ, {"LOG_LEVEL": "DEBUG"}, clear=False):
            _, stream = _stream_configure()
        logger = logging.getLogger("rex.test.debug_visible")
        logger.debug("debug visible message")
        output = stream.getvalue()
        assert "debug visible message" in output

    def test_per_module_override_via_kwarg(self) -> None:
        """module_levels kwarg sets per-logger level after root is configured."""
        _, stream = _stream_configure(level=logging.INFO)
        from rex.logging_utils import apply_module_log_levels

        apply_module_log_levels({"rex.test.override_mod": logging.DEBUG})
        logger = logging.getLogger("rex.test.override_mod")
        # DEBUG messages should propagate through the root INFO handler
        logger.debug("module debug message")
        output = stream.getvalue()
        assert "module debug message" in output

    def test_per_module_override_via_env(self) -> None:
        """LOG_LEVEL_<module>=DEBUG lets a specific module emit DEBUG."""
        with patch.dict(
            os.environ,
            {"LOG_LEVEL": "INFO", "LOG_LEVEL_rex.test.envmod": "DEBUG"},
            clear=False,
        ):
            _, stream = _stream_configure()

        logger = logging.getLogger("rex.test.envmod")
        logger.debug("envmod debug message")
        output = stream.getvalue()
        assert "envmod debug message" in output
