"""Smoke tests for rex.logging_utils."""

from __future__ import annotations

import logging
from io import StringIO
from pathlib import Path
from types import SimpleNamespace


def test_import():
    """Module imports without error."""
    import rex.logging_utils as lu

    assert lu is not None


def test_json_formatter_basic():
    """JsonFormatter produces valid JSON with expected keys."""
    import json

    from rex.logging_utils import JsonFormatter

    formatter = JsonFormatter()
    record = logging.LogRecord(
        name="test.logger",
        level=logging.INFO,
        pathname="test.py",
        lineno=1,
        msg="hello world",
        args=(),
        exc_info=None,
    )
    output = formatter.format(record)
    parsed = json.loads(output)

    assert parsed["level"] == "INFO"
    assert parsed["logger"] == "test.logger"
    assert parsed["message"] == "hello world"
    assert "timestamp" in parsed


def test_json_formatter_with_exception():
    """JsonFormatter includes exception key when exc_info is set."""
    import json

    from rex.logging_utils import JsonFormatter

    formatter = JsonFormatter()
    try:
        raise ValueError("oops")
    except ValueError:
        import sys

        exc_info = sys.exc_info()

    record = logging.LogRecord(
        name="test.logger",
        level=logging.ERROR,
        pathname="test.py",
        lineno=1,
        msg="error",
        args=(),
        exc_info=exc_info,
    )
    output = formatter.format(record)
    parsed = json.loads(output)

    assert "exception" in parsed
    assert "ValueError" in parsed["exception"]


def test_env_log_level_default(monkeypatch):
    """_env_log_level returns INFO when no env var is set."""
    from rex.logging_utils import _env_log_level

    monkeypatch.delenv("LOG_LEVEL", raising=False)
    monkeypatch.delenv("REX_LOG_LEVEL", raising=False)

    assert _env_log_level() == logging.INFO


def test_env_log_level_from_env(monkeypatch):
    """_env_log_level reads from LOG_LEVEL env var."""
    from rex.logging_utils import _env_log_level

    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.delenv("REX_LOG_LEVEL", raising=False)

    assert _env_log_level() == logging.DEBUG


def test_env_log_level_legacy(monkeypatch):
    """_env_log_level falls back to REX_LOG_LEVEL."""
    from rex.logging_utils import _env_log_level

    monkeypatch.delenv("LOG_LEVEL", raising=False)
    monkeypatch.setenv("REX_LOG_LEVEL", "WARNING")

    assert _env_log_level() == logging.WARNING


def test_env_log_level_invalid_ignored(monkeypatch):
    """_env_log_level ignores unknown level names and defaults to INFO."""
    from rex.logging_utils import _env_log_level

    monkeypatch.setenv("LOG_LEVEL", "NONSENSE")
    monkeypatch.delenv("REX_LOG_LEVEL", raising=False)

    assert _env_log_level() == logging.INFO


def test_env_module_levels(monkeypatch):
    """_env_module_levels parses LOG_LEVEL_<module>=<level> env vars."""
    from rex.logging_utils import _env_module_levels

    monkeypatch.setenv("LOG_LEVEL_rex.http_errors", "DEBUG")
    monkeypatch.setenv("LOG_LEVEL_rex.llm_client", "WARNING")

    result = _env_module_levels()

    assert result.get("rex.http_errors") == logging.DEBUG
    assert result.get("rex.llm_client") == logging.WARNING


def test_apply_module_log_levels():
    """apply_module_log_levels sets individual logger levels."""
    from rex.logging_utils import apply_module_log_levels

    apply_module_log_levels({"rex.test_module_x": logging.DEBUG})

    assert logging.getLogger("rex.test_module_x").level == logging.DEBUG


def test_apply_module_log_levels_string():
    """apply_module_log_levels accepts string level names."""
    from rex.logging_utils import apply_module_log_levels

    apply_module_log_levels({"rex.test_module_y": "ERROR"})

    assert logging.getLogger("rex.test_module_y").level == logging.ERROR


def test_set_global_level():
    """set_global_level updates root logger level."""
    from rex.logging_utils import set_global_level

    set_global_level(logging.WARNING)
    assert logging.getLogger().level == logging.WARNING

    # Restore
    set_global_level(logging.INFO)


def test_active_runtime_log_path_defaults_to_data_logs(monkeypatch):
    """Current session logs default to data/logs, not the legacy logs directory."""
    from rex.log_paths import PROJECT_ROOT, active_runtime_log_path

    monkeypatch.delenv("REX_LOG_PATH", raising=False)

    assert active_runtime_log_path().relative_to(PROJECT_ROOT) == Path("data/logs/rex.log")


def test_configure_logging_adds_file_handler_after_stream_configured(tmp_path, monkeypatch):
    """File logging is added even if stdout logging was configured before settings loaded."""
    import json

    import rex.logging_utils as lu

    root = logging.getLogger()
    original_handlers = root.handlers[:]
    original_level = root.level
    log_path = tmp_path / "rex.log"
    error_path = tmp_path / "error.log"

    try:
        for handler in root.handlers[:]:
            root.removeHandler(handler)
        root.addHandler(logging.StreamHandler(StringIO()))
        root.setLevel(logging.INFO)
        monkeypatch.setattr(
            lu,
            "settings",
            SimpleNamespace(
                file_logging_enabled=True,
                log_path=log_path,
                error_log_path=error_path,
            ),
        )
        monkeypatch.setenv("REX_JSON_LOGS", "1")

        lu.configure_logging()
        logging.getLogger("test.logging_utils.current").info("current session message")
        for handler in root.handlers:
            handler.flush()

        lines = log_path.read_text(encoding="utf-8").strip().splitlines()
        assert any("Rex Python runtime session started" in line for line in lines)
        parsed = json.loads(lines[-1])
        assert parsed["message"] == "current session message"
        assert parsed["timestamp"].endswith("+00:00")
    finally:
        root.handlers = original_handlers
        root.setLevel(original_level)


def test_json_logging_enabled_in_test(monkeypatch):
    """_json_logging_enabled returns False inside pytest (PYTEST_CURRENT_TEST set)."""
    from rex.logging_utils import _json_logging_enabled

    monkeypatch.delenv("REX_JSON_LOGS", raising=False)
    # PYTEST_CURRENT_TEST is always set when running under pytest
    assert not _json_logging_enabled()


def test_json_logging_enabled_explicit_on(monkeypatch):
    """_json_logging_enabled returns True when REX_JSON_LOGS=1."""
    from rex.logging_utils import _json_logging_enabled

    monkeypatch.setenv("REX_JSON_LOGS", "1")
    assert _json_logging_enabled()


def test_json_logging_enabled_explicit_off(monkeypatch):
    """_json_logging_enabled returns False when REX_JSON_LOGS=0."""
    from rex.logging_utils import _json_logging_enabled

    monkeypatch.setenv("REX_JSON_LOGS", "0")
    assert not _json_logging_enabled()
