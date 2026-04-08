"""Tests for US-071: Debug mode toggle."""

from __future__ import annotations

import logging

import pytest

# rex.cli has a module-level Python version guard that raises SystemExit on
# non-3.11 interpreters.  Skip CLI-import tests when on an unsupported version.
try:
    import rex.cli as _rex_cli  # noqa: F401

    _CLI_AVAILABLE = True
except SystemExit:
    _CLI_AVAILABLE = False

requires_cli = pytest.mark.skipif(
    not _CLI_AVAILABLE,
    reason="rex.cli unavailable on this Python version",
)


def test_appconfig_debug_mode_field():
    """AppConfig must have a debug_mode field."""
    from rex.config import AppConfig

    cfg = AppConfig()
    assert hasattr(cfg, "debug_mode")
    assert cfg.debug_mode is False


def test_appconfig_debug_mode_from_env(monkeypatch):
    """debug_mode must be True when REX_DEBUG=1 is set."""
    monkeypatch.setenv("REX_DEBUG", "1")
    import rex.config as _cfg_mod

    monkeypatch.setattr(_cfg_mod, "_cached_config", None)
    cfg = _cfg_mod.load_config()
    assert cfg.debug_mode is True


def test_appconfig_debug_mode_off_by_default(monkeypatch):
    """debug_mode must be False when REX_DEBUG is absent."""
    monkeypatch.delenv("REX_DEBUG", raising=False)
    import rex.config as _cfg_mod

    monkeypatch.setattr(_cfg_mod, "_cached_config", None)
    cfg = _cfg_mod.load_config()
    assert cfg.debug_mode is False


def test_appconfig_debug_mode_false_string(monkeypatch):
    """REX_DEBUG=false must leave debug_mode as False."""
    monkeypatch.setenv("REX_DEBUG", "false")
    import rex.config as _cfg_mod

    monkeypatch.setattr(_cfg_mod, "_cached_config", None)
    cfg = _cfg_mod.load_config()
    assert cfg.debug_mode is False


@requires_cli
def test_cli_debug_flag_exists():
    """rex --debug must be a recognised argument (no parser error)."""
    from rex.cli import create_parser

    parser = create_parser()
    args = parser.parse_args(["--debug", "doctor"])
    assert args.debug is True


@requires_cli
def test_cli_debug_flag_default_false():
    """debug flag must default to False when not supplied."""
    from rex.cli import create_parser

    parser = create_parser()
    args = parser.parse_args(["doctor"])
    assert getattr(args, "debug", False) is False


@requires_cli
def test_cli_doctor_debug_flag_exists():
    """rex doctor --debug must be a recognised argument."""
    from rex.cli import create_parser

    parser = create_parser()
    args = parser.parse_args(["doctor", "--debug"])
    assert args.debug is True


def test_run_diagnostics_accepts_debug(monkeypatch, capsys):
    """run_diagnostics(debug=True) must not raise and must print debug header."""
    from rex.doctor import run_diagnostics

    monkeypatch.setattr("rex.doctor.check_stt_warmup", lambda: _ok("STT warmup", "skipped"))
    monkeypatch.setattr(
        "rex.doctor.check_lm_studio_reachability", lambda: _ok("LM Studio", "skipped")
    )

    run_diagnostics(verbose=False, debug=True)
    out = capsys.readouterr().out
    assert "DEBUG INFO" in out


def test_run_diagnostics_debug_shows_log_level(monkeypatch, capsys):
    """Debug output must include current root log level."""
    from rex.doctor import run_diagnostics

    monkeypatch.setattr("rex.doctor.check_stt_warmup", lambda: _ok("STT warmup", "skipped"))
    monkeypatch.setattr(
        "rex.doctor.check_lm_studio_reachability", lambda: _ok("LM Studio", "skipped")
    )

    run_diagnostics(verbose=False, debug=True)
    out = capsys.readouterr().out
    assert "log level" in out.lower() or "Root log level" in out


@requires_cli
def test_debug_mode_sets_log_level(monkeypatch):
    """Activating debug mode via main() must set root logger to DEBUG."""
    import rex.cli as cli_mod

    monkeypatch.setenv("REX_DEBUG", "0")

    captured_level = []

    def fake_doctor(args):
        captured_level.append(logging.getLogger().getEffectiveLevel())
        return 0

    monkeypatch.setattr(cli_mod, "cmd_doctor", fake_doctor)

    cli_mod.main(["--debug", "doctor"])
    assert captured_level and captured_level[0] == logging.DEBUG


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ok(name: str, message: str):
    from rex.doctor import CheckResult, Status

    return CheckResult(name=name, status=Status.OK, message=message)
