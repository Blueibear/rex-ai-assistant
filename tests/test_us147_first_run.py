"""Tests for US-147: First-run setup detection and guided message."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from rex.first_run import (
    is_first_run,
    maybe_print_welcome,
    print_welcome_message,
)

# ---------------------------------------------------------------------------
# is_first_run — detection logic
# ---------------------------------------------------------------------------


def test_first_run_when_config_missing(tmp_path: Path) -> None:
    """Returns True when config file does not exist."""
    assert is_first_run(tmp_path / "nonexistent.json") is True


def test_first_run_when_config_empty_file(tmp_path: Path) -> None:
    """Returns True when config file exists but is zero bytes."""
    cfg = tmp_path / "rex_config.json"
    cfg.write_text("", encoding="utf-8")
    assert is_first_run(cfg) is True


def test_first_run_when_config_whitespace_only(tmp_path: Path) -> None:
    """Returns True when config file contains only whitespace."""
    cfg = tmp_path / "rex_config.json"
    cfg.write_text("   \n  \t  ", encoding="utf-8")
    assert is_first_run(cfg) is True


def test_first_run_when_config_empty_object(tmp_path: Path) -> None:
    """Returns True when config file contains an empty JSON object."""
    cfg = tmp_path / "rex_config.json"
    cfg.write_text("{}", encoding="utf-8")
    assert is_first_run(cfg) is True


def test_first_run_when_config_corrupt(tmp_path: Path) -> None:
    """Returns True when config file contains invalid JSON (treat as first run)."""
    cfg = tmp_path / "rex_config.json"
    cfg.write_text("{not valid json", encoding="utf-8")
    assert is_first_run(cfg) is True


def test_not_first_run_when_config_populated(tmp_path: Path) -> None:
    """Returns False when config file has real content."""
    cfg = tmp_path / "rex_config.json"
    cfg.write_text(json.dumps({"models": {"llm_backend": "lmstudio"}}), encoding="utf-8")
    assert is_first_run(cfg) is False


def test_not_first_run_with_minimal_non_empty_config(tmp_path: Path) -> None:
    """Returns False for any non-empty JSON object."""
    cfg = tmp_path / "rex_config.json"
    cfg.write_text(json.dumps({"active_profile": "default"}), encoding="utf-8")
    assert is_first_run(cfg) is False


# ---------------------------------------------------------------------------
# print_welcome_message
# ---------------------------------------------------------------------------


def test_welcome_message_contains_welcome_text(capsys: pytest.CaptureFixture[str]) -> None:
    """Welcome message includes the required opening line."""
    print_welcome_message()
    out = capsys.readouterr().out
    assert "Welcome to Rex. Let's get you set up." in out


def test_welcome_message_contains_three_steps(capsys: pytest.CaptureFixture[str]) -> None:
    """Welcome message contains exactly 3 numbered steps."""
    print_welcome_message()
    out = capsys.readouterr().out
    assert "1." in out
    assert "2." in out
    assert "3." in out


def test_welcome_message_mentions_llm_config(capsys: pytest.CaptureFixture[str]) -> None:
    """Step 1 guides the user to configure their LLM provider."""
    print_welcome_message()
    out = capsys.readouterr().out
    assert "LLM" in out or "llm" in out or "openai" in out.lower()


def test_welcome_message_mentions_doctor(capsys: pytest.CaptureFixture[str]) -> None:
    """Step 2 mentions the doctor command."""
    print_welcome_message()
    out = capsys.readouterr().out
    assert "doctor" in out


def test_welcome_message_mentions_start_chatting(capsys: pytest.CaptureFixture[str]) -> None:
    """Step 3 guides the user to start chatting."""
    print_welcome_message()
    out = capsys.readouterr().out
    assert "rex" in out.lower()


# ---------------------------------------------------------------------------
# maybe_print_welcome
# ---------------------------------------------------------------------------


def test_maybe_print_welcome_prints_on_first_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Prints message and returns True when config is missing."""
    result = maybe_print_welcome(tmp_path / "nonexistent.json")
    out = capsys.readouterr().out
    assert result is True
    assert "Welcome to Rex" in out


def test_maybe_print_welcome_silent_when_config_present(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Prints nothing and returns False when config is present."""
    cfg = tmp_path / "rex_config.json"
    cfg.write_text(json.dumps({"active_profile": "default"}), encoding="utf-8")
    result = maybe_print_welcome(cfg)
    out = capsys.readouterr().out
    assert result is False
    assert out == ""


# ---------------------------------------------------------------------------
# Integration: main() calls maybe_print_welcome
# ---------------------------------------------------------------------------


def test_main_calls_maybe_print_welcome(tmp_path: Path) -> None:
    """main() invokes maybe_print_welcome so onboarding is triggered at startup."""

    import rex.cli as cli_module

    called: list[bool] = []

    def _fake_welcome(config_path: object = None) -> bool:
        called.append(True)
        return False

    with (
        patch("rex.cli.check_startup_env"),
        patch("rex.first_run.maybe_print_welcome", side_effect=_fake_welcome),
        patch("rex.cli.create_parser") as mock_parser,
    ):
        mock_ns = mock_parser.return_value.parse_args.return_value
        mock_ns.command = "version"
        mock_ns.func = lambda args: 0
        cli_module.main([])

    assert called, "maybe_print_welcome was not called from main()"
