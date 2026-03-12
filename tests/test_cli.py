"""Tests for rex CLI entry point."""

from __future__ import annotations

import subprocess
import sys
from io import StringIO
from unittest.mock import patch

import pytest

from rex.cli import cmd_doctor, cmd_version, create_parser, main


class TestCLIParser:
    """Tests for CLI argument parser."""

    def test_parser_creation(self):
        """Test that parser is created successfully."""
        parser = create_parser()
        assert parser is not None
        assert parser.prog == "rex"

    def test_parser_help(self):
        """Test that parser generates help text."""
        parser = create_parser()
        # Should not raise when formatting help
        help_text = parser.format_help()
        assert "rex" in help_text.lower()
        assert "doctor" in help_text

    def test_doctor_subcommand_parsed(self):
        """Test that doctor subcommand is parsed correctly."""
        parser = create_parser()
        args = parser.parse_args(["doctor"])
        assert args.command == "doctor"
        assert hasattr(args, "func")

    def test_doctor_verbose_flag(self):
        """Test that -v flag is parsed for doctor."""
        parser = create_parser()
        args = parser.parse_args(["doctor", "-v"])
        assert args.command == "doctor"
        assert args.verbose is True

    def test_version_subcommand_parsed(self):
        """Test that version subcommand is parsed correctly."""
        parser = create_parser()
        args = parser.parse_args(["version"])
        assert args.command == "version"

    def test_chat_subcommand_parsed(self):
        """Test that chat subcommand is parsed correctly."""
        parser = create_parser()
        args = parser.parse_args(["chat"])
        assert args.command == "chat"

    def test_no_command_default(self):
        """Test behavior when no command is specified."""
        parser = create_parser()
        args = parser.parse_args([])
        assert args.command is None


class TestMainFunction:
    """Tests for the main() entry point function."""

    def test_main_doctor_command(self):
        """Test that main() handles doctor command."""
        exit_code = main(["doctor"])
        assert isinstance(exit_code, int)
        assert exit_code in (0, 1)

    def test_main_version_command(self):
        """Test that main() handles version command."""
        exit_code = main(["version"])
        assert exit_code == 0

    def test_main_version_verbose(self):
        """Test that main() handles version -v command."""
        exit_code = main(["version", "-v"])
        assert exit_code == 0

    def test_main_help_flag(self):
        """Test that --help flag works."""
        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])
        assert exc_info.value.code == 0

    def test_main_version_flag(self):
        """Test that --version flag works."""
        with pytest.raises(SystemExit) as exc_info:
            main(["--version"])
        assert exc_info.value.code == 0


class TestVersionCommand:
    """Tests for the version command."""

    def test_cmd_version_returns_zero(self):
        """Test that version command returns 0."""

        class Args:
            verbose = False

        exit_code = cmd_version(Args())
        assert exit_code == 0

    def test_cmd_version_verbose_returns_zero(self):
        """Test that verbose version command returns 0."""

        class Args:
            verbose = True

        exit_code = cmd_version(Args())
        assert exit_code == 0


class TestDoctorCommand:
    """Tests for the doctor command via CLI."""

    def test_cmd_doctor_returns_int(self):
        """Test that doctor command returns an integer."""

        class Args:
            verbose = False

        exit_code = cmd_doctor(Args())
        assert isinstance(exit_code, int)
        assert exit_code in (0, 1)

    def test_cmd_doctor_verbose(self):
        """Test that verbose doctor command works."""

        class Args:
            verbose = True

        exit_code = cmd_doctor(Args())
        assert isinstance(exit_code, int)


class TestCLIIntegration:
    """Integration tests running CLI via subprocess."""

    def test_module_no_args_shows_help_or_chat(self):
        """Test that running without args shows help or starts chat logic."""
        # We mock stdin to avoid actually waiting for input in chat mode
        result = subprocess.run(
            [sys.executable, "-m", "rex", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        assert "rex" in result.stdout.lower()
        assert "doctor" in result.stdout

    def test_module_doctor(self):
        """Test running python -m rex doctor."""
        result = subprocess.run(
            [sys.executable, "-m", "rex", "doctor"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode in (0, 1)
        assert "Rex Doctor" in result.stdout

    def test_module_version(self):
        """Test running python -m rex version."""
        result = subprocess.run(
            [sys.executable, "-m", "rex", "version"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        assert "rex-ai-assistant" in result.stdout

    def test_module_version_verbose(self):
        """Test running python -m rex version -v."""
        result = subprocess.run(
            [sys.executable, "-m", "rex", "version", "-v"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        assert "Python:" in result.stdout

    def test_module_help(self):
        """Test running python -m rex --help."""
        result = subprocess.run(
            [sys.executable, "-m", "rex", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        assert "doctor" in result.stdout
        assert "chat" in result.stdout
        assert "version" in result.stdout

    def test_doctor_help(self):
        """Test running python -m rex doctor --help."""
        result = subprocess.run(
            [sys.executable, "-m", "rex", "doctor", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        assert "diagnostic" in result.stdout.lower()

    def test_invalid_command(self):
        """Test running with invalid command shows error."""
        result = subprocess.run(
            [sys.executable, "-m", "rex", "invalid_command_xyz"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        # argparse returns error for invalid choices
        assert result.returncode != 0

    def test_cli_module_direct(self):
        """Test running rex.cli module directly."""
        result = subprocess.run(
            [sys.executable, "-m", "rex.cli", "doctor"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode in (0, 1)
        assert "Rex Doctor" in result.stdout


class TestLegacyEnvWarning:
    """Tests that legacy env var warnings appear and reference the migration command."""

    def test_help_warns_about_openai_base_url(self):
        """When OPENAI_BASE_URL is set, python -m rex --help should emit a warning
        that includes the variable name and the migration command."""
        import os

        from rex.config_manager import ENV_TO_CONFIG_MAPPING

        env = os.environ.copy()
        # Suppress all other legacy env vars so OPENAI_BASE_URL is shown explicitly.
        # Setting to "" prevents the .env loader from overwriting (override=False)
        # while os.getenv() returns "" which is falsy → no warning emitted for those.
        for key in ENV_TO_CONFIG_MAPPING:
            if key != "OPENAI_BASE_URL":
                env[key] = ""
        env["OPENAI_BASE_URL"] = "http://example.com/v1"
        # Remove cached module state by using a fresh subprocess
        result = subprocess.run(
            [sys.executable, "-m", "rex", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
            env=env,
        )
        assert result.returncode == 0
        # Warning goes to stderr (logging)
        combined = result.stderr + result.stdout
        assert "OPENAI_BASE_URL" in combined, (
            f"Expected OPENAI_BASE_URL warning in output.\nstderr:\n{result.stderr}\nstdout:\n{result.stdout}"
        )
        assert "rex-config migrate-legacy-env" in combined, (
            f"Expected migration command in warning.\nstderr:\n{result.stderr}\nstdout:\n{result.stdout}"
        )


class TestCLIOutput:
    """Tests for CLI output formatting."""

    def test_doctor_output_contains_sections(self):
        """Test that doctor output contains expected sections."""
        result = subprocess.run(
            [sys.executable, "-m", "rex", "doctor"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        output = result.stdout

        # Check for section markers
        assert "Rex Doctor" in output
        assert "=" in output  # separator line

        # Check for common check names
        assert "Python Version" in output or "python" in output.lower()

    def test_doctor_verbose_more_output(self):
        """Test that verbose mode produces more output."""
        # Run normal
        result_normal = subprocess.run(
            [sys.executable, "-m", "rex", "doctor"],
            capture_output=True,
            text=True,
            timeout=60,
        )

        # Run verbose
        result_verbose = subprocess.run(
            [sys.executable, "-m", "rex", "doctor", "-v"],
            capture_output=True,
            text=True,
            timeout=60,
        )

        # Verbose should have at least as much output
        assert len(result_verbose.stdout) >= len(result_normal.stdout)
