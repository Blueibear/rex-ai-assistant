"""Tests for the rex tools CLI command."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from rex.cli import cmd_tools, create_parser, main
from rex.credentials import CredentialManager, set_credential_manager
from rex.tool_registry import ToolMeta, ToolRegistry, set_tool_registry


class TestToolsCommandParser:
    """Tests for tools command argument parsing."""

    def test_tools_subcommand_parsed(self):
        """Test that tools subcommand is parsed correctly."""
        parser = create_parser()
        args = parser.parse_args(["tools"])
        assert args.command == "tools"
        assert hasattr(args, "func")

    def test_tools_verbose_flag(self):
        """Test that -v flag is parsed for tools."""
        parser = create_parser()
        args = parser.parse_args(["tools", "-v"])
        assert args.command == "tools"
        assert args.verbose is True

    def test_tools_all_flag(self):
        """Test that -a flag is parsed for tools."""
        parser = create_parser()
        args = parser.parse_args(["tools", "-a"])
        assert args.command == "tools"
        assert args.all is True

    def test_tools_combined_flags(self):
        """Test that -v and -a flags can be combined."""
        parser = create_parser()
        args = parser.parse_args(["tools", "-v", "-a"])
        assert args.verbose is True
        assert args.all is True


class TestCmdTools:
    """Tests for the cmd_tools function."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create a fresh registry for each test
        self.credential_manager = CredentialManager(
            config_path=Path("/nonexistent/path.json")
        )
        self.registry = ToolRegistry(credential_manager=self.credential_manager)
        set_tool_registry(self.registry)
        set_credential_manager(self.credential_manager)

    def teardown_method(self):
        """Clean up after each test."""
        set_tool_registry(None)  # type: ignore
        set_credential_manager(None)  # type: ignore

    def test_cmd_tools_no_tools(self, capsys):
        """Test output when no tools are registered."""
        class Args:
            verbose = False
            all = False

        exit_code = cmd_tools(Args())
        assert exit_code == 0

        captured = capsys.readouterr()
        assert "No tools registered" in captured.out

    def test_cmd_tools_with_tools(self, capsys):
        """Test output with registered tools."""
        self.registry.register_tool(ToolMeta(
            name="test_tool",
            description="A test tool for testing",
        ))

        class Args:
            verbose = False
            all = False

        exit_code = cmd_tools(Args())
        assert exit_code == 0

        captured = capsys.readouterr()
        assert "Rex Tool Registry" in captured.out
        assert "test_tool" in captured.out
        assert "A test tool for testing" in captured.out

    def test_cmd_tools_shows_status_icons(self, capsys):
        """Test that status icons are shown correctly."""
        # Ready tool
        self.registry.register_tool(ToolMeta(
            name="ready_tool",
            description="Ready tool",
            health_check=lambda: (True, "OK"),
        ))

        # Unhealthy tool
        self.registry.register_tool(ToolMeta(
            name="unhealthy_tool",
            description="Unhealthy tool",
            health_check=lambda: (False, "Down"),
        ))

        # Tool missing credentials
        self.registry.register_tool(ToolMeta(
            name="no_creds_tool",
            description="No creds tool",
            required_credentials=["missing_cred"],
        ))

        class Args:
            verbose = False
            all = False

        exit_code = cmd_tools(Args())
        assert exit_code == 0

        captured = capsys.readouterr()
        assert "[READY]" in captured.out
        assert "[UNHEALTHY]" in captured.out
        assert "[NO CREDS]" in captured.out

    def test_cmd_tools_verbose_output(self, capsys):
        """Test verbose output includes additional details."""
        self.credential_manager.set_token("api_key", "secret")
        self.registry.register_tool(ToolMeta(
            name="verbose_tool",
            description="Tool for verbose test",
            version="2.0.0",
            capabilities=["read", "write"],
            required_credentials=["api_key"],
            health_check=lambda: (True, "Service running"),
        ))

        class Args:
            verbose = True
            all = False

        exit_code = cmd_tools(Args())
        assert exit_code == 0

        captured = capsys.readouterr()
        assert "Version: 2.0.0" in captured.out
        assert "Capabilities: read, write" in captured.out
        assert "Required credentials: api_key" in captured.out
        assert "Health: Service running" in captured.out

    def test_cmd_tools_excludes_disabled_by_default(self, capsys):
        """Test that disabled tools are excluded by default."""
        self.registry.register_tool(ToolMeta(
            name="enabled_tool",
            description="Enabled",
        ))
        self.registry.register_tool(ToolMeta(
            name="disabled_tool",
            description="Disabled",
            enabled=False,
        ))

        class Args:
            verbose = False
            all = False

        exit_code = cmd_tools(Args())
        assert exit_code == 0

        captured = capsys.readouterr()
        assert "enabled_tool" in captured.out
        assert "disabled_tool" not in captured.out

    def test_cmd_tools_includes_disabled_with_flag(self, capsys):
        """Test that disabled tools are included with -a flag."""
        self.registry.register_tool(ToolMeta(
            name="enabled_tool",
            description="Enabled",
        ))
        self.registry.register_tool(ToolMeta(
            name="disabled_tool",
            description="Disabled",
            enabled=False,
        ))

        class Args:
            verbose = False
            all = True

        exit_code = cmd_tools(Args())
        assert exit_code == 0

        captured = capsys.readouterr()
        assert "enabled_tool" in captured.out
        assert "disabled_tool" in captured.out
        assert "[DISABLED]" in captured.out

    def test_cmd_tools_shows_summary(self, capsys):
        """Test that summary is shown at the end."""
        self.registry.register_tool(ToolMeta(
            name="tool1",
            description="Tool 1",
        ))
        self.registry.register_tool(ToolMeta(
            name="tool2",
            description="Tool 2",
            health_check=lambda: (False, "Down"),
        ))

        class Args:
            verbose = False
            all = False

        exit_code = cmd_tools(Args())
        assert exit_code == 0

        captured = capsys.readouterr()
        assert "Total: 2 tools" in captured.out

    def test_cmd_tools_verbose_shows_missing_creds(self, capsys):
        """Test that verbose output shows missing credentials."""
        self.registry.register_tool(ToolMeta(
            name="needs_creds",
            description="Needs credentials",
            required_credentials=["missing_key", "another_key"],
        ))

        class Args:
            verbose = True
            all = False

        exit_code = cmd_tools(Args())
        assert exit_code == 0

        captured = capsys.readouterr()
        assert "Missing credentials:" in captured.out
        assert "missing_key" in captured.out


class TestMainToolsCommand:
    """Tests for tools command via main function."""

    def setup_method(self):
        """Set up test fixtures."""
        self.credential_manager = CredentialManager(
            config_path=Path("/nonexistent/path.json")
        )
        self.registry = ToolRegistry(credential_manager=self.credential_manager)
        self.registry.register_tool(ToolMeta(
            name="main_test_tool",
            description="Tool for main test",
        ))
        set_tool_registry(self.registry)
        set_credential_manager(self.credential_manager)

    def teardown_method(self):
        """Clean up after each test."""
        set_tool_registry(None)  # type: ignore
        set_credential_manager(None)  # type: ignore

    def test_main_tools_command(self, capsys):
        """Test main() handles tools command."""
        exit_code = main(["tools"])
        assert exit_code == 0

        captured = capsys.readouterr()
        assert "Rex Tool Registry" in captured.out
        assert "main_test_tool" in captured.out

    def test_main_tools_verbose(self, capsys):
        """Test main() handles tools -v command."""
        exit_code = main(["tools", "-v"])
        assert exit_code == 0

        captured = capsys.readouterr()
        assert "Version:" in captured.out


class TestToolsCLIIntegration:
    """Integration tests running tools command via subprocess."""

    def test_module_tools_help(self):
        """Test running python -m rex tools --help."""
        result = subprocess.run(
            [sys.executable, "-m", "rex", "tools", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        assert "tools" in result.stdout.lower()
        assert "registered tools" in result.stdout.lower()

    def test_module_tools(self):
        """Test running python -m rex tools."""
        result = subprocess.run(
            [sys.executable, "-m", "rex", "tools"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        # Should show registry output
        assert "Rex Tool Registry" in result.stdout or "tools" in result.stdout.lower()

    def test_module_tools_verbose(self):
        """Test running python -m rex tools -v."""
        result = subprocess.run(
            [sys.executable, "-m", "rex", "tools", "-v"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0

    def test_module_tools_all(self):
        """Test running python -m rex tools -a."""
        result = subprocess.run(
            [sys.executable, "-m", "rex", "tools", "-a"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0

    def test_help_includes_tools_command(self):
        """Test that main help includes tools command."""
        result = subprocess.run(
            [sys.executable, "-m", "rex", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        assert "tools" in result.stdout


class TestToolsOutputFormat:
    """Tests for tools command output formatting."""

    def setup_method(self):
        """Set up test fixtures."""
        self.credential_manager = CredentialManager(
            config_path=Path("/nonexistent/path.json")
        )
        self.registry = ToolRegistry(credential_manager=self.credential_manager)
        set_tool_registry(self.registry)
        set_credential_manager(self.credential_manager)

    def teardown_method(self):
        """Clean up after each test."""
        set_tool_registry(None)  # type: ignore
        set_credential_manager(None)  # type: ignore

    def test_output_has_header(self, capsys):
        """Test that output has proper header."""
        self.registry.register_tool(ToolMeta(name="tool", description="Tool"))

        class Args:
            verbose = False
            all = False

        cmd_tools(Args())
        captured = capsys.readouterr()

        assert "Rex Tool Registry" in captured.out
        assert "=" in captured.out  # Separator

    def test_output_format_per_tool(self, capsys):
        """Test output format for each tool."""
        self.registry.register_tool(ToolMeta(
            name="format_test",
            description="Description for format test",
        ))

        class Args:
            verbose = False
            all = False

        cmd_tools(Args())
        captured = capsys.readouterr()

        # Check format: name [STATUS]
        assert "format_test [" in captured.out
        # Check description is indented
        assert "  Description: Description for format test" in captured.out

    def test_tools_sorted_alphabetically(self, capsys):
        """Test that tools are listed alphabetically."""
        self.registry.register_tool(ToolMeta(name="zebra", description="Z"))
        self.registry.register_tool(ToolMeta(name="alpha", description="A"))
        self.registry.register_tool(ToolMeta(name="middle", description="M"))

        class Args:
            verbose = False
            all = False

        cmd_tools(Args())
        captured = capsys.readouterr()

        # Find positions of tool names
        alpha_pos = captured.out.find("alpha")
        middle_pos = captured.out.find("middle")
        zebra_pos = captured.out.find("zebra")

        assert alpha_pos < middle_pos < zebra_pos
