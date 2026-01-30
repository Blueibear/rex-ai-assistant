"""Tests for CLI devtools commands."""

import argparse
from unittest.mock import Mock, patch

from rex.cli import cmd_browser, cmd_code, cmd_gh, cmd_os


class TestBrowserCLI:
    """Tests for browser CLI commands."""

    def test_cmd_browser_sessions(self):
        """Test 'rex browser sessions' command."""
        with patch("rex.cli.get_browser_service") as mock_service:
            mock_service.return_value.list_sessions.return_value = []

            args = argparse.Namespace(
                browser_command="sessions",
                verbose=False,
            )

            result = cmd_browser(args)
            assert result == 0

    def test_cmd_browser_screenshots(self):
        """Test 'rex browser screenshots' command."""
        with patch("rex.cli.get_browser_service") as mock_service:
            mock_service.return_value.list_screenshots.return_value = []

            args = argparse.Namespace(
                browser_command="screenshots",
                verbose=False,
            )

            result = cmd_browser(args)
            assert result == 0


class TestOSCLI:
    """Tests for OS CLI commands."""

    def test_cmd_os_run_allowed(self):
        """Test 'rex os run' command with allowed command."""
        with patch("rex.cli.get_os_service") as mock_service:
            mock_result = Mock()
            mock_result.returncode = 0
            mock_result.stdout = "hello\n"
            mock_result.stderr = ""
            mock_result.duration_ms = 10

            mock_service.return_value.run_command.return_value = mock_result

            args = argparse.Namespace(
                os_command="run",
                command="echo hello",
                verbose=False,
            )

            result = cmd_os(args)
            assert result == 0

    def test_cmd_os_trash(self):
        """Test 'rex os trash' command."""
        with patch("rex.cli.get_os_service") as mock_service:
            mock_service.return_value.list_trash.return_value = []

            args = argparse.Namespace(
                os_command="trash",
                verbose=False,
            )

            result = cmd_os(args)
            assert result == 0


class TestGitHubCLI:
    """Tests for GitHub CLI commands."""

    def test_cmd_gh_repos(self):
        """Test 'rex gh repos' command."""
        with patch("rex.cli.get_github_service") as mock_service:
            from rex.github_service import Repository

            mock_repo = Repository(
                name="test-repo",
                full_name="user/test-repo",
                owner="user",
                url="https://github.com/user/test-repo",
                description="Test repo",
                private=False,
                default_branch="main",
            )

            mock_service.return_value.list_repos.return_value = [mock_repo]

            args = argparse.Namespace(
                gh_command="repos",
                verbose=False,
            )

            result = cmd_gh(args)
            assert result == 0

    def test_cmd_gh_prs(self):
        """Test 'rex gh prs' command."""
        with patch("rex.cli.get_github_service") as mock_service:
            mock_service.return_value.list_prs.return_value = []

            args = argparse.Namespace(
                gh_command="prs",
                repo="user/repo",
                state="open",
                verbose=False,
            )

            result = cmd_gh(args)
            assert result == 0


class TestCodeCLI:
    """Tests for code CLI commands."""

    def test_cmd_code_open(self):
        """Test 'rex code open' command."""
        with patch("rex.cli.get_vscode_service") as mock_service:
            mock_service.return_value.open_file.return_value = {
                "status": "success",
                "path": "/path/to/file.txt",
                "content": "test content",
                "lines": 1,
                "size": 12,
                "modified": "2024-01-01T00:00:00",
            }

            args = argparse.Namespace(
                code_command="open",
                path="file.txt",
                verbose=False,
            )

            result = cmd_code(args)
            assert result == 0

    def test_cmd_code_test(self):
        """Test 'rex code test' command."""
        with patch("rex.cli.get_vscode_service") as mock_service:
            from rex.vscode_service import TestResult

            mock_result = TestResult(
                success=True,
                total=10,
                passed=10,
                failed=0,
                errors=0,
                skipped=0,
                duration_seconds=1.5,
                output="All tests passed",
            )

            mock_service.return_value.run_tests.return_value = mock_result

            args = argparse.Namespace(
                code_command="test",
                path=None,
                pattern=None,
                verbose=False,
            )

            result = cmd_code(args)
            assert result == 0
