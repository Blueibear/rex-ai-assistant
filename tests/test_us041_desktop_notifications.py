"""Tests for US-041: local desktop notifications (rex.notifications.desktop)."""

from __future__ import annotations

import logging
import sys
from unittest.mock import MagicMock, patch

import pytest

from rex.notifications.desktop import notify

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_plyer_mock() -> MagicMock:
    plyer_mod = MagicMock()
    plyer_mod.notification = MagicMock()
    return plyer_mod


# ---------------------------------------------------------------------------
# plyer backend
# ---------------------------------------------------------------------------


class TestNotifyPlyer:
    def test_plyer_called_with_correct_args(self) -> None:
        plyer_mock = _make_plyer_mock()
        with (
            patch.dict(sys.modules, {"plyer": plyer_mock}),
            patch("importlib.util.find_spec", return_value=MagicMock()),
        ):
            notify("Hello", "World", urgency="normal")

        plyer_mock.notification.notify.assert_called_once_with(
            title="Hello", message="World", app_name="Rex"
        )

    def test_plyer_not_called_when_unavailable(self) -> None:
        """When plyer is absent the function must not raise."""
        with patch("importlib.util.find_spec", return_value=None):
            # Should fall through to platform backend or log — no exception
            notify("Hi", "There")

    def test_plyer_exception_swallowed(self) -> None:
        """plyer raising an exception must not propagate."""
        plyer_mock = _make_plyer_mock()
        plyer_mock.notification.notify.side_effect = RuntimeError("boom")
        with (
            patch.dict(sys.modules, {"plyer": plyer_mock}),
            patch("importlib.util.find_spec", return_value=MagicMock()),
            patch("rex.notifications.desktop._notify_windows", return_value=True),
        ):
            notify("Fail", "Gracefully")
        # No exception raised = pass


# ---------------------------------------------------------------------------
# Platform-native backends (subprocess based)
# ---------------------------------------------------------------------------


class TestNotifyWindows:
    def test_subprocess_called_on_windows(self) -> None:
        import subprocess

        with (
            patch("sys.platform", "win32"),
            patch("platform.system", return_value="Windows"),
            patch("importlib.util.find_spec", return_value=None),
            patch("subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(returncode=0)
            notify("Test", "Message")

        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert args[0] == "powershell"

    def test_windows_failure_returns_false(self) -> None:
        with (
            patch("sys.platform", "win32"),
            patch("importlib.util.find_spec", return_value=None),
            patch("subprocess.run", side_effect=OSError("not found")),
        ):
            from rex.notifications.desktop import _notify_windows

            result = _notify_windows("T", "M")
        assert result is False

    def test_skipped_on_non_windows(self) -> None:
        with patch("sys.platform", "linux"):
            from rex.notifications.desktop import _notify_windows

            result = _notify_windows("T", "M")
        assert result is False


class TestNotifyMacOS:
    def test_subprocess_called_on_macos(self) -> None:
        with (
            patch("sys.platform", "darwin"),
            patch("platform.system", return_value="Darwin"),
            patch("importlib.util.find_spec", return_value=None),
            patch("subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(returncode=0)
            notify("Alert", "Now")

        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert args[0] == "osascript"

    def test_macos_failure_returns_false(self) -> None:
        with (
            patch("sys.platform", "darwin"),
            patch("subprocess.run", side_effect=FileNotFoundError),
        ):
            from rex.notifications.desktop import _notify_macos

            result = _notify_macos("T", "M")
        assert result is False

    def test_skipped_on_non_macos(self) -> None:
        with patch("sys.platform", "win32"):
            from rex.notifications.desktop import _notify_macos

            result = _notify_macos("T", "M")
        assert result is False

    def test_quotes_escaped_in_script(self) -> None:
        with (
            patch("sys.platform", "darwin"),
            patch("subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(returncode=0)
            from rex.notifications.desktop import _notify_macos

            _notify_macos('Say "hello"', "World's end")

        script_arg = mock_run.call_args[0][0][2]
        # Double-quotes in title must be escaped
        assert '\\"' in script_arg


class TestNotifyLinux:
    def test_subprocess_called_on_linux(self) -> None:
        with (
            patch("sys.platform", "linux"),
            patch("platform.system", return_value="Linux"),
            patch("importlib.util.find_spec", return_value=None),
            patch("subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(returncode=0)
            notify("Alert", "Test")

        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert args[0] == "notify-send"

    def test_urgency_mapped_to_critical_for_high(self) -> None:
        with (
            patch("sys.platform", "linux"),
            patch("subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(returncode=0)
            from rex.notifications.desktop import _notify_linux

            _notify_linux("T", "M", "high")

        cmd = mock_run.call_args[0][0]
        assert "--urgency=critical" in cmd

    def test_urgency_mapped_to_low_for_low(self) -> None:
        with (
            patch("sys.platform", "linux"),
            patch("subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(returncode=0)
            from rex.notifications.desktop import _notify_linux

            _notify_linux("T", "M", "low")

        cmd = mock_run.call_args[0][0]
        assert "--urgency=low" in cmd

    def test_linux_failure_returns_false(self) -> None:
        with (
            patch("sys.platform", "linux"),
            patch("subprocess.run", side_effect=FileNotFoundError),
        ):
            from rex.notifications.desktop import _notify_linux

            result = _notify_linux("T", "M", "normal")
        assert result is False

    def test_skipped_on_non_linux(self) -> None:
        with patch("sys.platform", "win32"):
            from rex.notifications.desktop import _notify_linux

            result = _notify_linux("T", "M", "normal")
        assert result is False


# ---------------------------------------------------------------------------
# Log-only fallback
# ---------------------------------------------------------------------------


class TestLogFallback:
    def test_warning_logged_when_all_backends_fail(self, caplog: pytest.LogCaptureFixture) -> None:
        with (
            patch("importlib.util.find_spec", return_value=None),
            patch("rex.notifications.desktop._notify_windows", return_value=False),
            patch("rex.notifications.desktop._notify_macos", return_value=False),
            patch("rex.notifications.desktop._notify_linux", return_value=False),
            caplog.at_level(logging.WARNING, logger="rex.notifications.desktop"),
        ):
            notify("Unreachable", "This is logged only")

        assert any("Desktop notification system unavailable" in r.message for r in caplog.records)

    def test_no_exception_when_all_backends_fail(self) -> None:
        with (
            patch("importlib.util.find_spec", return_value=None),
            patch("rex.notifications.desktop._notify_windows", return_value=False),
            patch("rex.notifications.desktop._notify_macos", return_value=False),
            patch("rex.notifications.desktop._notify_linux", return_value=False),
        ):
            # Must not raise
            notify("Title", "Body")


# ---------------------------------------------------------------------------
# Default urgency
# ---------------------------------------------------------------------------


class TestDefaultUrgency:
    def test_default_urgency_is_normal(self) -> None:
        """notify() must accept a call without specifying urgency."""
        with (
            patch("importlib.util.find_spec", return_value=None),
            patch("rex.notifications.desktop._notify_windows", return_value=True),
        ):
            notify("Title", "Body")  # no urgency kwarg — must not raise


# ---------------------------------------------------------------------------
# Module exports
# ---------------------------------------------------------------------------


class TestModuleExports:
    def test_notify_importable_from_package(self) -> None:
        from rex.notifications import notify as pkg_notify

        assert callable(pkg_notify)

    def test_desktop_module_all(self) -> None:
        from rex.notifications import desktop

        assert "notify" in desktop.__all__
