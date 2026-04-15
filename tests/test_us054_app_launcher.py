"""Tests for US-054: Desktop program launch capability.

Covers:
- load_app_registry: missing file, invalid JSON, valid registry, case normalisation
- launch_app: found (mocked subprocess), not found, case-insensitive lookup
- Platform dispatch: Windows (os.startfile), macOS (open), Linux (xdg-open)
- launch_app returns the "not found" message when app is absent from registry
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# load_app_registry
# ---------------------------------------------------------------------------


class TestLoadAppRegistry:
    def test_missing_file_returns_empty_dict(self, tmp_path: Path) -> None:
        from rex.computers.app_launcher import load_app_registry

        result = load_app_registry(tmp_path / "nonexistent.json")
        assert result == {}

    def test_valid_registry_loaded(self, tmp_path: Path) -> None:
        from rex.computers.app_launcher import load_app_registry

        reg = tmp_path / "reg.json"
        reg.write_text(json.dumps({"Notepad": "notepad.exe", "Chrome": "chrome.exe"}))
        result = load_app_registry(reg)
        assert result == {"notepad": "notepad.exe", "chrome": "chrome.exe"}

    def test_keys_normalized_to_lowercase(self, tmp_path: Path) -> None:
        from rex.computers.app_launcher import load_app_registry

        reg = tmp_path / "reg.json"
        reg.write_text(json.dumps({"MY APP": "/usr/bin/myapp"}))
        result = load_app_registry(reg)
        assert "my app" in result

    def test_invalid_json_returns_empty_dict(self, tmp_path: Path) -> None:
        from rex.computers.app_launcher import load_app_registry

        reg = tmp_path / "bad.json"
        reg.write_text("not-json{{{")
        result = load_app_registry(reg)
        assert result == {}

    def test_non_object_json_returns_empty_dict(self, tmp_path: Path) -> None:
        from rex.computers.app_launcher import load_app_registry

        reg = tmp_path / "list.json"
        reg.write_text(json.dumps(["a", "b"]))
        result = load_app_registry(reg)
        assert result == {}


# ---------------------------------------------------------------------------
# launch_app: not found path
# ---------------------------------------------------------------------------


class TestLaunchAppNotFound:
    def test_returns_not_found_message(self, tmp_path: Path) -> None:
        from rex.computers.app_launcher import APP_NOT_FOUND_MESSAGE, launch_app

        reg = tmp_path / "reg.json"
        reg.write_text(json.dumps({"notepad": "notepad.exe"}))
        result = launch_app("photoshop", registry_path=reg)
        assert result == APP_NOT_FOUND_MESSAGE

    def test_not_found_message_content(self) -> None:
        from rex.computers.app_launcher import APP_NOT_FOUND_MESSAGE

        assert "settings" in APP_NOT_FOUND_MESSAGE.lower()

    def test_empty_registry_returns_not_found(self, tmp_path: Path) -> None:
        from rex.computers.app_launcher import APP_NOT_FOUND_MESSAGE, launch_app

        reg = tmp_path / "reg.json"
        reg.write_text(json.dumps({}))
        assert launch_app("anything", registry_path=reg) == APP_NOT_FOUND_MESSAGE


# ---------------------------------------------------------------------------
# launch_app: found path (mocked subprocess / os.startfile)
# ---------------------------------------------------------------------------


class TestLaunchAppFound:
    def _make_registry(self, tmp_path: Path, entries: dict) -> Path:
        reg = tmp_path / "reg.json"
        reg.write_text(json.dumps(entries))
        return reg

    def test_returns_opening_message(self, tmp_path: Path) -> None:
        from rex.computers.app_launcher import launch_app

        reg = self._make_registry(tmp_path, {"notepad": "notepad.exe"})
        with patch("rex.computers.app_launcher._platform_launch") as mock_launch:
            result = launch_app("notepad", registry_path=reg)
        mock_launch.assert_called_once_with("notepad.exe")
        assert "notepad" in result.lower()

    def test_case_insensitive_lookup(self, tmp_path: Path) -> None:
        from rex.computers.app_launcher import launch_app

        reg = self._make_registry(tmp_path, {"chrome": "chrome.exe"})
        with patch("rex.computers.app_launcher._platform_launch"):
            result = launch_app("Chrome", registry_path=reg)
        assert "Chrome" in result or "chrome" in result.lower()

    def test_launch_failure_returns_error_message(self, tmp_path: Path) -> None:
        from rex.computers.app_launcher import launch_app

        reg = self._make_registry(tmp_path, {"badapp": "/nonexistent/app"})
        with patch(
            "rex.computers.app_launcher._platform_launch",
            side_effect=OSError("file not found"),
        ):
            result = launch_app("badapp", registry_path=reg)
        assert "couldn't open" in result.lower() or "couldn't" in result.lower()


# ---------------------------------------------------------------------------
# Platform dispatch (mocked)
# ---------------------------------------------------------------------------


class TestPlatformDispatch:
    def _make_registry(self, tmp_path: Path) -> Path:
        reg = tmp_path / "reg.json"
        reg.write_text(json.dumps({"myapp": "/some/app"}))
        return reg

    def test_windows_uses_startfile(self, tmp_path: Path) -> None:
        from rex.computers import app_launcher

        self._make_registry(tmp_path)
        mock_startfile = MagicMock()
        with (
            patch.object(sys, "platform", "win32"),
            patch.object(app_launcher.os, "startfile", mock_startfile, create=True),
        ):
            app_launcher._platform_launch("/some/app")
        mock_startfile.assert_called_once_with("/some/app")

    def test_macos_uses_open(self, tmp_path: Path) -> None:
        from rex.computers import app_launcher

        with (
            patch.object(sys, "platform", "darwin"),
            patch("subprocess.Popen") as mock_popen,
        ):
            app_launcher._platform_launch("/Applications/App.app")
        mock_popen.assert_called_once_with(["open", "/Applications/App.app"])

    def test_linux_uses_xdg_open(self, tmp_path: Path) -> None:
        from rex.computers import app_launcher

        with (
            patch.object(sys, "platform", "linux"),
            patch("subprocess.Popen") as mock_popen,
        ):
            app_launcher._platform_launch("/usr/bin/myapp")
        mock_popen.assert_called_once_with(["xdg-open", "/usr/bin/myapp"])
