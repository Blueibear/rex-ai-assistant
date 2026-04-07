"""Unit tests for rex.bridge_utils."""

import os
import sys
from pathlib import Path
from unittest.mock import patch

import rex.bridge_utils as bu
from rex.bridge_utils import repo_root, resolve_python


class TestResolvePython:
    def test_fallback_to_sys_executable_when_no_venv(self):
        """Without VIRTUAL_ENV, returns sys.executable."""
        env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
        with patch.dict(os.environ, env, clear=True):
            result = resolve_python()
        assert result == sys.executable

    def test_windows_path_from_venv(self, tmp_path):
        """On Windows, resolves Scripts/python.exe inside VIRTUAL_ENV."""
        scripts_dir = tmp_path / "Scripts"
        scripts_dir.mkdir()
        fake_python = scripts_dir / "python.exe"
        fake_python.write_text("")

        with patch.dict(os.environ, {"VIRTUAL_ENV": str(tmp_path)}):
            with patch.object(bu.sys, "platform", "win32"):
                result = resolve_python()

        assert result == str(fake_python)

    def test_unix_path_from_venv(self, tmp_path):
        """On Linux/macOS, resolves bin/python inside VIRTUAL_ENV."""
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        fake_python = bin_dir / "python"
        fake_python.write_text("")

        with patch.dict(os.environ, {"VIRTUAL_ENV": str(tmp_path)}):
            with patch.object(bu.sys, "platform", "linux"):
                result = resolve_python()

        assert result == str(fake_python)

    def test_macos_path_from_venv(self, tmp_path):
        """On macOS (darwin), resolves bin/python inside VIRTUAL_ENV."""
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        fake_python = bin_dir / "python"
        fake_python.write_text("")

        with patch.dict(os.environ, {"VIRTUAL_ENV": str(tmp_path)}):
            with patch.object(bu.sys, "platform", "darwin"):
                result = resolve_python()

        assert result == str(fake_python)

    def test_falls_back_when_venv_python_missing(self, tmp_path):
        """If VIRTUAL_ENV binary doesn't exist, fallback to sys.executable."""
        with patch.dict(os.environ, {"VIRTUAL_ENV": str(tmp_path)}):
            result = resolve_python()
        assert result == sys.executable

    def test_returns_string(self):
        assert isinstance(resolve_python(), str)


class TestRepoRoot:
    def test_returns_path_containing_pyproject_toml(self):
        root = repo_root()
        assert (root / "pyproject.toml").exists()

    def test_returns_path_object(self):
        assert isinstance(repo_root(), Path)
