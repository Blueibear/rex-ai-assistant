"""Tests for US-139: Create a single-command install script.

Acceptance criteria:
  AC1  install.ps1 (Windows) and install.sh (Linux/macOS) exist at the repo root
  AC2  the script creates a virtual environment, installs Rex with all required
       dependencies, and verifies the install
  AC3  on success, the script prints a clear "Rex is installed. Run `rex` to start." message
  AC4  on failure, the script prints a specific error and exits with a non-zero code
  AC5  Typecheck passes
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# AC1 — scripts exist at the repo root
# ---------------------------------------------------------------------------


def test_install_sh_exists() -> None:
    assert (REPO_ROOT / "install.sh").is_file(), "install.sh must exist at the repo root"


def test_install_ps1_exists() -> None:
    assert (REPO_ROOT / "install.ps1").is_file(), "install.ps1 must exist at the repo root"


# ---------------------------------------------------------------------------
# AC2 — scripts create a venv and run pip install
# ---------------------------------------------------------------------------


def test_install_sh_creates_venv() -> None:
    content = (REPO_ROOT / "install.sh").read_text()
    assert "venv" in content, "install.sh must create a virtual environment"


def test_install_sh_runs_pip_install() -> None:
    content = (REPO_ROOT / "install.sh").read_text()
    assert "pip install" in content, "install.sh must run pip install"


def test_install_sh_installs_repo() -> None:
    content = (REPO_ROOT / "install.sh").read_text()
    assert (
        "[full]" in content or "REPO_DIR" in content
    ), "install.sh must install from the repo directory with required extras"


def test_install_ps1_creates_venv() -> None:
    content = (REPO_ROOT / "install.ps1").read_text()
    assert "venv" in content.lower(), "install.ps1 must create a virtual environment"


def test_install_ps1_runs_pip_install() -> None:
    content = (REPO_ROOT / "install.ps1").read_text()
    assert "pip install" in content.lower() or "Pip" in content, "install.ps1 must run pip install"


def test_install_ps1_installs_repo() -> None:
    content = (REPO_ROOT / "install.ps1").read_text()
    assert (
        "[full]" in content or "RepoDir" in content
    ), "install.ps1 must install from the repo directory with required extras"


def test_install_sh_verifies_install() -> None:
    content = (REPO_ROOT / "install.sh").read_text()
    assert (
        "--help" in content or "verify" in content.lower()
    ), "install.sh must verify the install succeeds"


def test_install_ps1_verifies_install() -> None:
    content = (REPO_ROOT / "install.ps1").read_text()
    assert (
        "--help" in content or "verify" in content.lower()
    ), "install.ps1 must verify the install succeeds"


# ---------------------------------------------------------------------------
# AC3 — success message
# ---------------------------------------------------------------------------


def test_install_sh_success_message() -> None:
    content = (REPO_ROOT / "install.sh").read_text()
    assert (
        "Rex is installed" in content
    ), "install.sh must print 'Rex is installed. Run `rex` to start.' on success"
    assert "rex" in content, "install.sh success message must reference the `rex` command"


def test_install_ps1_success_message() -> None:
    content = (REPO_ROOT / "install.ps1").read_text()
    assert (
        "Rex is installed" in content
    ), "install.ps1 must print 'Rex is installed. Run `rex` to start.' on success"


# ---------------------------------------------------------------------------
# AC4 — failure exits non-zero with a specific message
# ---------------------------------------------------------------------------


def test_install_sh_has_error_handling() -> None:
    content = (REPO_ROOT / "install.sh").read_text()
    assert (
        "fail" in content.lower() or "exit 1" in content
    ), "install.sh must exit with a non-zero code on failure"
    assert (
        "ERROR" in content or "fail()" in content
    ), "install.sh must print a specific error message on failure"


def test_install_ps1_has_error_handling() -> None:
    content = (REPO_ROOT / "install.ps1").read_text()
    assert (
        "Fail" in content or "exit" in content
    ), "install.ps1 must exit with a non-zero code on failure"
    assert (
        "ERROR" in content or "Fail" in content
    ), "install.ps1 must print a specific error message on failure"


# ---------------------------------------------------------------------------
# Executable bit (Linux/macOS) — best-effort on Windows
# ---------------------------------------------------------------------------


@pytest.mark.skipif(os.name == "nt", reason="executable bit not relevant on Windows")
def test_install_sh_is_executable() -> None:
    path = REPO_ROOT / "install.sh"
    mode = path.stat().st_mode
    assert mode & stat.S_IXUSR, "install.sh must have the executable bit set"
