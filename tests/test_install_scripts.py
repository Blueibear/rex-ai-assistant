from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest


def test_install_lean_script_dry_run(tmp_path: Path) -> None:
    if os.name == "nt":
        bash_path = shutil.which("bash")
        if bash_path is None:
            pytest.skip("bash not available")
        probe = subprocess.run(
            ["bash", "-lc", "echo ok"],
            capture_output=True,
            text=True,
            check=False,
        )
        stderr = probe.stderr or ""
        if probe.returncode != 0 or "WSL" in stderr or "execvpe(/bin/bash) failed" in stderr:
            pytest.skip("bash not usable on Windows")
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "install_lean.sh"

    # On Windows, shell scripts may have CRLF line endings which bash rejects.
    # Copy the script to tmp_path with LF endings (binary write to prevent re-CRLF).
    script_bytes = script.read_bytes().replace(b"\r\n", b"\n")
    tmp_script = tmp_path / "install_lean.sh"
    tmp_script.write_bytes(script_bytes)

    env = os.environ.copy()
    env["REX_DRY_RUN"] = "1"
    env["REX_SKIP_SERVICE"] = "1"

    # On Windows, spawning bash directly via subprocess does not propagate env
    # vars set in Python (Git Bash / MSYS2 quirk).  Using shell=True routes
    # through cmd.exe which correctly inherits the modified environment.
    if os.name == "nt":
        cmd: str | list[str] = 'bash "install_lean.sh"'
        use_shell = True
    else:
        cmd = ["bash", "install_lean.sh"]
        use_shell = False

    result = subprocess.run(
        cmd,
        cwd=str(tmp_path),
        env=env,
        capture_output=True,
        text=True,
        shell=use_shell,
        check=False,
    )

    assert result.returncode == 0
    assert "DRY RUN" in result.stdout
