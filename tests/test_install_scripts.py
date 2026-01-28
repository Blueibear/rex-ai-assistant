from __future__ import annotations

import os
import subprocess
from pathlib import Path


def test_install_lean_script_dry_run(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "install_lean.sh"

    env = os.environ.copy()
    env["REX_DRY_RUN"] = "1"
    env["REX_SKIP_SERVICE"] = "1"

    result = subprocess.run(
        ["bash", str(script)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "DRY RUN" in result.stdout
