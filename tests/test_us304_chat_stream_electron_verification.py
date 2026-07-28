from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
GUI_DIR = REPO_ROOT / "gui"
VERIFY_SCRIPT = GUI_DIR / "tmp_verify_chat_stream.cjs"
ELECTRON_CMD = GUI_DIR / "node_modules" / ".bin" / "electron.cmd"


@pytest.mark.skipif(not ELECTRON_CMD.exists(), reason="electron.cmd not found")
def test_chat_streaming_works_in_electron(tmp_path: Path) -> None:
    """Electron chat page renders a typed streamed reply incrementally."""
    assert VERIFY_SCRIPT.exists(), f"Missing Electron verification script: {VERIFY_SCRIPT}"
    assert ELECTRON_CMD.exists(), f"Missing Electron binary: {ELECTRON_CMD}"

    app_data = tmp_path / "appdata"
    session_dir = app_data / "rex-ai"
    session_dir.mkdir(parents=True)
    (session_dir / "session.json").write_text(
        json.dumps({"active_user": "electron-test-user"}), encoding="utf-8"
    )
    env = os.environ.copy()
    env["LOCALAPPDATA"] = str(app_data)

    try:
        result = subprocess.run(
            [str(ELECTRON_CMD), str(VERIFY_SCRIPT)],
            cwd=str(GUI_DIR),
            capture_output=True,
            text=True,
            timeout=120,
            env=env,
        )
    except OSError as exc:
        pytest.skip(f"Electron verification unavailable: {exc}")

    stdout_lines = [line for line in result.stdout.splitlines() if line.strip()]
    assert stdout_lines, f"Electron verification produced no stdout. stderr: {result.stderr[:500]}"

    payload = json.loads(stdout_lines[-1])
    assert result.returncode == 0 and payload.get("ok") is True, (
        "Electron chat verification failed.\n"
        f"stdout: {result.stdout[:1000]}\n"
        f"stderr: {result.stderr[:1000]}"
    )

    verification = payload["result"]
    assert verification["sawUserMessage"] is True
    assert verification["sawFirstPartial"] is True
    assert verification["sawSecondPartial"] is True
    assert verification["sawFinal"] is True
