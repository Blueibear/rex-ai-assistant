"""
US-303 verification tests: bridge scripts exist and respond gracefully.

These tests serve as the automated proxy for the manual Electron verification
criterion: "launch the Electron app and confirm Tasks, Reminders, and Shopping
List pages load without 'bridge exited' errors".

A bridge exits with a non-zero code on bad input, NOT on import/startup, which
is the runtime condition that would produce "bridge exited" errors in Electron.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent

# All bridges that must exist at repo root (mirrors BRIDGE_REGISTRY in bridgeResolver.ts)
ALL_BRIDGE_SCRIPTS = [
    "rex_tasks_bridge.py",
    "rex_reminders_bridge.py",
    "rex_shopping_list_bridge.py",
    "rex_speaker_bridge.py",
    "rex_chat_bridge.py",
    "rex_chat_stream_bridge.py",
    "rex_voices_bridge.py",
    "rex_voice_enrollment_bridge.py",
    "rex_voice_sample_bridge.py",
    "rex_voice_upload_bridge.py",
    "rex_wakeword_list_bridge.py",
    "rex_wakeword_train_bridge.py",
    "rex_wakeword_sample_bridge.py",
    "rex_stt_bridge.py",
    "rex_memories_bridge.py",
    "rex_file_extract_bridge.py",
    "rex_voice_bridge.py",
]

# Bridges critical for Tasks / Reminders / Shopping List pages
CRITICAL_BRIDGES = [
    "rex_tasks_bridge.py",
    "rex_reminders_bridge.py",
    "rex_shopping_list_bridge.py",
]


@pytest.mark.parametrize("script", ALL_BRIDGE_SCRIPTS)
def test_bridge_script_exists(script):
    """Every bridge registered in bridgeResolver.ts must exist at repo root."""
    path = REPO_ROOT / script
    assert path.exists(), f"Missing bridge script: {path}"


@pytest.mark.parametrize("script", CRITICAL_BRIDGES)
def test_critical_bridge_responds_to_bad_json(script):
    """
    Tasks / Reminders / Shopping bridges must start and return a JSON error
    response for bad input rather than crashing (exit code != 0 with no output).

    In Electron, an empty stdout + non-zero exit is reported as "bridge exited".
    A well-formed JSON error response means the bridge at least started OK.
    """
    path = REPO_ROOT / script
    result = subprocess.run(
        [sys.executable, str(path)],
        input=b"",  # empty stdin — invalid JSON
        capture_output=True,
        timeout=30,
        cwd=str(REPO_ROOT),
    )
    # Bridge must produce some output (not silent crash)
    stdout_text = result.stdout.decode("utf-8", errors="replace").strip()
    assert stdout_text, (
        f"{script} produced no stdout on bad input — Electron would show 'bridge exited' error. "
        f"stderr: {result.stderr.decode('utf-8', errors='replace')[:500]}"
    )

    # Output must be valid JSON (bridges communicate via JSON stdout)
    json_lines = [line for line in stdout_text.splitlines() if line.startswith("{")]
    assert json_lines, f"{script} produced no JSON output lines. stdout: {stdout_text[:500]}"
    last_json = json.loads(json_lines[-1])
    assert isinstance(last_json, dict), f"{script} last JSON line is not an object"


def test_bridge_registry_completeness():
    """All scripts listed in ALL_BRIDGE_SCRIPTS exist — validates the list itself."""
    missing = [s for s in ALL_BRIDGE_SCRIPTS if not (REPO_ROOT / s).exists()]
    assert not missing, f"Bridge scripts missing from repo root: {missing}"
