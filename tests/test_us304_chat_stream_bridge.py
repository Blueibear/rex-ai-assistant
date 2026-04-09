"""Tests for US-304: GUI text chat streaming bridge."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
BRIDGE_SCRIPT = REPO_ROOT / "rex_chat_stream_bridge.py"


# ---------------------------------------------------------------------------
# AC: --help exits 0
# ---------------------------------------------------------------------------


def test_chat_stream_bridge_help_exits_0():
    """python rex_chat_stream_bridge.py --help must exit 0."""
    result = subprocess.run(
        [sys.executable, str(BRIDGE_SCRIPT), "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"Expected exit code 0 from --help, got {result.returncode}.\n"
        f"stdout: {result.stdout[:500]}\nstderr: {result.stderr[:500]}"
    )


def test_chat_stream_bridge_help_mentions_stdin():
    """--help output should describe the stdin JSON protocol."""
    result = subprocess.run(
        [sys.executable, str(BRIDGE_SCRIPT), "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    combined = result.stdout + result.stderr
    assert "stdin" in combined.lower() or "message" in combined.lower(), (
        f"--help output should mention stdin or message protocol, got:\n{combined[:500]}"
    )


# ---------------------------------------------------------------------------
# AC: bridge uses Assistant.generate_reply() not a direct LLM call
# ---------------------------------------------------------------------------


def test_chat_stream_bridge_uses_assistant_generate_reply():
    """The streaming bridge must call Assistant.generate_reply() (or stream_reply)."""
    source = BRIDGE_SCRIPT.read_text(encoding="utf-8")
    assert "Assistant" in source, "Bridge must import/use the Assistant class"
    assert "generate_reply" in source or "stream_reply" in source, (
        "Bridge must call generate_reply or stream_reply on an Assistant instance"
    )


# ---------------------------------------------------------------------------
# AC: error path emits a structured JSON error (not raw exit code 2)
# ---------------------------------------------------------------------------


def test_chat_stream_bridge_emits_json_error_on_bad_input():
    """Malformed stdin produces a JSON error line, not a raw Python traceback."""
    import json

    result = subprocess.run(
        [sys.executable, str(BRIDGE_SCRIPT)],
        input="not-valid-json",
        capture_output=True,
        text=True,
        timeout=30,
    )
    # The bridge should emit a {"type": "error", ...} line somewhere in stdout
    # (other lines may be config warnings from the logging setup)
    json_lines = [
        line for line in result.stdout.splitlines()
        if line.strip().startswith("{")
    ]
    assert json_lines, (
        f"Expected at least one JSON line in stdout on bad input.\n"
        f"stdout: {result.stdout[:500]}\nstderr: {result.stderr[:300]}"
    )
    last_json_line = json_lines[-1]
    try:
        obj = json.loads(last_json_line)
    except json.JSONDecodeError:
        pytest.fail(f"Last JSON line in stdout was not valid JSON: {last_json_line[:300]}")
    assert obj.get("type") == "error", f"Expected type=error, got: {obj}"
    assert "error" in obj, "Error payload must have an 'error' key"


# ---------------------------------------------------------------------------
# AC: bridge script exists and is importable (no syntax errors)
# ---------------------------------------------------------------------------


def test_chat_stream_bridge_exists():
    """rex_chat_stream_bridge.py must exist at repo root."""
    assert BRIDGE_SCRIPT.exists(), f"Missing bridge script: {BRIDGE_SCRIPT}"


def test_chat_stream_bridge_no_syntax_errors():
    """Bridge script must compile without syntax errors."""
    result = subprocess.run(
        [sys.executable, "-m", "py_compile", str(BRIDGE_SCRIPT)],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, (
        f"Syntax error in {BRIDGE_SCRIPT.name}:\n{result.stderr}"
    )
