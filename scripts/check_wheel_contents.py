#!/usr/bin/env python3
"""Wheel contents smoke test for askrex-assistant (US-015).

Builds dist/askrex_assistant-*.whl and asserts every required file is present.
Exits non-zero and names each missing file together with the install audience
that requires it.

Usage:
    python scripts/check_wheel_contents.py              # build then check
    python scripts/check_wheel_contents.py <wheel.whl>  # check a pre-built wheel
"""

from __future__ import annotations

import glob
import subprocess
import sys
import zipfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Required wheel entries
#
# Each tuple is (wheel_path, install_audience, description).
# "wheel_path" is the path as it appears inside the .whl zip archive.
#
# US-016 will add bridge scripts and config/rex_config.example.json here
# once those resources are wired into the packaging configuration.
# ---------------------------------------------------------------------------
REQUIRED_ENTRIES: list[tuple[str, str, str]] = [
    # -- Console script entry-point modules ----------------------------------
    # Every declared [project.scripts] entry must have its source module
    # present so `pip install` wires the console script correctly.
    ("rex/__init__.py", "developers/operators", "rex package (all console scripts)"),
    ("rex/cli.py", "developers/operators", "rex console script (rex.cli:main)"),
    ("rex/config.py", "developers/operators", "rex-config console script (rex.config:cli)"),
    ("rex/gui_app.py", "developers/operators", "rex-gui console script (rex.gui_app:main)"),
    (
        "rex/computers/agent_server.py",
        "developers/operators",
        "rex-agent console script (rex.computers.agent_server:main)",
    ),
    (
        "rex/openclaw/tool_server.py",
        "developers/operators",
        "rex-tool-server console script (rex.openclaw.tool_server:main)",
    ),
    # -- Root py_modules: backward-compat shims and entry-point module -------
    (
        "rex_speak_api.py",
        "developers/operators",
        "rex-speak-api entry-point module (rex_speak_api:main)",
    ),
    (
        "config.py",
        "developers/operators",
        "config compat shim (→ rex.config; used by test suite)",
    ),
    (
        "llm_client.py",
        "developers/operators",
        "llm_client compat shim (→ rex.llm_client; used by test suite)",
    ),
    # -- Type marker ---------------------------------------------------------
    # PEP 561: presence signals that the rex package ships inline type stubs.
    ("rex/py.typed", "developers/operators (type checkers)", "PEP 561 type marker"),
    # -- Bridge scripts (Electron desktop app) — added in US-016 -------------
    # ("bridge/rex_chat_bridge.py",          "Electron desktop app", "chat bridge"),
    # ("bridge/rex_voice_bridge.py",         "Electron desktop app", "voice bridge"),
    # ("bridge/rex_stt_bridge.py",           "Electron desktop app", "STT bridge"),
    # ("bridge/rex_speaker_bridge.py",       "Electron desktop app", "speaker bridge"),
    # ("bridge/rex_voices_bridge.py",        "Electron desktop app", "voices bridge"),
    # ("bridge/rex_memories_bridge.py",      "Electron desktop app", "memories bridge"),
    # ("bridge/rex_reminders_bridge.py",     "Electron desktop app", "reminders bridge"),
    # ("bridge/rex_tasks_bridge.py",         "Electron desktop app", "tasks bridge"),
    # ("bridge/rex_quick_actions_bridge.py", "Electron desktop app", "quick-actions bridge"),
    # ("bridge/rex_setup_bridge.py",         "Electron desktop app", "setup bridge"),
    # ("bridge/rex_history_bridge.py",       "Electron desktop app", "history bridge"),
    # ("bridge/rex_shopping_list_bridge.py", "Electron desktop app", "shopping-list bridge"),
    # ("bridge/rex_file_extract_bridge.py",  "Electron desktop app", "file-extract bridge"),
    # ("bridge/rex_email_bridge.py",         "Electron desktop app", "email bridge"),
    # ("bridge/rex_sms_bridge.py",           "Electron desktop app", "SMS bridge"),
    # ("bridge/rex_calendar_bridge.py",      "Electron desktop app", "calendar bridge"),
    # ("bridge/rex_wakeword_list_bridge.py", "Electron desktop app", "wake-word list bridge"),
    # ("bridge/rex_wakeword_sample_bridge.py","Electron desktop app","wake-word sample bridge"),
    # ("bridge/rex_wakeword_train_bridge.py","Electron desktop app", "wake-word train bridge"),
    # ("bridge/rex_voice_bridge.py",         "Electron desktop app", "voice bridge"),
    # ("bridge/rex_voice_enrollment_bridge.py","Electron desktop app","voice enrollment bridge"),
    # ("bridge/rex_voice_sample_bridge.py",  "Electron desktop app", "voice sample bridge"),
    # ("bridge/rex_voice_upload_bridge.py",  "Electron desktop app", "voice upload bridge"),
    # -- Config example — added in US-016 ------------------------------------
    # ("config/rex_config.example.json", "developers/operators",
    #  "example configuration file documented in INSTALL.md"),
]


def check_wheel(wheel_path: Path) -> list[tuple[str, str, str]]:
    """Return list of (wheel_path, audience, description) tuples for missing entries."""
    with zipfile.ZipFile(wheel_path) as zf:
        present = set(zf.namelist())

    missing = []
    for entry, audience, description in REQUIRED_ENTRIES:
        if entry not in present:
            missing.append((entry, audience, description))
    return missing


def build_wheel(repo_root: Path) -> Path:
    """Run `python -m build --wheel` and return the resulting .whl path."""
    subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "--no-isolation"],
        cwd=repo_root,
        check=True,
    )
    wheels = sorted(glob.glob(str(repo_root / "dist" / "askrex_assistant-*.whl")))
    if not wheels:
        raise FileNotFoundError("No wheel found in dist/ after build")
    return Path(wheels[-1])


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    repo_root = Path(__file__).parent.parent

    if args:
        wheel_path = Path(args[0])
        if not wheel_path.exists():
            print(f"ERROR: wheel not found: {wheel_path}", file=sys.stderr)
            return 1
    else:
        print("Building wheel …")
        try:
            wheel_path = build_wheel(repo_root)
        except subprocess.CalledProcessError as exc:
            print(f"ERROR: wheel build failed: {exc}", file=sys.stderr)
            return 1
        except FileNotFoundError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        print(f"Built: {wheel_path}")

    missing = check_wheel(wheel_path)
    if missing:
        print(f"FAIL: {len(missing)} required file(s) missing from {wheel_path.name}:")
        for entry, audience, description in missing:
            print(f"  MISSING  {entry}")
            print(f"           audience:    {audience}")
            print(f"           description: {description}")
        return 1

    print(f"OK: all required files present in {wheel_path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
