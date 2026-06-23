#!/usr/bin/env python3
"""Wheel contents smoke test for askrex-assistant (US-015, US-016).

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
# Each tuple is (logical_path, install_audience, description).
#
# "logical_path" is the path as it would appear directly in the wheel zip for
# package files (e.g. "rex/cli.py"), or the suffix after ".data/data/" for
# data files installed via data_files (e.g. "bridge/rex_chat_bridge.py",
# which appears in the zip as "{name}-{version}.data/data/bridge/rex_chat_bridge.py").
#
# check_wheel() resolves both forms — see its docstring for details.
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
    # -- Bridge scripts (Electron desktop app) — US-016 ----------------------
    # These arrive in the wheel via data_files in setup.py.
    # In the zip they appear as "{name}-{version}.data/data/bridge/{file}".
    # check_wheel() accepts both the exact zip path and the logical suffix form.
    ("bridge/rex_calendar_bridge.py", "Electron desktop app", "calendar bridge"),
    ("bridge/rex_chat_bridge.py", "Electron desktop app", "chat bridge"),
    ("bridge/rex_chat_stream_bridge.py", "Electron desktop app", "chat stream bridge"),
    ("bridge/rex_email_bridge.py", "Electron desktop app", "email bridge"),
    ("bridge/rex_file_extract_bridge.py", "Electron desktop app", "file-extract bridge"),
    ("bridge/rex_history_bridge.py", "Electron desktop app", "history bridge"),
    ("bridge/rex_memories_bridge.py", "Electron desktop app", "memories bridge"),
    ("bridge/rex_quick_actions_bridge.py", "Electron desktop app", "quick-actions bridge"),
    ("bridge/rex_reminders_bridge.py", "Electron desktop app", "reminders bridge"),
    ("bridge/rex_setup_bridge.py", "Electron desktop app", "setup bridge"),
    ("bridge/rex_shopping_list_bridge.py", "Electron desktop app", "shopping-list bridge"),
    ("bridge/rex_sms_bridge.py", "Electron desktop app", "SMS bridge"),
    ("bridge/rex_speaker_bridge.py", "Electron desktop app", "speaker bridge"),
    ("bridge/rex_stt_bridge.py", "Electron desktop app", "STT bridge"),
    ("bridge/rex_tasks_bridge.py", "Electron desktop app", "tasks bridge"),
    ("bridge/rex_voice_bridge.py", "Electron desktop app", "voice bridge"),
    ("bridge/rex_voice_enrollment_bridge.py", "Electron desktop app", "voice enrollment bridge"),
    ("bridge/rex_voice_sample_bridge.py", "Electron desktop app", "voice sample bridge"),
    ("bridge/rex_voice_upload_bridge.py", "Electron desktop app", "voice upload bridge"),
    ("bridge/rex_voices_bridge.py", "Electron desktop app", "voices bridge"),
    ("bridge/rex_wakeword_list_bridge.py", "Electron desktop app", "wake-word list bridge"),
    ("bridge/rex_wakeword_sample_bridge.py", "Electron desktop app", "wake-word sample bridge"),
    ("bridge/rex_wakeword_train_bridge.py", "Electron desktop app", "wake-word train bridge"),
    # -- Config example — US-016 ---------------------------------------------
    # Arrives via data_files; zip path: "{name}-{version}.data/data/config/{file}".
    (
        "config/rex_config.example.json",
        "developers/operators",
        "example configuration file documented in INSTALL.md",
    ),
]


def check_wheel(wheel_path: Path) -> list[tuple[str, str, str]]:
    """Return list of (logical_path, audience, description) tuples for missing entries.

    Two matching strategies are applied for each REQUIRED_ENTRIES entry:

    1. **Exact match**: the entry path appears verbatim in the wheel zip (used for
       package files such as ``rex/cli.py``).

    2. **Data-file suffix match**: the wheel zip contains an entry of the form
       ``{name}-{version}.data/data/{suffix}`` where ``suffix`` equals the entry
       path (used for files installed via ``data_files`` in setup.py, such as
       ``bridge/rex_chat_bridge.py``).

    This means REQUIRED_ENTRIES can use logical paths like ``bridge/rex_chat_bridge.py``
    regardless of which setuptools version or wheel name is in use.
    """
    with zipfile.ZipFile(wheel_path) as zf:
        present = set(zf.namelist())

    # Build suffix → full_path mapping for data-file entries.
    # Data files in a wheel appear as "{name}-{version}.data/data/{suffix}".
    data_suffixes: dict[str, str] = {}
    for name in present:
        if ".data/data/" in name:
            suffix = name.split(".data/data/", 1)[1]
            data_suffixes[suffix] = name

    missing = []
    for entry, audience, description in REQUIRED_ENTRIES:
        if entry in present:
            continue
        if entry in data_suffixes:
            continue
        missing.append((entry, audience, description))
    return missing


def build_wheel(repo_root: Path) -> Path:
    """Run `python -m build --wheel` and return the resulting .whl path."""
    subprocess.run(
        [sys.executable, "-m", "build", "--wheel"],
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
