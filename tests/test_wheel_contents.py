"""Tests for scripts/check_wheel_contents.py (US-015, US-016).

These tests use synthetic wheel ZIP archives so they run fast without
building the real package.  They verify:
  - check_wheel() returns [] when all required files are present.
  - check_wheel() names each missing file together with its audience.
  - main() exits 0 on a complete synthetic wheel.
  - main() exits 1 and reports each missing file when files are absent.
  - data-file entries (bridge scripts, config) are matched via the
    "{name}-{version}.data/data/{suffix}" form used in real wheels.
"""

from __future__ import annotations

import sys
import zipfile
from io import BytesIO
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
import check_wheel_contents as script  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FAKE_DATA_PREFIX = "askrex_assistant-0.1.0.data/data"


def _make_wheel(tmp_path: Path, entries: list[str]) -> Path:
    """Create a minimal fake .whl ZIP at tmp_path/fake.whl containing *entries*."""
    wheel_path = tmp_path / "fake_wheel.whl"
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for entry in entries:
            zf.writestr(entry, b"")
    wheel_path.write_bytes(buf.getvalue())
    return wheel_path


def _all_required_entries() -> list[str]:
    """Return every path from the REQUIRED_ENTRIES constant."""
    return [path for path, _audience, _desc in script.REQUIRED_ENTRIES]


def _make_wheel_with_data_files(tmp_path: Path) -> Path:
    """Create a fake wheel where data-file entries use the real .data/data/ format.

    Package entries (rex/*, *.py at root) are stored verbatim.
    Data-file entries (bridge/*, config/*) are stored under _FAKE_DATA_PREFIX.
    This mirrors what `python -m build --wheel` produces when setup.py uses
    data_files.
    """
    entries = []
    for path, _audience, _desc in script.REQUIRED_ENTRIES:
        if path.startswith("bridge/") or path.startswith("config/"):
            entries.append(f"{_FAKE_DATA_PREFIX}/{path}")
        else:
            entries.append(path)
    return _make_wheel(tmp_path, entries)


# ---------------------------------------------------------------------------
# check_wheel() — core logic
# ---------------------------------------------------------------------------


def test_check_wheel_all_present_returns_empty(tmp_path: Path) -> None:
    wheel = _make_wheel(tmp_path, _all_required_entries())
    assert script.check_wheel(wheel) == []


def test_check_wheel_missing_one_returns_that_entry(tmp_path: Path) -> None:
    all_entries = _all_required_entries()
    missing_entry = all_entries[0]
    wheel = _make_wheel(tmp_path, all_entries[1:])
    result = script.check_wheel(wheel)
    assert len(result) == 1
    assert result[0][0] == missing_entry


def test_check_wheel_missing_multiple_returns_all(tmp_path: Path) -> None:
    all_entries = _all_required_entries()
    wheel = _make_wheel(tmp_path, [])  # empty wheel
    result = script.check_wheel(wheel)
    assert len(result) == len(all_entries)


def test_check_wheel_extra_files_do_not_cause_failure(tmp_path: Path) -> None:
    entries = _all_required_entries() + [
        "some_extra/module.py",
        "askrex_assistant-0.1.0.dist-info/RECORD",
    ]
    wheel = _make_wheel(tmp_path, entries)
    assert script.check_wheel(wheel) == []


def test_check_wheel_missing_entry_includes_audience(tmp_path: Path) -> None:
    wheel = _make_wheel(tmp_path, [])  # empty
    result = script.check_wheel(wheel)
    for _path, audience, _desc in result:
        assert audience, "audience must be non-empty for every missing entry"


def test_check_wheel_missing_entry_includes_description(tmp_path: Path) -> None:
    wheel = _make_wheel(tmp_path, [])  # empty
    result = script.check_wheel(wheel)
    for _path, _audience, description in result:
        assert description, "description must be non-empty for every missing entry"


# ---------------------------------------------------------------------------
# REQUIRED_ENTRIES structure validation
# ---------------------------------------------------------------------------


def test_required_entries_is_nonempty() -> None:
    assert len(script.REQUIRED_ENTRIES) > 0, "REQUIRED_ENTRIES must not be empty"


def test_required_entries_no_duplicates() -> None:
    paths = [p for p, _, _ in script.REQUIRED_ENTRIES]
    assert len(paths) == len(set(paths)), "REQUIRED_ENTRIES contains duplicate paths"


def test_required_entries_all_have_audience_and_description() -> None:
    for path, audience, description in script.REQUIRED_ENTRIES:
        assert path, "path must be non-empty"
        assert audience, f"audience missing for entry: {path!r}"
        assert description, f"description missing for entry: {path!r}"


# ---------------------------------------------------------------------------
# main() exit codes
# ---------------------------------------------------------------------------


def test_main_exits_0_when_wheel_is_complete(tmp_path: Path) -> None:
    wheel = _make_wheel(tmp_path, _all_required_entries())
    rc = script.main([str(wheel)])
    assert rc == 0


def test_main_exits_1_when_wheel_is_missing_files(tmp_path: Path) -> None:
    wheel = _make_wheel(tmp_path, [])
    rc = script.main([str(wheel)])
    assert rc == 1


def test_main_exits_1_for_nonexistent_wheel(tmp_path: Path) -> None:
    rc = script.main([str(tmp_path / "nonexistent.whl")])
    assert rc == 1


def test_main_exits_0_with_extra_dist_info_files(tmp_path: Path) -> None:
    entries = _all_required_entries() + [
        "askrex_assistant-0.1.0.dist-info/WHEEL",
        "askrex_assistant-0.1.0.dist-info/METADATA",
        "askrex_assistant-0.1.0.dist-info/RECORD",
        "askrex_assistant-0.1.0.dist-info/entry_points.txt",
    ]
    wheel = _make_wheel(tmp_path, entries)
    rc = script.main([str(wheel)])
    assert rc == 0


# ---------------------------------------------------------------------------
# Console-script modules must be in REQUIRED_ENTRIES
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "module_path",
    [
        "rex/cli.py",
        "rex/config.py",
        "rex/gui_app.py",
        "rex/computers/agent_server.py",
        "rex/openclaw/tool_server.py",
        "rex_speak_api.py",
    ],
)
def test_console_script_module_is_required(module_path: str) -> None:
    required_paths = {p for p, _, _ in script.REQUIRED_ENTRIES}
    assert (
        module_path in required_paths
    ), f"{module_path!r} is a console-script module and must be in REQUIRED_ENTRIES"


def test_py_typed_marker_is_required() -> None:
    required_paths = {p for p, _, _ in script.REQUIRED_ENTRIES}
    assert (
        "rex/py.typed" in required_paths
    ), "rex/py.typed (PEP 561 type marker) must be in REQUIRED_ENTRIES"


# ---------------------------------------------------------------------------
# US-016: data-file path resolution (bridge scripts + config example)
# ---------------------------------------------------------------------------


def test_check_wheel_data_file_paths_accepted(tmp_path: Path) -> None:
    """A wheel with data-file format entries passes the check."""
    wheel = _make_wheel_with_data_files(tmp_path)
    assert script.check_wheel(wheel) == []


def test_check_wheel_missing_bridge_script_is_reported(tmp_path: Path) -> None:
    """A wheel missing one bridge script is reported as missing."""
    entries = []
    first_bridge = None
    for path, _audience, _desc in script.REQUIRED_ENTRIES:
        if path.startswith("bridge/") and first_bridge is None:
            first_bridge = path
            continue
        if path.startswith("bridge/") or path.startswith("config/"):
            entries.append(f"{_FAKE_DATA_PREFIX}/{path}")
        else:
            entries.append(path)
    wheel = _make_wheel(tmp_path, entries)
    result = script.check_wheel(wheel)
    assert len(result) == 1
    assert result[0][0] == first_bridge


def test_check_wheel_accepts_data_file_via_suffix_match(tmp_path: Path) -> None:
    """check_wheel() matches bridge entries stored under the .data/data/ prefix."""
    entries = [
        f"{_FAKE_DATA_PREFIX}/bridge/rex_chat_bridge.py",
        f"{_FAKE_DATA_PREFIX}/bridge/rex_chat_bridge.py",  # duplicate is fine
    ]
    wheel = _make_wheel(tmp_path, entries)
    result = script.check_wheel(wheel)
    # bridge/rex_chat_bridge.py must NOT appear in missing
    missing_paths = {r[0] for r in result}
    assert "bridge/rex_chat_bridge.py" not in missing_paths


def test_config_example_is_required() -> None:
    """config/rex_config.example.json must be in REQUIRED_ENTRIES."""
    required_paths = {p for p, _, _ in script.REQUIRED_ENTRIES}
    assert (
        "config/rex_config.example.json" in required_paths
    ), "config/rex_config.example.json must be in REQUIRED_ENTRIES (US-016)"


@pytest.mark.parametrize(
    "bridge_path",
    [
        "bridge/rex_chat_bridge.py",
        "bridge/rex_chat_stream_bridge.py",
        "bridge/rex_voice_bridge.py",
        "bridge/rex_stt_bridge.py",
        "bridge/rex_speaker_bridge.py",
        "bridge/rex_voices_bridge.py",
        "bridge/rex_tasks_bridge.py",
        "bridge/rex_reminders_bridge.py",
    ],
)
def test_core_bridge_scripts_are_required(bridge_path: str) -> None:
    required_paths = {p for p, _, _ in script.REQUIRED_ENTRIES}
    assert (
        bridge_path in required_paths
    ), f"{bridge_path!r} is a core bridge script and must be in REQUIRED_ENTRIES"
