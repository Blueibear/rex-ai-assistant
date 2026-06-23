"""Tests for scripts/check_wheel_contents.py (US-015).

These tests use synthetic wheel ZIP archives so they run fast without
building the real package.  They verify:
  - check_wheel() returns [] when all required files are present.
  - check_wheel() names each missing file together with its audience.
  - main() exits 0 on a complete synthetic wheel.
  - main() exits 1 and reports each missing file when files are absent.
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
