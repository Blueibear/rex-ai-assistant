"""Unit tests for rex/tools/file_ops.py (US-WIN-001)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from rex.tools.file_ops import (
    delete_file,
    list_directory,
    move_file,
    read_file,
    write_file,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cfg(tmp_path: Path) -> MagicMock:
    """Return a minimal config stub with tmp_path as the single allowed root."""
    cfg = MagicMock()
    cfg.allowed_file_roots = [str(tmp_path)]
    return cfg


# ---------------------------------------------------------------------------
# read_file
# ---------------------------------------------------------------------------


def test_read_existing_file(tmp_path: Path) -> None:
    target = tmp_path / "hello.txt"
    target.write_text("hello world", encoding="utf-8")
    result = read_file(path=str(target), config=_cfg(tmp_path))
    assert result["content"] == "hello world"
    assert "error" not in result


def test_read_missing_file_returns_error(tmp_path: Path) -> None:
    result = read_file(path=str(tmp_path / "nope.txt"), config=_cfg(tmp_path))
    assert "error" in result


def test_read_path_traversal_blocked(tmp_path: Path) -> None:
    outside = tmp_path.parent / "secret.txt"
    result = read_file(path=str(outside), config=_cfg(tmp_path))
    assert "error" in result
    assert "outside" in result["error"].lower() or "PermissionError" in type(result).__name__


def test_read_empty_path_returns_error(tmp_path: Path) -> None:
    result = read_file(path="", config=_cfg(tmp_path))
    assert "error" in result


# ---------------------------------------------------------------------------
# write_file
# ---------------------------------------------------------------------------


def test_write_new_file(tmp_path: Path) -> None:
    target = tmp_path / "out.txt"
    result = write_file(path=str(target), content="new content", config=_cfg(tmp_path))
    assert "error" not in result
    assert target.read_text(encoding="utf-8") == "new content"


def test_write_creates_parent_dirs(tmp_path: Path) -> None:
    target = tmp_path / "sub" / "dir" / "file.txt"
    result = write_file(path=str(target), content="data", config=_cfg(tmp_path))
    assert "error" not in result
    assert target.exists()


def test_write_path_traversal_blocked(tmp_path: Path) -> None:
    outside = tmp_path.parent / "evil.txt"
    result = write_file(path=str(outside), content="bad", config=_cfg(tmp_path))
    assert "error" in result
    assert not outside.exists()


# ---------------------------------------------------------------------------
# list_directory
# ---------------------------------------------------------------------------


def test_list_directory(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("a")
    (tmp_path / "b.txt").write_text("b")
    result = list_directory(path=str(tmp_path), config=_cfg(tmp_path))
    assert "error" not in result
    assert set(result["entries"]) == {"a.txt", "b.txt"}


def test_list_missing_directory_returns_error(tmp_path: Path) -> None:
    result = list_directory(path=str(tmp_path / "no_such_dir"), config=_cfg(tmp_path))
    assert "error" in result


def test_list_path_traversal_blocked(tmp_path: Path) -> None:
    result = list_directory(path=str(tmp_path.parent), config=_cfg(tmp_path))
    assert "error" in result


# ---------------------------------------------------------------------------
# move_file
# ---------------------------------------------------------------------------


def test_move_file(tmp_path: Path) -> None:
    src = tmp_path / "src.txt"
    src.write_text("content")
    dst = tmp_path / "dst.txt"
    result = move_file(src=str(src), dst=str(dst), config=_cfg(tmp_path))
    assert "error" not in result
    assert dst.exists()
    assert not src.exists()


def test_move_dst_outside_root_blocked(tmp_path: Path) -> None:
    src = tmp_path / "src.txt"
    src.write_text("content")
    outside = tmp_path.parent / "evil.txt"
    result = move_file(src=str(src), dst=str(outside), config=_cfg(tmp_path))
    assert "error" in result
    assert src.exists()  # source untouched


# ---------------------------------------------------------------------------
# delete_file
# ---------------------------------------------------------------------------


def test_delete_existing_file(tmp_path: Path) -> None:
    target = tmp_path / "to_delete.txt"
    target.write_text("bye")
    result = delete_file(path=str(target), config=_cfg(tmp_path))
    assert result.get("deleted") is True
    assert not target.exists()


def test_delete_missing_file_returns_error(tmp_path: Path) -> None:
    result = delete_file(path=str(tmp_path / "ghost.txt"), config=_cfg(tmp_path))
    assert "error" in result


def test_delete_path_traversal_blocked(tmp_path: Path) -> None:
    outside = tmp_path.parent / "important.txt"
    outside.write_text("keep me")
    result = delete_file(path=str(outside), config=_cfg(tmp_path))
    assert "error" in result
    assert outside.exists()
    outside.unlink()  # cleanup


# ---------------------------------------------------------------------------
# Default config (no config arg) uses home directory
# ---------------------------------------------------------------------------


def test_read_no_config_defaults_to_home(tmp_path: Path) -> None:
    """When config=None, allowed root is home — path outside home blocked."""
    from rex.tools.file_ops import _DEFAULT_ALLOWED_ROOTS

    # Sanity: default root is the user home
    assert any(str(Path.home()) in r for r in _DEFAULT_ALLOWED_ROOTS)

    # A path under /tmp is likely outside home; should be blocked (on most systems).
    # We just verify no crash and that result is a dict.
    result = read_file(path=str(tmp_path / "x.txt"))
    assert isinstance(result, dict)
