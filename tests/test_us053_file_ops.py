"""Tests for US-053: Desktop file read/write capability.

Covers:
- read_file: happy path, blocked path, missing file, directory error
- write_file: happy path, blocked path, creates parent dirs
- list_dir: happy path, blocked path, file-not-dir error
- path normalization (symlink/relative resolved before allowlist check)
- Works on Windows, macOS, Linux (pathlib handles separators)
"""

from __future__ import annotations

from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _roots(tmp_path: Path) -> list[str]:
    """Allowlist containing only tmp_path."""
    return [str(tmp_path)]


# ---------------------------------------------------------------------------
# read_file
# ---------------------------------------------------------------------------


class TestReadFile:
    def test_reads_existing_file(self, tmp_path: Path) -> None:
        from rex.computers.file_ops import read_file

        f = tmp_path / "hello.txt"
        f.write_text("hello world", encoding="utf-8")
        assert read_file(f, allowed_roots=_roots(tmp_path)) == "hello world"

    def test_reads_file_by_string_path(self, tmp_path: Path) -> None:
        from rex.computers.file_ops import read_file

        f = tmp_path / "data.txt"
        f.write_text("content", encoding="utf-8")
        assert read_file(str(f), allowed_roots=_roots(tmp_path)) == "content"

    def test_blocked_path_raises_permission_error(self, tmp_path: Path) -> None:
        from rex.computers.file_ops import read_file

        outside = tmp_path.parent / "outside.txt"
        outside.write_text("secret", encoding="utf-8")
        with pytest.raises(PermissionError, match="Access denied"):
            read_file(outside, allowed_roots=_roots(tmp_path))

    def test_missing_file_raises_file_not_found(self, tmp_path: Path) -> None:
        from rex.computers.file_ops import read_file

        with pytest.raises(FileNotFoundError):
            read_file(tmp_path / "nonexistent.txt", allowed_roots=_roots(tmp_path))

    def test_directory_raises_is_a_directory_error(self, tmp_path: Path) -> None:
        from rex.computers.file_ops import read_file

        subdir = tmp_path / "adir"
        subdir.mkdir()
        with pytest.raises(IsADirectoryError):
            read_file(subdir, allowed_roots=_roots(tmp_path))

    def test_path_normalization_resolves_dots(self, tmp_path: Path) -> None:
        """Paths like /allowed/subdir/../file should still be allowed."""
        from rex.computers.file_ops import read_file

        f = tmp_path / "file.txt"
        f.write_text("ok", encoding="utf-8")
        # Construct path with redundant dot-dot that resolves back inside tmp_path
        tricky = tmp_path / "subdir" / ".." / "file.txt"
        assert read_file(tricky, allowed_roots=_roots(tmp_path)) == "ok"

    def test_blocked_traversal_via_dot_dot(self, tmp_path: Path) -> None:
        """Path that resolves outside the root via ../ must be blocked."""
        from rex.computers.file_ops import read_file

        subdir = tmp_path / "sub"
        subdir.mkdir()
        outside = subdir.parent.parent / "evil.txt"
        outside.write_text("bad", encoding="utf-8")
        with pytest.raises(PermissionError):
            read_file(outside, allowed_roots=[str(subdir)])


# ---------------------------------------------------------------------------
# write_file
# ---------------------------------------------------------------------------


class TestWriteFile:
    def test_writes_new_file(self, tmp_path: Path) -> None:
        from rex.computers.file_ops import write_file

        dest = tmp_path / "output.txt"
        write_file(dest, "written content", allowed_roots=_roots(tmp_path))
        assert dest.read_text(encoding="utf-8") == "written content"

    def test_overwrites_existing_file(self, tmp_path: Path) -> None:
        from rex.computers.file_ops import write_file

        dest = tmp_path / "out.txt"
        dest.write_text("old", encoding="utf-8")
        write_file(dest, "new", allowed_roots=_roots(tmp_path))
        assert dest.read_text(encoding="utf-8") == "new"

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        from rex.computers.file_ops import write_file

        dest = tmp_path / "a" / "b" / "c.txt"
        write_file(dest, "deep", allowed_roots=_roots(tmp_path))
        assert dest.read_text(encoding="utf-8") == "deep"

    def test_blocked_path_raises_permission_error(self, tmp_path: Path) -> None:
        from rex.computers.file_ops import write_file

        outside = tmp_path.parent / "bad.txt"
        with pytest.raises(PermissionError, match="Access denied"):
            write_file(outside, "data", allowed_roots=_roots(tmp_path))


# ---------------------------------------------------------------------------
# list_dir
# ---------------------------------------------------------------------------


class TestListDir:
    def test_lists_entries(self, tmp_path: Path) -> None:
        from rex.computers.file_ops import list_dir

        (tmp_path / "a.txt").write_text("", encoding="utf-8")
        (tmp_path / "b.txt").write_text("", encoding="utf-8")
        (tmp_path / "subdir").mkdir()
        result = list_dir(tmp_path, allowed_roots=_roots(tmp_path))
        assert result == ["a.txt", "b.txt", "subdir"]

    def test_returns_sorted_list(self, tmp_path: Path) -> None:
        from rex.computers.file_ops import list_dir

        (tmp_path / "z.txt").touch()
        (tmp_path / "a.txt").touch()
        (tmp_path / "m.txt").touch()
        assert list_dir(tmp_path, allowed_roots=_roots(tmp_path)) == ["a.txt", "m.txt", "z.txt"]

    def test_empty_directory(self, tmp_path: Path) -> None:
        from rex.computers.file_ops import list_dir

        assert list_dir(tmp_path, allowed_roots=_roots(tmp_path)) == []

    def test_blocked_path_raises_permission_error(self, tmp_path: Path) -> None:
        from rex.computers.file_ops import list_dir

        outside = tmp_path.parent
        with pytest.raises(PermissionError, match="Access denied"):
            list_dir(outside, allowed_roots=_roots(tmp_path))

    def test_file_path_raises_not_a_directory(self, tmp_path: Path) -> None:
        from rex.computers.file_ops import list_dir

        f = tmp_path / "file.txt"
        f.write_text("", encoding="utf-8")
        with pytest.raises(NotADirectoryError):
            list_dir(f, allowed_roots=_roots(tmp_path))


# ---------------------------------------------------------------------------
# Multiple allowed roots
# ---------------------------------------------------------------------------


class TestMultipleRoots:
    def test_second_root_is_allowed(self, tmp_path: Path) -> None:
        from rex.computers.file_ops import read_file

        root1 = tmp_path / "r1"
        root2 = tmp_path / "r2"
        root1.mkdir()
        root2.mkdir()
        f = root2 / "file.txt"
        f.write_text("from root2", encoding="utf-8")
        assert read_file(f, allowed_roots=[str(root1), str(root2)]) == "from root2"

    def test_neither_root_blocks_access(self, tmp_path: Path) -> None:
        from rex.computers.file_ops import read_file

        root1 = tmp_path / "r1"
        root2 = tmp_path / "r2"
        root1.mkdir()
        root2.mkdir()
        outside = tmp_path / "outside.txt"
        outside.write_text("bad", encoding="utf-8")
        with pytest.raises(PermissionError):
            read_file(outside, allowed_roots=[str(root1), str(root2)])
