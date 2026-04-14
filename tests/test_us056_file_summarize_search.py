"""Tests for US-056: File summarization and search.

Covers:
- summarize_file: reads file content, passes to LLM prompt, returns summary;
  LLM-unavailable fallback returns preview
- search_files: finds matching lines across text files, respects allowlist,
  returns {file, line_number, line} dicts
- Both functions respect the directory allowlist (blocked path raises PermissionError)
- Both exported from rex.computers.file_ops
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def _roots(tmp_path: Path) -> list[str]:
    return [str(tmp_path)]


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------


def test_summarize_file_is_exported() -> None:
    from rex.computers import file_ops

    assert hasattr(file_ops, "summarize_file")


def test_search_files_is_exported() -> None:
    from rex.computers import file_ops

    assert hasattr(file_ops, "search_files")


# ---------------------------------------------------------------------------
# summarize_file
# ---------------------------------------------------------------------------


class TestSummarizeFile:
    def test_calls_llm_with_file_content(self, tmp_path: Path) -> None:
        from rex.computers.file_ops import summarize_file

        f = tmp_path / "doc.txt"
        f.write_text("This is a long document about Python.", encoding="utf-8")

        mock_lm = MagicMock()
        mock_lm.generate.return_value = "A document about Python."

        with (
            patch("rex.config.load_config"),
            patch("rex.llm_client.LanguageModel", return_value=mock_lm),
        ):
            result = summarize_file(f, allowed_roots=_roots(tmp_path))

        assert result == "A document about Python."
        # Verify the file content was passed to the LLM
        prompt_arg = mock_lm.generate.call_args[0][0]
        assert "This is a long document about Python." in prompt_arg

    def test_llm_unavailable_returns_preview(self, tmp_path: Path) -> None:
        from rex.computers.file_ops import summarize_file

        f = tmp_path / "doc.txt"
        content = "Hello " * 100  # long enough
        f.write_text(content, encoding="utf-8")

        with patch("rex.config.load_config", side_effect=ImportError("no llm")):
            result = summarize_file(f, allowed_roots=_roots(tmp_path))

        assert "[LLM unavailable]" in result
        assert "Hello" in result

    def test_blocked_path_raises_permission_error(self, tmp_path: Path) -> None:
        from rex.computers.file_ops import summarize_file

        outside = tmp_path.parent / "evil.txt"
        outside.write_text("bad", encoding="utf-8")

        with pytest.raises(PermissionError):
            summarize_file(outside, allowed_roots=_roots(tmp_path))

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        from rex.computers.file_ops import summarize_file

        with pytest.raises(FileNotFoundError):
            summarize_file(tmp_path / "missing.txt", allowed_roots=_roots(tmp_path))

    def test_truncates_long_content_at_max_chars(self, tmp_path: Path) -> None:
        """LLM prompt must not receive more than max_chars chars."""
        from rex.computers.file_ops import summarize_file

        long_content = "x" * 20000
        f = tmp_path / "big.txt"
        f.write_text(long_content, encoding="utf-8")

        captured: list[str] = []

        def fake_generate(prompt: str) -> str:
            captured.append(prompt)
            return "summary"

        mock_lm = MagicMock()
        mock_lm.generate.side_effect = fake_generate

        with (
            patch("rex.config.load_config"),
            patch("rex.llm_client.LanguageModel", return_value=mock_lm),
        ):
            summarize_file(f, allowed_roots=_roots(tmp_path), max_chars=8000)

        assert len(captured) == 1
        # The prompt includes the prefix text + at most 8000 chars of content
        assert "x" * 8001 not in captured[0]


# ---------------------------------------------------------------------------
# search_files
# ---------------------------------------------------------------------------


class TestSearchFiles:
    def _populate(self, tmp_path: Path) -> None:
        (tmp_path / "a.txt").write_text("Hello world\nFoo bar\n", encoding="utf-8")
        (tmp_path / "b.txt").write_text("Hello again\nBaz qux\n", encoding="utf-8")
        (tmp_path / "notes.md").write_text("Hello markdown\n", encoding="utf-8")

    def test_finds_matching_lines(self, tmp_path: Path) -> None:
        from rex.computers.file_ops import search_files

        self._populate(tmp_path)
        results = search_files(tmp_path, "Hello", allowed_roots=_roots(tmp_path))
        assert len(results) == 2  # a.txt and b.txt (md excluded by default pattern)
        assert all("Hello" in r["line"] for r in results)

    def test_returns_file_line_number_line(self, tmp_path: Path) -> None:
        from rex.computers.file_ops import search_files

        (tmp_path / "x.txt").write_text("line one\nfind me\nline three\n", encoding="utf-8")
        results = search_files(tmp_path, "find me", allowed_roots=_roots(tmp_path))
        assert len(results) == 1
        assert results[0]["line_number"] == 2
        assert results[0]["line"] == "find me"
        assert "x.txt" in results[0]["file"]

    def test_case_insensitive_match(self, tmp_path: Path) -> None:
        from rex.computers.file_ops import search_files

        (tmp_path / "c.txt").write_text("HELLO WORLD\n", encoding="utf-8")
        results = search_files(tmp_path, "hello", allowed_roots=_roots(tmp_path))
        assert len(results) == 1

    def test_no_match_returns_empty_list(self, tmp_path: Path) -> None:
        from rex.computers.file_ops import search_files

        self._populate(tmp_path)
        results = search_files(tmp_path, "zzznomatch", allowed_roots=_roots(tmp_path))
        assert results == []

    def test_only_searches_txt_by_default(self, tmp_path: Path) -> None:
        from rex.computers.file_ops import search_files

        self._populate(tmp_path)
        # notes.md contains "Hello" but pattern="*.txt" by default
        results = search_files(tmp_path, "Hello", allowed_roots=_roots(tmp_path))
        assert all(r["file"].endswith(".txt") for r in results)

    def test_custom_pattern_matches_md(self, tmp_path: Path) -> None:
        from rex.computers.file_ops import search_files

        self._populate(tmp_path)
        results = search_files(tmp_path, "Hello", allowed_roots=_roots(tmp_path), pattern="*.md")
        assert len(results) == 1
        assert "notes.md" in results[0]["file"]

    def test_blocked_directory_raises_permission_error(self, tmp_path: Path) -> None:
        from rex.computers.file_ops import search_files

        outside = tmp_path.parent
        with pytest.raises(PermissionError):
            search_files(outside, "anything", allowed_roots=_roots(tmp_path))

    def test_not_a_directory_raises(self, tmp_path: Path) -> None:
        from rex.computers.file_ops import search_files

        f = tmp_path / "file.txt"
        f.write_text("text", encoding="utf-8")
        with pytest.raises(NotADirectoryError):
            search_files(f, "text", allowed_roots=_roots(tmp_path))
