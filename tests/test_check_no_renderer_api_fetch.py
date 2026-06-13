"""Tests for scripts/check_no_renderer_api_fetch.py (US-003)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
import check_no_renderer_api_fetch as script  # noqa: E402


def _gui_file(tmp_path: Path, rel: str, content: str) -> Path:
    fpath = tmp_path / rel
    fpath.parent.mkdir(parents=True, exist_ok=True)
    fpath.write_text(content, encoding="utf-8")
    return fpath


def _allowlist(tmp_path: Path, text: str) -> None:
    path = tmp_path / "gui" / "src" / "ALLOWED_API_FETCHES.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# find_violations()
# ---------------------------------------------------------------------------


def test_no_api_fetches_returns_empty(tmp_path: Path) -> None:
    _gui_file(
        tmp_path, "gui/src/pages/Clean.tsx", "export default function Clean() { return null; }\n"
    )
    assert script.find_violations(tmp_path) == []


def test_missing_gui_src_returns_empty(tmp_path: Path) -> None:
    assert script.find_violations(tmp_path) == []


def test_single_quote_fetch_detected(tmp_path: Path) -> None:
    _gui_file(tmp_path, "gui/src/pages/Bad.tsx", "const r = await fetch('/api/status');\n")
    violations = script.find_violations(tmp_path)
    assert len(violations) == 1
    assert violations[0][0] == "gui/src/pages/Bad.tsx"
    assert violations[0][1] == 1


def test_double_quote_fetch_detected(tmp_path: Path) -> None:
    _gui_file(tmp_path, "gui/src/pages/Bad.tsx", 'const r = await fetch("/api/status");\n')
    assert len(script.find_violations(tmp_path)) == 1


def test_template_literal_fetch_detected(tmp_path: Path) -> None:
    _gui_file(
        tmp_path,
        "gui/src/pages/Bad.tsx",
        "const r = await fetch(`/api/devices/${id}/command`);\n",
    )
    assert len(script.find_violations(tmp_path)) == 1


def test_non_api_fetch_not_flagged(tmp_path: Path) -> None:
    _gui_file(
        tmp_path, "gui/src/pages/Good.tsx", "const r = await fetch('https://example.com/data');\n"
    )
    assert script.find_violations(tmp_path) == []


def test_allowlisted_entry_is_permitted(tmp_path: Path) -> None:
    _gui_file(tmp_path, "gui/src/pages/AboutPage.tsx", "const r = await fetch('/api/status');\n")
    _allowlist(tmp_path, "gui/src/pages/AboutPage.tsx:1  # pending US-004\n")
    assert script.find_violations(tmp_path) == []


def test_allowlist_wrong_line_still_flags(tmp_path: Path) -> None:
    _gui_file(
        tmp_path,
        "gui/src/pages/AboutPage.tsx",
        "// comment\nconst r = await fetch('/api/status');\n",
    )
    _allowlist(tmp_path, "gui/src/pages/AboutPage.tsx:1  # wrong line\n")
    violations = script.find_violations(tmp_path)
    assert len(violations) == 1
    assert violations[0][1] == 2


def test_allowlist_comment_stripping(tmp_path: Path) -> None:
    _gui_file(tmp_path, "gui/src/pages/AboutPage.tsx", "const r = await fetch('/api/status');\n")
    _allowlist(tmp_path, "  gui/src/pages/AboutPage.tsx:1   # inline comment stripped\n")
    assert script.find_violations(tmp_path) == []


def test_multiple_violations_reported(tmp_path: Path) -> None:
    _gui_file(
        tmp_path,
        "gui/src/pages/Multi.tsx",
        "fetch('/api/a');\nfetch('/api/b');\n",
    )
    violations = script.find_violations(tmp_path)
    assert len(violations) == 2


def test_js_file_also_scanned(tmp_path: Path) -> None:
    _gui_file(tmp_path, "gui/src/utils/helper.js", "fetch('/api/data');\n")
    assert len(script.find_violations(tmp_path)) == 1


def test_blank_allowlist_file_flags_matches(tmp_path: Path) -> None:
    _gui_file(tmp_path, "gui/src/pages/Bad.tsx", "fetch('/api/status');\n")
    _allowlist(tmp_path, "# just comments\n\n# another comment\n")
    assert len(script.find_violations(tmp_path)) == 1


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------


def test_main_exits_0_clean(tmp_path: Path) -> None:
    _gui_file(tmp_path, "gui/src/pages/Clean.tsx", "export default null;\n")
    assert script.main(["--repo-root", str(tmp_path)]) == 0


def test_main_exits_1_violation(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    _gui_file(tmp_path, "gui/src/pages/Bad.tsx", "fetch('/api/status');\n")
    result = script.main(["--repo-root", str(tmp_path)])
    assert result == 1
    out = capsys.readouterr().out
    assert "Bad.tsx" in out
    assert "ERROR" in out


def test_main_exits_0_with_allowlist(tmp_path: Path) -> None:
    _gui_file(tmp_path, "gui/src/pages/AboutPage.tsx", "fetch('/api/status');\n")
    _allowlist(tmp_path, "gui/src/pages/AboutPage.tsx:1  # pending US-004\n")
    assert script.main(["--repo-root", str(tmp_path)]) == 0
