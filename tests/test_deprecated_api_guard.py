from __future__ import annotations

from pathlib import Path


def test_deprecated_api_guard_detects_banned_calls(tmp_path: Path) -> None:
    from scripts.check_deprecated_apis import scan_file

    source = tmp_path / "bad.py"
    source.write_text(
        "import asyncio\nfrom datetime import datetime\n"
        "datetime.utcnow()\nasyncio.get_event_loop()\n",
        encoding="utf-8",
    )
    findings = scan_file(source)
    assert [finding.api for finding in findings] == [
        "datetime.utcnow",
        "asyncio.get_event_loop",
    ]


def test_deprecated_api_guard_ignores_strings_and_comments(tmp_path: Path) -> None:
    from scripts.check_deprecated_apis import scan_file

    source = tmp_path / "good.py"
    source.write_text(
        "# datetime.utcnow() and asyncio.get_event_loop() are banned\n"
        'TEXT = "datetime.utcnow() asyncio.get_event_loop()"\n',
        encoding="utf-8",
    )
    assert scan_file(source) == []


def test_deprecated_api_guard_excludes_archived_paths(tmp_path: Path) -> None:
    from scripts.check_deprecated_apis import scan_paths

    current = tmp_path / "rex" / "current.py"
    archived = tmp_path / "archived" / "old.py"
    current.parent.mkdir(parents=True)
    archived.parent.mkdir(parents=True)
    current.write_text("from datetime import datetime\ndatetime.utcnow()\n", encoding="utf-8")
    archived.write_text("import asyncio\nasyncio.get_event_loop()\n", encoding="utf-8")

    findings = scan_paths([current, archived], root=tmp_path)
    assert len(findings) == 1
    assert findings[0].path == current


def test_deprecated_api_guard_handles_utf8_bom(tmp_path: Path) -> None:
    from scripts.check_deprecated_apis import scan_file

    source = tmp_path / "bom.py"
    source.write_bytes(b"\xef\xbb\xbfimport asyncio\nasyncio.get_event_loop()\n")
    findings = scan_file(source)
    assert [finding.api for finding in findings] == ["asyncio.get_event_loop"]


def test_ci_runs_deprecated_api_guard() -> None:
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert "python scripts/check_deprecated_apis.py" in workflow
