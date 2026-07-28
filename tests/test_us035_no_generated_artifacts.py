"""US-035: CI gate rejecting committed generated artifacts."""

from __future__ import annotations

from scripts import check_no_generated_artifacts as checker


def test_flags_each_generated_pattern() -> None:
    offenders = checker.classify_paths(
        [
            ".coverage",
            "coverage.xml",
            "coverage.txt",
            "htmlcov/index.html",
            "dist/askrex_assistant-1.0-py3-none-any.whl",
            "build/lib/rex/cli.py",
            "rex/__pycache__/cli.cpython-311.pyc",
            "askrex_assistant.egg-info/PKG-INFO",
            "sub/dir/.coverage.hostname",
        ]
    )
    flagged = {path for path, _ in offenders}
    assert flagged == {
        ".coverage",
        "coverage.xml",
        "coverage.txt",
        "htmlcov/index.html",
        "dist/askrex_assistant-1.0-py3-none-any.whl",
        "build/lib/rex/cli.py",
        "rex/__pycache__/cli.cpython-311.pyc",
        "askrex_assistant.egg-info/PKG-INFO",
        "sub/dir/.coverage.hostname",
    }


def test_clean_paths_pass() -> None:
    assert (
        checker.classify_paths(
            [
                "rex/cli.py",
                "docs/distribution.md",
                "tests/conftest.py",
                "gui/src/main/index.ts",
                "scripts/build_managed_python_runtime.ps1",
                # Names that contain but do not match generated patterns.
                "docs/coverage_policy.md",
                "rex/distribute.py",
            ]
        )
        == []
    )


def test_allowlisted_dashboard_bundle_is_permitted() -> None:
    assert "rex/ui/dist/index.html" in checker.ALLOWLIST
    assert checker.classify_paths(["rex/ui/dist/index.html"]) == []


def test_allowlist_entries_all_have_justifications() -> None:
    for path, reason in checker.ALLOWLIST.items():
        assert reason.strip(), f"ALLOWLIST entry {path} must state why it is committed"


def test_current_tree_has_no_tracked_generated_artifacts() -> None:
    assert checker.find_tracked_generated_artifacts() == []
