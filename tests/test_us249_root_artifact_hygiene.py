from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GITIGNORE = ROOT / ".gitignore"
TRACKED_ARTIFACTS = (
    ".coverage",
    "coverage.txt",
    "test-audit-coverage.txt",
    "test-audit-final-results.txt",
)
IGNORED_ARTIFACTS = (
    ".coverage",
    "coverage.txt",
    "coverage.html",
    "coverage/example.txt",
    "test-audit-coverage.txt",
    "test-audit-final-results.txt",
    "example.patch",
)


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def test_gitignore_contains_us249_artifact_rules() -> None:
    text = GITIGNORE.read_text(encoding="utf-8")

    for pattern in (
        ".coverage",
        "coverage.txt",
        "coverage.html",
        "coverage/",
        "test-audit-*.txt",
        "*.patch",
    ):
        assert pattern in text


def test_us249_artifacts_are_ignored_by_git() -> None:
    for artifact in IGNORED_ARTIFACTS:
        result = _git("check-ignore", "-v", artifact)
        assert ".gitignore" in result.stdout, f"{artifact} should be ignored by .gitignore"


def test_us249_artifacts_are_not_tracked() -> None:
    result = _git("ls-files", "--", *TRACKED_ARTIFACTS)
    tracked = [line for line in result.stdout.splitlines() if line]
    assert tracked == [], f"Expected no tracked US-249 artifacts, found: {tracked}"
