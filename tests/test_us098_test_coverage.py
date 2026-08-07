"""Coverage-contract regression tests.

US-098 originally required a generated ``coverage.txt`` file in the repository
root. That file is now intentionally ignored and is created only by CI's
``tee`` pipeline so the skip-budget checker can parse the pytest summary.
Coverage correctness is enforced by pytest-cov and the configured threshold,
not by the presence of a leftover generated text file.
"""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"
CI_YML = ROOT / ".github" / "workflows" / "ci.yml"
GITIGNORE = ROOT / ".gitignore"
GAP_REPORT = ROOT / "test-audit-coverage-gaps.json"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _fail_under() -> int:
    content = _read(PYPROJECT)
    section = re.search(r"\[tool\.coverage\.report\](.*?)(?=\n\[|\Z)", content, re.DOTALL)
    assert section, "pyproject.toml missing [tool.coverage.report]"
    match = re.search(r"fail_under\s*=\s*(\d+)", section.group(1))
    assert match, "[tool.coverage.report] missing fail_under"
    return int(match.group(1))


def test_pytest_cov_installed() -> None:
    assert importlib.util.find_spec("pytest_cov") is not None


def test_coverage_module_importable() -> None:
    assert importlib.util.find_spec("coverage") is not None


def test_ci_generates_human_and_machine_readable_coverage_reports() -> None:
    ci = _read(CI_YML)
    assert "--cov=rex" in ci
    assert "--cov-report=term-missing" in ci
    assert "--cov-report=html" in ci
    assert "--cov-report=xml" in ci


def test_ci_text_capture_is_ephemeral_not_a_required_repo_artifact() -> None:
    ci = _read(CI_YML)
    assert "tee coverage.txt" in ci
    assert "coverage.txt" in _read(GITIGNORE)


def test_committed_gap_inventory_is_parseable() -> None:
    data = json.loads(_read(GAP_REPORT))
    assert isinstance(data, list) and data
    assert all("module_path" in row and "current_coverage_pct" in row for row in data)


def test_committed_gap_inventory_identifies_low_coverage_modules() -> None:
    data = json.loads(_read(GAP_REPORT))
    below_50 = [row for row in data if int(row["current_coverage_pct"]) < 50]
    assert below_50, "coverage-gap inventory must explicitly identify below-50% modules"


def test_pyproject_has_coverage_source_and_threshold() -> None:
    content = _read(PYPROJECT)
    assert "[tool.coverage.run]" in content
    assert 'source = ["rex"]' in content
    assert "[tool.coverage.report]" in content
    assert 50 <= _fail_under() <= 100


def test_ci_threshold_matches_pyproject() -> None:
    ci = _read(CI_YML)
    match = re.search(r"--cov-fail-under=(\d+)", ci)
    assert match, "CI pytest invocation missing --cov-fail-under"
    assert int(match.group(1)) == _fail_under()


def test_agreed_coverage_threshold_is_75_percent() -> None:
    assert _fail_under() == 75
