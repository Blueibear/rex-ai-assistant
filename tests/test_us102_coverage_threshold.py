"""
US-102: Enforce coverage threshold in CI

Verifies:
- pyproject.toml stores the coverage threshold in [tool.coverage.report] fail_under
- CI workflow runs pytest with --cov-fail-under matching the configured threshold
- The threshold is at least 70 (meaningful floor)
- Typecheck passes (this file is mypy-clean)
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).parent.parent
PYPROJECT = ROOT / "pyproject.toml"
CI_YML = ROOT / ".github" / "workflows" / "ci.yml"

# ---------------------------------------------------------------------------
# pyproject.toml checks
# ---------------------------------------------------------------------------


def _read_pyproject() -> str:
    return PYPROJECT.read_text(encoding="utf-8")


def _read_ci() -> str:
    return CI_YML.read_text(encoding="utf-8")


def _extract_fail_under_from_pyproject() -> int:
    """Return the integer value of [tool.coverage.report] fail_under."""
    content = _read_pyproject()
    # Find the [tool.coverage.report] section and extract fail_under
    section_match = re.search(
        r"\[tool\.coverage\.report\](.*?)(?=\n\[|\Z)", content, re.DOTALL
    )
    assert section_match, "[tool.coverage.report] section not found in pyproject.toml"
    section = section_match.group(1)
    value_match = re.search(r"fail_under\s*=\s*(\d+)", section)
    assert value_match, "fail_under not found in [tool.coverage.report]"
    return int(value_match.group(1))


def _extract_cov_fail_under_from_ci() -> int:
    """Return the --cov-fail-under=N value from the CI pytest command."""
    content = _read_ci()
    match = re.search(r"--cov-fail-under=(\d+)", content)
    assert match, "--cov-fail-under=<N> not found in ci.yml"
    return int(match.group(1))


# ---------------------------------------------------------------------------
# Tests: pyproject.toml
# ---------------------------------------------------------------------------


class TestPyprojectCoverageThreshold:
    def test_coverage_report_section_exists(self) -> None:
        content = _read_pyproject()
        assert "[tool.coverage.report]" in content

    def test_fail_under_present(self) -> None:
        threshold = _extract_fail_under_from_pyproject()
        assert isinstance(threshold, int)

    def test_fail_under_at_least_70(self) -> None:
        """Threshold must be meaningful — at least 70%."""
        threshold = _extract_fail_under_from_pyproject()
        assert threshold >= 70, f"Coverage threshold {threshold} is below the 70% floor"

    def test_fail_under_at_most_100(self) -> None:
        threshold = _extract_fail_under_from_pyproject()
        assert threshold <= 100

    def test_coverage_run_section_exists(self) -> None:
        content = _read_pyproject()
        assert "[tool.coverage.run]" in content

    def test_coverage_run_source_includes_rex(self) -> None:
        content = _read_pyproject()
        section_match = re.search(
            r"\[tool\.coverage\.run\](.*?)(?=\n\[|\Z)", content, re.DOTALL
        )
        assert section_match, "[tool.coverage.run] section not found"
        assert 'source = ["rex"]' in section_match.group(1) or "rex" in section_match.group(1)


# ---------------------------------------------------------------------------
# Tests: CI workflow
# ---------------------------------------------------------------------------


class TestCiCoverageThreshold:
    def test_ci_yml_exists(self) -> None:
        assert CI_YML.exists(), "ci.yml not found"

    def test_ci_runs_pytest_with_cov(self) -> None:
        content = _read_ci()
        assert "--cov=rex" in content, "--cov=rex not found in ci.yml"

    def test_ci_has_cov_fail_under(self) -> None:
        content = _read_ci()
        assert "--cov-fail-under=" in content, "--cov-fail-under= not found in ci.yml"

    def test_ci_cov_fail_under_value_matches_pyproject(self) -> None:
        """The --cov-fail-under in CI must match [tool.coverage.report] fail_under."""
        ci_threshold = _extract_cov_fail_under_from_ci()
        pyproject_threshold = _extract_fail_under_from_pyproject()
        assert ci_threshold == pyproject_threshold, (
            f"CI threshold {ci_threshold} != pyproject threshold {pyproject_threshold}"
        )

    def test_ci_cov_fail_under_at_least_70(self) -> None:
        threshold = _extract_cov_fail_under_from_ci()
        assert threshold >= 70, f"CI coverage threshold {threshold} is below the 70% floor"

    def test_ci_tests_job_has_coverage_step(self) -> None:
        content = _read_ci()
        assert "Run tests with coverage" in content or "pytest" in content

    def test_ci_cov_fail_under_on_same_line_as_cov_rex(self) -> None:
        """Both --cov=rex and --cov-fail-under appear in the same pytest invocation."""
        content = _read_ci()
        # Find the pytest line(s)
        pytest_lines = [line for line in content.splitlines() if "pytest" in line and "--cov" in line]
        assert any(
            "--cov=rex" in line and "--cov-fail-under=" in line for line in pytest_lines
        ), f"No single pytest line has both --cov=rex and --cov-fail-under=. Lines: {pytest_lines}"


# ---------------------------------------------------------------------------
# Tests: threshold value
# ---------------------------------------------------------------------------


class TestThresholdDocumented:
    def test_threshold_is_75(self) -> None:
        """The agreed threshold for this project is 75%."""
        threshold = _extract_fail_under_from_pyproject()
        assert threshold == 75, f"Expected threshold 75, got {threshold}"

    def test_ci_threshold_is_75(self) -> None:
        threshold = _extract_cov_fail_under_from_ci()
        assert threshold == 75, f"Expected CI threshold 75, got {threshold}"
