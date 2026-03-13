"""US-057: CI pipeline — validate the GitHub Actions CI configuration."""

from __future__ import annotations

import pathlib

import pytest

CI_WORKFLOW = pathlib.Path(__file__).parent.parent / ".github" / "workflows" / "ci.yml"


@pytest.fixture(scope="module")
def ci_content() -> str:
    return CI_WORKFLOW.read_text(encoding="utf-8")


def test_ci_file_exists() -> None:
    assert CI_WORKFLOW.exists(), "CI workflow file must exist"


def test_ci_runs_on_pull_request(ci_content: str) -> None:
    assert "pull_request" in ci_content, "CI must trigger on pull_request"
    assert (
        "main" in ci_content or "master" in ci_content
    ), "CI must target the active default branch"


def test_ci_lint_job_present(ci_content: str) -> None:
    assert "lint" in ci_content.lower(), "CI must have a lint job"


def test_ci_ruff_executed(ci_content: str) -> None:
    assert "ruff" in ci_content, "CI must run ruff"


def test_ci_black_executed(ci_content: str) -> None:
    assert "black" in ci_content, "CI must run black"


def test_ci_typecheck_job_present(ci_content: str) -> None:
    assert (
        "typecheck" in ci_content.lower() or "mypy" in ci_content
    ), "CI must have a typecheck or mypy step"


def test_ci_mypy_executed(ci_content: str) -> None:
    assert "mypy" in ci_content, "CI must run mypy"


def test_ci_tests_job_present(ci_content: str) -> None:
    assert "pytest" in ci_content, "CI must run pytest"


def test_ci_coverage_reported(ci_content: str) -> None:
    assert "--cov" in ci_content, "CI must run tests with coverage"
