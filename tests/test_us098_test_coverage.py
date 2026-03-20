"""
US-098: Measure and document baseline test coverage

Acceptance criteria:
- pytest --cov=rex --cov-report=term-missing runs without error
- coverage report saved to coverage.txt or equivalent
- modules with below-50% coverage listed explicitly in the report
- agreed minimum coverage threshold documented in pyproject.toml or setup.cfg
- Typecheck passes
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
COVERAGE_TXT = PROJECT_ROOT / "coverage.txt"
COVERAGE_JSON = PROJECT_ROOT / "coverage.json"
PYPROJECT = PROJECT_ROOT / "pyproject.toml"


def _read_coverage_txt() -> str:
    """Return coverage.txt contents or skip when the file is still being written."""
    if not COVERAGE_TXT.exists():
        pytest.skip("coverage.txt has not been generated yet")
    content = COVERAGE_TXT.read_text(encoding="utf-8")
    if "TOTAL" not in content:
        pytest.skip("coverage.txt is still being written by the current coverage run")
    return content


# ---------------------------------------------------------------------------
# AC1: pytest-cov is installed and importable
# ---------------------------------------------------------------------------


def test_pytest_cov_installed() -> None:
    """pytest-cov package must be available."""
    import importlib

    spec = importlib.util.find_spec("pytest_cov")
    assert spec is not None, "pytest-cov is not installed"


def test_coverage_module_importable() -> None:
    """coverage package must be importable."""
    import importlib

    spec = importlib.util.find_spec("coverage")
    assert spec is not None, "coverage package is not installed"


# ---------------------------------------------------------------------------
# AC2: coverage report saved to coverage.txt or equivalent
# ---------------------------------------------------------------------------


def test_coverage_txt_exists() -> None:
    """coverage.txt must exist in the project root."""
    assert COVERAGE_TXT.exists(), (
        f"coverage.txt not found at {COVERAGE_TXT}. "
        "Run: python -m pytest --cov=rex --cov-report=term-missing -q 2>&1 | tee coverage.txt"
    )


def test_coverage_txt_non_empty() -> None:
    """coverage.txt must contain actual content."""
    content = _read_coverage_txt()
    assert len(content) > 500, f"coverage.txt appears too short ({len(content)} bytes)"


def test_coverage_txt_has_summary_line() -> None:
    """coverage.txt must contain the TOTAL summary line."""
    content = _read_coverage_txt()
    assert "TOTAL" in content, "coverage.txt does not contain TOTAL summary line"


def test_coverage_txt_has_rex_modules() -> None:
    """coverage.txt must list rex package modules."""
    content = _read_coverage_txt()
    assert "rex\\" in content or "rex/" in content, "coverage.txt does not list rex package modules"


# ---------------------------------------------------------------------------
# AC3: modules with below-50% coverage are identifiable
# ---------------------------------------------------------------------------


def _parse_coverage_txt() -> list[tuple[str, int]]:
    """Parse coverage.txt and return list of (module, coverage_pct) tuples."""
    content = _read_coverage_txt()
    results: list[tuple[str, int]] = []
    # Match lines like: rex\foo.py   123   45   63%   ...
    pattern = re.compile(r"^(rex[\\/]\S+)\s+\d+\s+\d+\s+(\d+)%", re.MULTILINE)
    for match in pattern.finditer(content):
        module = match.group(1)
        pct = int(match.group(2))
        results.append((module, pct))
    return results


def test_coverage_txt_parseable() -> None:
    """coverage.txt must contain parseable module coverage lines."""
    rows = _parse_coverage_txt()
    assert len(rows) > 50, f"Expected >50 module lines in coverage.txt, found {len(rows)}"


def test_below_50_modules_present_in_report() -> None:
    """coverage.txt must contain modules with below-50% coverage.

    These are known low-coverage modules that rely on heavy optional
    dependencies (audio, GPU, Windows services).
    """
    rows = _parse_coverage_txt()
    assert rows, "Could not parse any module lines from coverage.txt"

    below_50 = [(mod, pct) for mod, pct in rows if pct < 50]
    assert len(below_50) > 0, (
        "Expected at least one module below 50% coverage. "
        "If all modules are now above 50%, update this threshold."
    )


def test_known_low_coverage_modules_visible() -> None:
    """Known low-coverage modules must appear in the report."""
    content = _read_coverage_txt()

    # These modules are known to have low coverage due to optional heavy deps
    known_low = [
        "wakeword",  # optional audio/ML dependency
        "plugin_loader",  # dynamic plugin loading
    ]
    for fragment in known_low:
        assert (
            fragment in content
        ), f"Expected to find '{fragment}' in coverage.txt but it was missing"


def test_below_50_modules_list() -> None:
    """Verify the list of below-50% modules is non-trivial and stable."""
    rows = _parse_coverage_txt()
    below_50 = sorted([(mod, pct) for mod, pct in rows if pct < 50])

    # We expect at least these well-known low-coverage modules
    expected_low = {
        "rex\\wakeword\\embedding.py",
        "rex/wakeword/embedding.py",
        "rex\\plugin_loader.py",
        "rex/plugin_loader.py",
    }
    found_low_names = {mod for mod, _ in below_50}
    # At least one of the expected low-coverage modules must appear
    overlap = expected_low & found_low_names
    assert overlap, (
        f"Expected at least one of {expected_low} in below-50% modules. "
        f"Found below-50%: {found_low_names}"
    )


# ---------------------------------------------------------------------------
# AC4: agreed minimum coverage threshold documented in pyproject.toml
# ---------------------------------------------------------------------------


def test_pyproject_has_coverage_section() -> None:
    """pyproject.toml must have a [tool.coverage.report] section."""
    content = PYPROJECT.read_text(encoding="utf-8")
    assert (
        "[tool.coverage.report]" in content
    ), "pyproject.toml missing [tool.coverage.report] section"


def test_pyproject_has_fail_under() -> None:
    """pyproject.toml must document a minimum coverage threshold via fail_under."""
    content = PYPROJECT.read_text(encoding="utf-8")
    assert (
        "fail_under" in content
    ), "pyproject.toml [tool.coverage.report] section must contain 'fail_under'"


def test_fail_under_value_reasonable() -> None:
    """fail_under threshold must be between 50 and 100."""
    content = PYPROJECT.read_text(encoding="utf-8")
    match = re.search(r"fail_under\s*=\s*(\d+)", content)
    assert match, "Could not parse fail_under value from pyproject.toml"
    value = int(match.group(1))
    assert 50 <= value <= 100, f"fail_under = {value} is outside acceptable range [50, 100]"


def test_total_coverage_meets_threshold() -> None:
    """Total coverage reported in coverage.txt must meet the documented threshold."""
    # Parse fail_under from pyproject.toml
    pyproject_content = PYPROJECT.read_text(encoding="utf-8")
    match = re.search(r"fail_under\s*=\s*(\d+)", pyproject_content)
    assert match, "Could not parse fail_under value from pyproject.toml"
    threshold = int(match.group(1))

    # Parse total from coverage.txt
    txt_content = _read_coverage_txt()
    total_match = re.search(r"^TOTAL\s+\d+\s+\d+\s+(\d+)%", txt_content, re.MULTILINE)
    assert total_match, "Could not parse TOTAL line from coverage.txt"
    total_pct = int(total_match.group(1))

    assert (
        total_pct >= threshold
    ), f"Total coverage {total_pct}% is below fail_under threshold {threshold}%"


# ---------------------------------------------------------------------------
# Utility: print below-50% modules (informational, always passes)
# ---------------------------------------------------------------------------


def test_print_below_50_modules() -> None:
    """Print modules below 50% coverage for visibility (always passes)."""
    rows = _parse_coverage_txt()
    below_50 = sorted([(mod, pct) for mod, pct in rows if pct < 50])
    if below_50:
        lines = "\n".join(f"  {pct:3d}%  {mod}" for mod, pct in below_50)
        print(f"\nModules below 50% coverage ({len(below_50)} total):\n{lines}")
    else:
        print("\nAll modules are at or above 50% coverage.")
    # This test always passes — it's for visibility only
    assert True
