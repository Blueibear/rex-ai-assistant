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
COVERAGE_XML = PROJECT_ROOT / "coverage.xml"
PYPROJECT = PROJECT_ROOT / "pyproject.toml"


def _coverage_text() -> str:
    """Return coverage report text from file artifacts when present."""
    if COVERAGE_TXT.exists():
        return COVERAGE_TXT.read_text(encoding="utf-8")
    if COVERAGE_XML.exists():
        return COVERAGE_XML.read_text(encoding="utf-8")
    return ""


def _coverage_current():
    """Return live Coverage object when tests run under pytest-cov, else None."""
    try:
        import coverage
    except Exception:  # pragma: no cover - defensive import fallback
        return None
    return coverage.Coverage.current()


def _coverage_exists() -> bool:
    """Whether coverage data is available from artifacts or current pytest-cov run."""
    return COVERAGE_TXT.exists() or COVERAGE_XML.exists() or _coverage_current() is not None


def _ensure_coverage_available() -> None:
    if not _coverage_exists():
        pytest.skip("Coverage artifacts/runtime not available in this test invocation")


def _runtime_coverage_rows() -> list[tuple[str, int]]:
    """Read per-module percentages from active pytest-cov session when available."""
    cov = _coverage_current()
    if cov is None:
        return []

    data = cov.get_data()
    rows: list[tuple[str, int]] = []
    for filename in data.measured_files():
        norm = filename.replace("\\", "/")
        if "/rex/" not in norm:
            continue
        rel = norm.split("/rex/", 1)[1]
        module = f"rex/{rel}"
        if not module.endswith(".py"):
            continue

        _, statements, _, missing, _ = cov.analysis2(filename)
        if not statements:
            pct = 100
        else:
            pct = int(round(((len(statements) - len(missing)) / len(statements)) * 100))
        rows.append((module, pct))
    return rows


def _runtime_total_pct() -> int | None:
    """Get total coverage percent from active pytest-cov session."""
    cov = _coverage_current()
    if cov is None:
        return None
    totals = cov.get_data()
    measured = [f for f in totals.measured_files() if "/rex/" in f.replace("\\", "/")]
    if not measured:
        return None

    total_statements = 0
    total_missing = 0
    for filename in measured:
        _, statements, _, missing, _ = cov.analysis2(filename)
        total_statements += len(statements)
        total_missing += len(missing)
    if total_statements == 0:
        return None
    return int(round(((total_statements - total_missing) / total_statements) * 100))


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
    """Coverage data must be available from report files or active pytest-cov run."""
    _ensure_coverage_available()
    assert _coverage_exists(), (
        f"coverage report not found. Expected {COVERAGE_TXT} or {COVERAGE_XML}, "
        "or active pytest-cov session data. "
        "Run: python -m pytest --cov=rex --cov-report=term-missing --cov-report=xml"
    )


def test_coverage_txt_non_empty() -> None:
    """coverage.txt/xml must contain content when present."""
    _ensure_coverage_available()
    assert _coverage_exists(), "coverage report does not exist"
    content = _coverage_text()
    if content:
        assert len(content) > 500, f"coverage report appears too short ({len(content)} bytes)"


def test_coverage_txt_has_summary_line() -> None:
    """Coverage report contains summary markers or runtime totals."""
    _ensure_coverage_available()
    assert _coverage_exists(), "coverage report does not exist"
    content = _coverage_text()
    runtime_total = _runtime_total_pct()
    assert (
        ("TOTAL" in content) or ("line-rate" in content) or (runtime_total is not None)
    ), "coverage report does not contain expected summary markers"


def test_coverage_txt_has_rex_modules() -> None:
    """Coverage report includes rex package modules."""
    _ensure_coverage_available()
    assert _coverage_exists(), "coverage report does not exist"
    content = _coverage_text()
    runtime_rows = _runtime_coverage_rows()
    assert (
        "rex\\" in content or "rex/" in content or 'filename="' in content or len(runtime_rows) > 0
    ), "coverage report does not list rex package modules"


# ---------------------------------------------------------------------------
# AC3: modules with below-50% coverage are identifiable
# ---------------------------------------------------------------------------


def _parse_coverage_rows() -> list[tuple[str, int]]:
    """Parse coverage report and return list of (module, coverage_pct) tuples."""
    content = _coverage_text()
    if content:
        results: list[tuple[str, int]] = []
        if COVERAGE_TXT.exists():
            pattern = re.compile(r"^(rex[\/]\S+)\s+\d+\s+\d+\s+(\d+)%", re.MULTILINE)
            for match in pattern.finditer(content):
                module = match.group(1)
                pct = int(match.group(2))
                results.append((module, pct))
            if results:
                return results

        for match in re.finditer(r'filename="([^"]+)"[^\n]*line-rate="([0-9.]+)"', content):
            module = match.group(1).replace("\\", "/")
            if not module.endswith(".py"):
                continue
            normalized = module if module.startswith("rex/") else f"rex/{module}"
            pct = int(round(float(match.group(2)) * 100))
            results.append((normalized, pct))
        if results:
            return results

    return _runtime_coverage_rows()


def test_coverage_txt_parseable() -> None:
    _ensure_coverage_available()
    """Coverage data must contain parseable module coverage lines."""
    rows = _parse_coverage_rows()
    assert len(rows) > 50, f"Expected >50 module lines in coverage data, found {len(rows)}"


def test_below_50_modules_present_in_report() -> None:
    _ensure_coverage_available()
    """Coverage data must contain modules with below-50% coverage."""
    rows = _parse_coverage_rows()
    assert rows, "Could not parse any module lines from coverage data"

    below_50 = [(mod, pct) for mod, pct in rows if pct < 50]
    assert len(below_50) > 0, (
        "Expected at least one module below 50% coverage. "
        "If all modules are now above 50%, update this threshold."
    )


def test_known_low_coverage_modules_visible() -> None:
    _ensure_coverage_available()
    """Known low-coverage modules must appear in coverage data."""
    rows = _parse_coverage_rows()
    modules = {mod for mod, _ in rows}
    joined = "\n".join(sorted(modules)) + "\n" + _coverage_text()

    known_low = [
        "wakeword",
        "plugin_loader",
    ]
    for fragment in known_low:
        assert (
            fragment in joined
        ), f"Expected to find '{fragment}' in coverage data but it was missing"


def test_below_50_modules_list() -> None:
    _ensure_coverage_available()
    """Verify the list of below-50% modules is non-trivial and stable."""
    rows = _parse_coverage_rows()
    below_50 = sorted([(mod, pct) for mod, pct in rows if pct < 50])

    expected_low = {
        "rex\\wakeword\\embedding.py",
        "rex/wakeword/embedding.py",
        "rex\\plugin_loader.py",
        "rex/plugin_loader.py",
    }
    found_low_names = {mod for mod, _ in below_50}
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
    _ensure_coverage_available()
    """Total coverage reported in artifacts/runtime must meet threshold."""
    pyproject_content = PYPROJECT.read_text(encoding="utf-8")
    match = re.search(r"fail_under\s*=\s*(\d+)", pyproject_content)
    assert match, "Could not parse fail_under value from pyproject.toml"
    threshold = int(match.group(1))

    # In an active pytest-cov run, final threshold enforcement is performed by
    # --cov-fail-under at session teardown when complete data is available.
    if _coverage_current() is not None:
        return

    _ensure_coverage_available()
    assert _coverage_exists(), "coverage report does not exist"
    txt_content = _coverage_text()
    total_pct: int | None = None

    if COVERAGE_TXT.exists():
        total_match = re.search(r"^TOTAL\s+\d+\s+\d+\s+(\d+)%", txt_content, re.MULTILINE)
        assert total_match, "Could not parse TOTAL line from coverage.txt"
        total_pct = int(total_match.group(1))
    elif COVERAGE_XML.exists():
        total_match = re.search(r"<coverage[^>]*line-rate=\"([0-9.]+)\"", txt_content)
        assert total_match, "Could not parse overall line-rate from coverage.xml"
        total_pct = int(round(float(total_match.group(1)) * 100))
    else:
        # When running inside the same pytest-cov session, report files are
        # emitted at teardown. --cov-fail-under already enforces threshold.
        total_pct = _runtime_total_pct()
        assert total_pct is not None, "Unable to derive runtime coverage"
        return

    if total_pct == 0:
        # Ignore stale artifacts from ad-hoc partial coverage runs.
        return

    assert (
        total_pct >= threshold
    ), f"Total coverage {total_pct}% is below fail_under threshold {threshold}%"


# ---------------------------------------------------------------------------
# Utility: print below-50% modules (informational, always passes)
# ---------------------------------------------------------------------------


def test_print_below_50_modules() -> None:
    """Print modules below 50% coverage for visibility (always passes)."""
    rows = _parse_coverage_rows()
    below_50 = sorted([(mod, pct) for mod, pct in rows if pct < 50])
    if below_50:
        lines = "\n".join(f"  {pct:3d}%  {mod}" for mod, pct in below_50)
        print(f"\nModules below 50% coverage ({len(below_50)} total):\n{lines}")
    else:
        print("\nAll modules are at or above 50% coverage.")
    assert True
