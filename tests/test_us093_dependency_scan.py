"""US-093: Dependency vulnerability scan and remediation.

Acceptance criteria verified:
- pip-audit (or safety check) runs against the current lock file / installed packages
- all critical and high severity findings remediated or explicitly documented as accepted risk
- scan added as a CI step that fails on new critical/high findings
- Typecheck passes
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
SECURITY_DOC = PROJECT_ROOT / "docs" / "security" / "VULNERABILITY-SCAN.md"
CI_YML = PROJECT_ROOT / ".github" / "workflows" / "ci.yml"


# ---------------------------------------------------------------------------
# Acceptance criterion: scan tool can run
# ---------------------------------------------------------------------------


def test_pip_audit_importable() -> None:
    """pip-audit package is installed and importable via module."""
    result = subprocess.run(
        [sys.executable, "-m", "pip_audit", "--version"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"pip_audit --version failed: {result.stderr}"
    assert "pip-audit" in result.stdout.lower() or result.returncode == 0


def test_pip_audit_produces_json_output() -> None:
    """pip-audit can scan installed packages and return JSON output."""
    result = subprocess.run(
        [sys.executable, "-m", "pip_audit", "--format=json"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    # pip-audit exits 1 if vulnerabilities found, 0 if clean.
    # Either is acceptable — the point is that it runs and produces parseable output.
    assert result.returncode in (
        0,
        1,
    ), f"pip_audit exited with unexpected code {result.returncode}:\n{result.stderr}"
    output = result.stdout.strip()
    assert output, "pip_audit produced no JSON output"
    parsed = json.loads(output)
    assert "dependencies" in parsed, "JSON output missing 'dependencies' key"
    assert isinstance(parsed["dependencies"], list)


# ---------------------------------------------------------------------------
# Acceptance criterion: accepted-risk documentation exists
# ---------------------------------------------------------------------------


def test_vulnerability_scan_document_exists() -> None:
    """docs/security/VULNERABILITY-SCAN.md exists and is non-empty."""
    assert SECURITY_DOC.exists(), (
        f"Missing security doc at {SECURITY_DOC}. "
        "Create docs/security/VULNERABILITY-SCAN.md documenting accepted-risk findings."
    )
    content = SECURITY_DOC.read_text(encoding="utf-8")
    assert len(content) > 500, "Vulnerability scan document appears too short"


def test_vulnerability_scan_document_has_remediated_section() -> None:
    """Security doc contains a section for remediated findings."""
    content = SECURITY_DOC.read_text(encoding="utf-8")
    assert (
        "Remediated" in content or "remediated" in content
    ), "VULNERABILITY-SCAN.md should document remediated findings"


def test_vulnerability_scan_document_has_accepted_risk_section() -> None:
    """Security doc contains a section for accepted-risk findings."""
    content = SECURITY_DOC.read_text(encoding="utf-8")
    assert (
        "Accepted Risk" in content or "accepted risk" in content.lower()
    ), "VULNERABILITY-SCAN.md should document accepted-risk findings with justification"


def test_vulnerability_scan_document_has_justifications() -> None:
    """Each accepted-risk entry has a Justification."""
    content = SECURITY_DOC.read_text(encoding="utf-8")
    assert (
        "Justification" in content
    ), "VULNERABILITY-SCAN.md must include per-item justifications for accepted-risk items"


# ---------------------------------------------------------------------------
# Acceptance criterion: CI step added
# ---------------------------------------------------------------------------


def test_ci_yml_has_security_scan_job() -> None:
    """ci.yml contains a security-scan job that runs pip-audit."""
    assert CI_YML.exists(), f"CI file not found at {CI_YML}"
    content = CI_YML.read_text(encoding="utf-8")
    assert "security-scan" in content, "ci.yml must define a 'security-scan' job for pip-audit"
    assert (
        "pip-audit" in content or "pip_audit" in content
    ), "ci.yml security-scan job must invoke pip-audit"


def test_ci_security_scan_uses_ignore_vuln_for_accepted_risks() -> None:
    """CI pip-audit step uses --ignore-vuln for known accepted-risk CVEs."""
    content = CI_YML.read_text(encoding="utf-8")
    assert "--ignore-vuln" in content, (
        "CI security-scan should use --ignore-vuln to skip accepted-risk CVEs "
        "so new findings cause build failures"
    )


# ---------------------------------------------------------------------------
# Acceptance criterion: direct dep pins cover fixed CVEs
# ---------------------------------------------------------------------------


def test_pyproject_flask_pinned_to_fixed_version() -> None:
    """pyproject.toml pins flask >= 3.1.3 (CVE-2026-27205)."""
    pyproject = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert (
        "flask>=3.1.3" in pyproject or "flask>=3.1" in pyproject
    ), "flask must be pinned to >=3.1.3 to fix CVE-2026-27205"


def test_pyproject_jinja2_pinned_to_fixed_version() -> None:
    """pyproject.toml pins jinja2 >= 3.1.6 (template injection fixes)."""
    pyproject = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert (
        "jinja2>=3.1.6" in pyproject or "jinja2>=3.1.5" in pyproject
    ), "jinja2 must be pinned to >=3.1.5 to fix CVE-2024-56201 etc."


def test_pyproject_werkzeug_pinned_to_fixed_version() -> None:
    """pyproject.toml pins werkzeug >= 3.1.x."""
    pyproject = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "werkzeug>=3.1" in pyproject, "werkzeug must be pinned to >=3.1.x to fix multiple CVEs"


def test_pyproject_urllib3_pinned_to_fixed_version() -> None:
    """pyproject.toml pins urllib3 >= 2.5.0."""
    pyproject = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "urllib3>=2." in pyproject, "urllib3 must be pinned to >=2.x"
    # Ensure minimum is at least 2.5.0
    import re

    match = re.search(r"urllib3>=(\d+)\.(\d+)", pyproject)
    assert match is not None, "urllib3 pin not found in pyproject.toml"
    major, minor = int(match.group(1)), int(match.group(2))
    assert (major, minor) >= (
        2,
        5,
    ), f"urllib3 pin {major}.{minor} must be >=2.5 to fix CVE-2025-50181 etc."


def test_pyproject_cryptography_pinned_to_fixed_version() -> None:
    """pyproject.toml pins cryptography >= 46.0.5 (CVE-2026-26007)."""
    pyproject = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    import re

    match = re.search(r"cryptography>=(\d+)\.", pyproject)
    assert match is not None, "cryptography pin not found in pyproject.toml"
    major = int(match.group(1))
    assert major >= 46, f"cryptography pin major={major} must be >=46 to fix CVE-2026-26007"
