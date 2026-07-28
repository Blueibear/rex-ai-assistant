"""Tests for US-096: Hardcoded credential and secret scan.

Acceptance criteria:
- detect-secrets (or equivalent) run against the full git history / source
- zero confirmed hardcoded secrets found
- historical findings documented and rotated if real credentials
- pre-commit hook or CI step added to block future secret commits
- Typecheck passes
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
BASELINE_FILE = REPO_ROOT / ".secrets.baseline"
PRECOMMIT_CONFIG = REPO_ROOT / ".pre-commit-config.yaml"
CI_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"
SECRET_SCAN_REPORT = REPO_ROOT / "docs" / "security" / "SECRET-SCAN.md"


# ---------------------------------------------------------------------------
# Baseline file exists and is valid
# ---------------------------------------------------------------------------


class TestBaselineFile:
    def test_baseline_file_exists(self):
        assert BASELINE_FILE.exists(), ".secrets.baseline must exist in repo root"

    def test_baseline_file_is_valid_json(self):
        content = BASELINE_FILE.read_text(encoding="utf-8")
        data = json.loads(content)
        assert isinstance(data, dict)

    def test_baseline_contains_version(self):
        data = json.loads(BASELINE_FILE.read_text(encoding="utf-8"))
        assert "version" in data

    def test_baseline_contains_results_key(self):
        data = json.loads(BASELINE_FILE.read_text(encoding="utf-8"))
        assert "results" in data

    def test_baseline_contains_plugins_used(self):
        data = json.loads(BASELINE_FILE.read_text(encoding="utf-8"))
        assert "plugins_used" in data
        assert len(data["plugins_used"]) > 0


# ---------------------------------------------------------------------------
# detect-secrets scan finds no NEW findings beyond baseline
# ---------------------------------------------------------------------------


class TestSecretScan:
    def test_detect_secrets_installed(self):
        """detect-secrets must be importable."""
        result = subprocess.run(
            [sys.executable, "-m", "detect_secrets", "--version"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, "detect-secrets not installed"
        assert result.stdout.strip(), "detect-secrets version not reported"

    def test_scan_against_baseline_passes(self, tmp_path):
        """Scanning against a baseline copy returns no new findings.

        detect-secrets rewrites the file passed to --baseline in place.
        Scan against a throwaway copy so the tracked baseline is never mutated.
        """
        tracked_baseline_bytes = BASELINE_FILE.read_bytes()
        baseline_copy = tmp_path / ".secrets.baseline"
        shutil.copyfile(BASELINE_FILE, baseline_copy)

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "detect_secrets",
                "scan",
                "--exclude-files",
                r"\.venv|__pycache__|\.git|\.egg-info",
                "--baseline",
                str(baseline_copy),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (
            f"detect-secrets found new secrets not in baseline.\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
        assert BASELINE_FILE.read_bytes() == tracked_baseline_bytes, (
            "Tracked .secrets.baseline was modified by the scan; "
            "the test must only ever write to its tmp_path copy"
        )

    def test_planted_secret_under_config_is_detected(self, tmp_path):
        """US-028: a token-looking string under config/ must fail the scan.

        Proves the scanner actually detects secrets in config JSON rather
        than merely running. Uses a synthetic high-entropy AWS-style key in
        a throwaway directory mimicking config/.
        """
        import base64
        import hashlib

        config_dir = tmp_path / "config"
        config_dir.mkdir()
        planted = config_dir / "gui_settings.json"
        # Derive a deterministic high-entropy value at runtime so no
        # secret-looking string ever appears in this source file.
        derived = base64.b64encode(hashlib.sha256(b"askrex-us028-fixture").digest()).decode()
        planted.write_text(
            json.dumps(
                {
                    "james": {
                        "aws_access_key_id": "AKIA" + "IOSFODNN7EXAMPLE",
                        "aws_secret_access_key": derived[:40],
                    }
                }
            ),
            encoding="utf-8",
        )
        result = subprocess.run(
            [sys.executable, "-m", "detect_secrets", "scan", str(planted)],
            capture_output=True,
            text=True,
            cwd=tmp_path,
        )
        assert result.returncode == 0, result.stderr
        report = json.loads(result.stdout)
        findings = report.get("results", {})
        assert findings, (
            "detect-secrets failed to flag a planted AWS-style secret in a "
            f"config/ JSON file.\nstdout: {result.stdout}"
        )

    def test_no_confirmed_real_secrets_in_baseline(self):
        """Baseline results must contain zero is_verified=True entries."""
        data = json.loads(BASELINE_FILE.read_text(encoding="utf-8"))
        verified = [
            f"{fname}:{entry['line_number']}"
            for fname, entries in data["results"].items()
            for entry in entries
            if entry.get("is_verified", False)
        ]
        assert verified == [], f"Confirmed (is_verified=True) secrets found: {verified}"


# ---------------------------------------------------------------------------
# Pre-commit hook blocks future secret commits
# ---------------------------------------------------------------------------


class TestPreCommitHook:
    def test_precommit_config_exists(self):
        assert PRECOMMIT_CONFIG.exists(), ".pre-commit-config.yaml must exist in repo root"

    def test_precommit_config_is_valid_yaml(self):
        try:
            import yaml  # type: ignore[import-untyped]
        except ImportError:
            pytest.skip("PyYAML not installed; skipping YAML parse test")

        content = PRECOMMIT_CONFIG.read_text(encoding="utf-8")
        data = yaml.safe_load(content)
        assert isinstance(data, dict)
        assert "repos" in data

    def test_precommit_config_includes_detect_secrets_hook(self):
        content = PRECOMMIT_CONFIG.read_text(encoding="utf-8")
        assert (
            "detect-secrets" in content
        ), ".pre-commit-config.yaml must include detect-secrets hook"

    def test_precommit_config_references_baseline(self):
        content = PRECOMMIT_CONFIG.read_text(encoding="utf-8")
        assert (
            ".secrets.baseline" in content
        ), "pre-commit detect-secrets hook must reference .secrets.baseline"


# ---------------------------------------------------------------------------
# CI step blocks future secret commits
# ---------------------------------------------------------------------------


class TestCISecretScanJob:
    def test_ci_workflow_exists(self):
        assert CI_WORKFLOW.exists(), ".github/workflows/ci.yml must exist"

    def test_ci_workflow_includes_secret_scan_job(self):
        content = CI_WORKFLOW.read_text(encoding="utf-8")
        assert "secret-scan" in content, "CI workflow must contain a secret-scan job"

    def test_ci_workflow_uses_detect_secrets(self):
        content = CI_WORKFLOW.read_text(encoding="utf-8")
        assert "detect-secrets" in content, "CI secret-scan job must use detect-secrets"

    def test_ci_workflow_secret_scan_uses_baseline(self):
        content = CI_WORKFLOW.read_text(encoding="utf-8")
        assert ".secrets.baseline" in content, "CI secret-scan job must reference .secrets.baseline"


# ---------------------------------------------------------------------------
# Findings documented
# ---------------------------------------------------------------------------


class TestScanDocumentation:
    def test_secret_scan_report_exists(self):
        assert SECRET_SCAN_REPORT.exists(), "docs/security/SECRET-SCAN.md must exist"

    def test_report_documents_false_positive_rationale(self):
        content = SECRET_SCAN_REPORT.read_text(encoding="utf-8")
        assert (
            "false positive" in content.lower()
        ), "Report must document that all findings are false positives"

    def test_report_documents_remediation_status(self):
        content = SECRET_SCAN_REPORT.read_text(encoding="utf-8")
        assert (
            "rotation" in content.lower() or "rotate" in content.lower()
        ), "Report must address rotation status for any findings"

    def test_report_documents_prevention_mechanism(self):
        content = SECRET_SCAN_REPORT.read_text(encoding="utf-8")
        # Must document at least one prevention approach
        assert (
            "pre-commit" in content.lower() or "ci" in content.upper()
        ), "Report must document the prevention mechanism (pre-commit or CI)"
