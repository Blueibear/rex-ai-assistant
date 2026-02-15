"""Unit tests for scripts/security_audit.py.

Covers merge-marker detection, secret scanning, and false-positive suppression
for Markdown separators, fenced code blocks, and inline backtick patterns.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

# Import the audit module directly so we can unit-test scan_file / helpers.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import security_audit  # noqa: E402


# ---------------------------------------------------------------------------
# Merge-marker tests
# ---------------------------------------------------------------------------


class TestMergeMarkerDetection:
    """Merge marker regex should only match real Git conflict tokens."""

    def test_real_conflict_marker_detected(self, tmp_path: Path):
        """A genuine merge conflict block must be detected."""
        f = tmp_path / "conflict.py"
        f.write_text(
            textwrap.dedent("""\
                x = 1
                <<<<<<< HEAD
                y = 2
                =======
                y = 3
                >>>>>>> feature-branch
                z = 4
            """)
        )
        merge, _, _ = security_audit.scan_file(f)
        assert len(merge) == 3, f"Expected 3 merge markers, got {len(merge)}: {merge}"

    def test_markdown_separator_not_flagged(self, tmp_path: Path):
        """Markdown decorative '====...' separators should NOT trigger."""
        f = tmp_path / "doc.md"
        f.write_text(
            textwrap.dedent("""\
                MERGE CONFLICT MARKERS
                ==================================================
                Some text here
                ========================================
            """)
        )
        merge, _, _ = security_audit.scan_file(f)
        assert merge == [], f"Expected no merge markers, got: {merge}"

    def test_fenced_code_block_marker_not_flagged(self, tmp_path: Path):
        """Merge-like tokens inside fenced code blocks in .md should be skipped."""
        f = tmp_path / "example.md"
        f.write_text(
            textwrap.dedent("""\
                # Example
                ```
                <<<<<<< HEAD
                =======
                >>>>>>> branch
                ```
            """)
        )
        merge, _, _ = security_audit.scan_file(f)
        assert merge == [], f"Expected no merge markers inside fenced block, got: {merge}"

    def test_equals_only_exact_seven(self, tmp_path: Path):
        """Exactly seven '=' at the start of a line IS a marker; more is not (if extra chars)."""
        f = tmp_path / "test.txt"
        f.write_text("=======\n========extra\n")
        merge, _, _ = security_audit.scan_file(f)
        # First line is exactly "=======" -> match
        # Second line starts with 8+ '=' -> no match (regex requires exactly 7)
        assert len(merge) == 1


# ---------------------------------------------------------------------------
# Secret detection tests
# ---------------------------------------------------------------------------


class TestSecretDetection:
    """Secret scanning should catch real secrets and skip doc patterns."""

    # Build secret-like strings dynamically so the audit script does not
    # flag *this* test file itself as containing secrets.
    _PK_HEADER = "-----BEGIN " + "PRIVATE KEY-----"
    _PK_FOOTER = "-----END " + "PRIVATE KEY-----"
    _RSA_PK_HEADER = "-----BEGIN RSA " + "PRIVATE KEY-----"

    def test_real_private_key_detected(self, tmp_path: Path):
        """An actual private key block must be flagged."""
        f = tmp_path / "key.txt"
        f.write_text(f"{self._PK_HEADER}\nMIIEvQIBADANBg...\n{self._PK_FOOTER}\n")
        _, _, secrets = security_audit.scan_file(f)
        assert len(secrets) >= 1
        assert any("Private key" in s for s in secrets)

    def test_backtick_wrapped_pattern_not_flagged(self, tmp_path: Path):
        """Pattern text wrapped in backticks in a .md file should be skipped."""
        f = tmp_path / "audit_report.md"
        f.write_text(f"- Private keys: `{self._PK_HEADER}`\n")
        _, _, secrets = security_audit.scan_file(f)
        assert secrets == [], f"Expected no secrets for backtick-wrapped pattern, got: {secrets}"

    def test_fenced_block_secret_not_flagged(self, tmp_path: Path):
        """Secret patterns inside Markdown fenced blocks should be skipped."""
        f = tmp_path / "docs.md"
        f.write_text(f"# Patterns we check\n```\n{self._RSA_PK_HEADER}\n```\n")
        _, _, secrets = security_audit.scan_file(f)
        assert secrets == [], f"Expected no secrets in fenced block, got: {secrets}"

    def test_real_openai_key_detected(self, tmp_path: Path):
        """A realistic OpenAI key (sk- followed by 48 chars) must be flagged."""
        fake_key = "sk-" + "A" * 48
        f = tmp_path / "config.py"
        f.write_text(f'OPENAI_KEY = "{fake_key}"\n')
        _, _, secrets = security_audit.scan_file(f)
        assert len(secrets) >= 1
        assert any("OpenAI" in s for s in secrets)


# ---------------------------------------------------------------------------
# End-to-end script execution
# ---------------------------------------------------------------------------


class TestSecurityAuditScript:
    """End-to-end: running the script on the clean repo should exit 0."""

    def test_clean_repo_exits_zero(self):
        """The script must exit 0 when run against the repository root."""
        result = subprocess.run(
            [sys.executable, "scripts/security_audit.py"],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).resolve().parent.parent),
        )
        assert result.returncode == 0, (
            f"security_audit.py exited {result.returncode}.\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
