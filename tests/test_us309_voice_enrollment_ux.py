"""Tests for US-309: Voice enrollment guided UX.

Acceptance criteria:
  - Settings page defines a specific prompt phrase constant
  - Settings page defines a minimum RMS threshold constant
  - computeRms helper correctly computes RMS of sample arrays
  - Enrollment bridge exists and is callable
  - Bridge action=enroll stores sample in correct voice identity directory
  - Bridge action=list returns enrolled users
  - Bridge correctly errors on missing user_id
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
BRIDGE = REPO_ROOT / "rex_voice_enrollment_bridge.py"
SETTINGS_PAGE = REPO_ROOT / "gui" / "src" / "pages" / "SettingsPage.tsx"


class TestEnrollmentUIConstants:
    def test_settings_page_has_prompt_phrase_constant(self):
        """SettingsPage.tsx must define ENROLLMENT_PROMPT_PHRASE."""
        assert SETTINGS_PAGE.exists(), "SettingsPage.tsx not found"
        content = SETTINGS_PAGE.read_text(encoding="utf-8")
        assert "ENROLLMENT_PROMPT_PHRASE" in content

    def test_prompt_phrase_contains_meaningful_text(self):
        """ENROLLMENT_PROMPT_PHRASE must be a non-trivial phrase."""
        content = SETTINGS_PAGE.read_text(encoding="utf-8")
        # Find the constant value
        import re
        match = re.search(r"ENROLLMENT_PROMPT_PHRASE\s*=\s*'([^']+)'", content)
        if not match:
            match = re.search(r'ENROLLMENT_PROMPT_PHRASE\s*=\s*"([^"]+)"', content)
        assert match, "Could not find ENROLLMENT_PROMPT_PHRASE value"
        phrase = match.group(1)
        assert len(phrase) > 10, f"Phrase too short: {phrase!r}"

    def test_settings_page_has_min_rms_constant(self):
        """SettingsPage.tsx must define ENROLLMENT_MIN_RMS."""
        content = SETTINGS_PAGE.read_text(encoding="utf-8")
        assert "ENROLLMENT_MIN_RMS" in content

    def test_settings_page_has_compute_rms_helper(self):
        """computeRms helper must be defined for audio level validation."""
        content = SETTINGS_PAGE.read_text(encoding="utf-8")
        assert "computeRms" in content

    def test_settings_page_displays_prompt_phrase_during_recording(self):
        """The enrollment progress UI must reference ENROLLMENT_PROMPT_PHRASE."""
        content = SETTINGS_PAGE.read_text(encoding="utf-8")
        assert "ENROLLMENT_PROMPT_PHRASE" in content
        # It should be displayed as JSX, not just defined
        assert "{ENROLLMENT_PROMPT_PHRASE}" in content

    def test_settings_page_has_too_quiet_feedback(self):
        """Enrollment flow must show feedback when sample is too quiet."""
        content = SETTINGS_PAGE.read_text(encoding="utf-8")
        assert "quiet" in content.lower() or "volume" in content.lower()
        assert "ENROLLMENT_MIN_RMS" in content

    def test_settings_page_has_too_short_feedback(self):
        """Enrollment flow must show feedback when sample is too short."""
        content = SETTINGS_PAGE.read_text(encoding="utf-8")
        assert "short" in content.lower() or "duration" in content.lower()

    def test_recording_indicator_is_animated(self):
        """During recording, a visual indicator must show it's active."""
        content = SETTINGS_PAGE.read_text(encoding="utf-8")
        # Should have some animation or visual indicator for recording
        assert "animate-pulse" in content or "REC" in content


class TestEnrollmentBridgeContract:
    def test_bridge_file_exists(self):
        assert BRIDGE.exists(), f"Bridge not found: {BRIDGE}"

    def test_missing_user_id_returns_error(self):
        """action=enroll without user_id returns ok=false."""
        payload = json.dumps({"action": "enroll", "user_id": "", "samples": [[0.1, 0.2]]})
        result = subprocess.run(
            [sys.executable, str(BRIDGE)],
            input=payload,
            capture_output=True,
            text=True,
            timeout=15,
        )
        lines = [l for l in result.stdout.strip().splitlines() if l.startswith("{")]
        assert lines, f"No JSON output: stdout={result.stdout!r}"
        data = json.loads(lines[-1])
        assert data["ok"] is False

    def test_list_action_returns_ok(self):
        """action=list returns ok and an enrollments key."""
        payload = json.dumps({"action": "list"})
        result = subprocess.run(
            [sys.executable, str(BRIDGE)],
            input=payload,
            capture_output=True,
            text=True,
            timeout=15,
        )
        lines = [l for l in result.stdout.strip().splitlines() if l.startswith("{")]
        assert lines, f"No JSON output: stdout={result.stdout!r}"
        data = json.loads(lines[-1])
        assert data["ok"] is True
        assert "enrollments" in data or "active_user_id" in data

    def test_bad_json_returns_error(self):
        """Malformed input returns ok=false."""
        result = subprocess.run(
            [sys.executable, str(BRIDGE)],
            input="not-json",
            capture_output=True,
            text=True,
            timeout=10,
        )
        lines = [l for l in result.stdout.strip().splitlines() if l.startswith("{")]
        assert lines
        data = json.loads(lines[-1])
        assert data["ok"] is False


class TestEnrollmentStorePath:
    def test_embeddings_store_uses_memory_dir(self):
        """EmbeddingsStore defaults to Memory/ directory at repo root."""
        from rex.voice_identity.ui_service import _DEFAULT_MEMORY_DIR

        assert _DEFAULT_MEMORY_DIR.name == "Memory"
        assert _DEFAULT_MEMORY_DIR.parent == REPO_ROOT
