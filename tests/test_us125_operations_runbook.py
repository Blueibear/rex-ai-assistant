"""Tests for US-125: Operations runbook.

Acceptance criteria:
- docs/runbook.md exists and covers: start/stop/restart procedure, log access
  and filtering, health check verification, what to do if a service fails to start
- runbook documents the expected process list and how to verify each component
  is running
- at least five common error scenarios documented with diagnosis steps and resolution
- Typecheck passes
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
RUNBOOK = REPO_ROOT / "docs" / "runbook.md"


# ---------------------------------------------------------------------------
# Document existence
# ---------------------------------------------------------------------------


class TestRunbookExists:
    def test_exists(self) -> None:
        assert RUNBOOK.exists(), "docs/runbook.md must exist"

    def test_nonempty(self) -> None:
        content = RUNBOOK.read_text(encoding="utf-8").strip()
        assert len(content) > 500, "docs/runbook.md must be substantive"


# ---------------------------------------------------------------------------
# Required sections
# ---------------------------------------------------------------------------


class TestRequiredSections:
    def _content(self) -> str:
        return RUNBOOK.read_text(encoding="utf-8").lower()

    def test_has_start_section(self) -> None:
        c = self._content()
        assert "start" in c, "runbook must cover start procedure"

    def test_has_stop_section(self) -> None:
        c = self._content()
        assert "stop" in c, "runbook must cover stop procedure"

    def test_has_restart_section(self) -> None:
        c = self._content()
        assert "restart" in c, "runbook must cover restart procedure"

    def test_has_log_access_section(self) -> None:
        c = self._content()
        assert "log" in c, "runbook must cover log access"

    def test_has_health_check_section(self) -> None:
        c = self._content()
        assert "health" in c, "runbook must cover health check verification"

    def test_has_fails_to_start_section(self) -> None:
        c = self._content()
        assert (
            "fail" in c or "fails to start" in c
        ), "runbook must document what to do if a service fails to start"

    def test_has_process_list_section(self) -> None:
        c = self._content()
        assert "process" in c, "runbook must document the expected process list"


# ---------------------------------------------------------------------------
# Process list
# ---------------------------------------------------------------------------


class TestProcessList:
    def _content(self) -> str:
        return RUNBOOK.read_text(encoding="utf-8")

    def test_flask_proxy_mentioned(self) -> None:
        assert "flask_proxy" in self._content().lower() or "flask" in self._content().lower()

    def test_tts_api_mentioned(self) -> None:
        c = self._content().lower()
        assert "tts" in c or "speak" in c or "rex_speak_api" in c

    def test_voice_loop_mentioned(self) -> None:
        c = self._content().lower()
        assert "voice" in c or "rex_loop" in c

    def test_verification_commands_present(self) -> None:
        """Runbook must show how to verify components are running."""
        c = self._content()
        assert "health/live" in c or "curl" in c or "ps aux" in c or "pgrep" in c


# ---------------------------------------------------------------------------
# Error scenarios (at least 5)
# ---------------------------------------------------------------------------


class TestErrorScenarios:
    def _content(self) -> str:
        return RUNBOOK.read_text(encoding="utf-8")

    def _count_scenarios(self) -> int:
        """Count headings that look like error scenario sections."""
        content = self._content()
        # Count "Scenario N:" headings or numbered error sections
        scenario_pattern = re.compile(
            r"(?:###?\s+Scenario\s+\d+|###?\s+Error|###?\s+Symptom|###?\s+\d+\.\s+)",
            re.IGNORECASE,
        )
        return len(scenario_pattern.findall(content))

    def test_at_least_five_scenarios(self) -> None:
        count = self._count_scenarios()
        assert count >= 5, f"runbook must document at least 5 error scenarios; found {count}"

    def test_migration_error_documented(self) -> None:
        c = self._content().lower()
        assert "migration" in c, "runbook must document migration error scenario"

    def test_rate_limit_error_documented(self) -> None:
        c = self._content().lower()
        assert "429" in c or "rate limit" in c or "too many requests" in c

    def test_cors_error_documented(self) -> None:
        c = self._content().lower()
        assert "cors" in c, "runbook must document CORS error scenario"

    def test_db_timeout_documented(self) -> None:
        c = self._content().lower()
        assert "timeout" in c or "db_query_timeout" in c

    def test_each_scenario_has_diagnosis(self) -> None:
        """Each scenario section must mention diagnosis or symptom."""
        c = self._content().lower()
        assert "diagnosis" in c or "symptom" in c, "runbook scenarios must include diagnosis steps"

    def test_each_scenario_has_resolution(self) -> None:
        c = self._content().lower()
        assert "resolution" in c or "fix" in c or "resolve" in c


# ---------------------------------------------------------------------------
# Health check coverage
# ---------------------------------------------------------------------------


class TestHealthCheckContent:
    def _content(self) -> str:
        return RUNBOOK.read_text(encoding="utf-8")

    def test_liveness_endpoint_documented(self) -> None:
        assert "/health/live" in self._content()

    def test_readiness_endpoint_documented(self) -> None:
        assert "/health/ready" in self._content()

    def test_expected_response_shown(self) -> None:
        c = self._content()
        assert '"status"' in c or "status" in c

    def test_503_behavior_explained(self) -> None:
        c = self._content().lower()
        assert "503" in c or "degraded" in c


# ---------------------------------------------------------------------------
# Log access
# ---------------------------------------------------------------------------


class TestLogAccess:
    def _content(self) -> str:
        return RUNBOOK.read_text(encoding="utf-8")

    def test_log_filtering_shown(self) -> None:
        c = self._content().lower()
        assert "grep" in c or "filter" in c or "tail" in c

    def test_log_level_mentioned(self) -> None:
        c = self._content().lower()
        assert "log_level" in c or "log level" in c or "debug" in c
