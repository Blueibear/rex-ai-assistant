"""Tests for US-128: Process supervisor configuration.

Verifies that:
- systemd unit files exist for all long-running Rex processes
- every unit file configures automatic restart on failure
- every unit file sets a backoff/burst limit to prevent restart loops
- docs/deployment.md documents the process supervisor section
- liveness-check command is referenced in the deployment docs
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent
SYSTEMD_DIR = REPO_ROOT / "deploy" / "systemd"
DEPLOYMENT_DOC = REPO_ROOT / "docs" / "deployment.md"

EXPECTED_UNITS = [
    "rex-api.service",
    "rex-tts.service",
    "rex-voice.service",
    "rex-agent.service",
]


def _read_unit(name: str) -> str:
    return (SYSTEMD_DIR / name).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. Unit files exist
# ---------------------------------------------------------------------------


class TestUnitFilesExist:
    def test_systemd_directory_exists(self) -> None:
        assert SYSTEMD_DIR.is_dir(), f"deploy/systemd/ directory not found at {SYSTEMD_DIR}"

    @pytest.mark.parametrize("unit", EXPECTED_UNITS)
    def test_unit_file_exists(self, unit: str) -> None:
        path = SYSTEMD_DIR / unit
        assert path.is_file(), f"Unit file not found: {path}"

    @pytest.mark.parametrize("unit", EXPECTED_UNITS)
    def test_unit_file_nonempty(self, unit: str) -> None:
        content = _read_unit(unit)
        assert len(content.strip()) > 0, f"Unit file is empty: {unit}"


# ---------------------------------------------------------------------------
# 2. [Unit] section – description and After= ordering
# ---------------------------------------------------------------------------


class TestUnitSection:
    @pytest.mark.parametrize("unit", EXPECTED_UNITS)
    def test_has_description(self, unit: str) -> None:
        content = _read_unit(unit)
        assert re.search(
            r"^Description\s*=\s*.+", content, re.MULTILINE
        ), f"{unit} missing Description= in [Unit]"

    @pytest.mark.parametrize("unit", EXPECTED_UNITS)
    def test_has_after_network(self, unit: str) -> None:
        content = _read_unit(unit)
        assert "After=" in content, f"{unit} missing After= ordering directive"

    @pytest.mark.parametrize("unit", EXPECTED_UNITS)
    def test_has_unit_section(self, unit: str) -> None:
        content = _read_unit(unit)
        assert "[Unit]" in content, f"{unit} missing [Unit] section"

    @pytest.mark.parametrize("unit", EXPECTED_UNITS)
    def test_has_service_section(self, unit: str) -> None:
        content = _read_unit(unit)
        assert "[Service]" in content, f"{unit} missing [Service] section"

    @pytest.mark.parametrize("unit", EXPECTED_UNITS)
    def test_has_install_section(self, unit: str) -> None:
        content = _read_unit(unit)
        assert "[Install]" in content, f"{unit} missing [Install] section"


# ---------------------------------------------------------------------------
# 3. Restart on failure
# ---------------------------------------------------------------------------


class TestRestartOnFailure:
    @pytest.mark.parametrize("unit", EXPECTED_UNITS)
    def test_restart_on_failure(self, unit: str) -> None:
        content = _read_unit(unit)
        assert re.search(
            r"^Restart\s*=\s*on-failure", content, re.MULTILINE
        ), f"{unit} does not set Restart=on-failure"

    @pytest.mark.parametrize("unit", EXPECTED_UNITS)
    def test_restart_sec(self, unit: str) -> None:
        content = _read_unit(unit)
        assert re.search(
            r"^RestartSec\s*=", content, re.MULTILINE
        ), f"{unit} missing RestartSec= (backoff delay)"


# ---------------------------------------------------------------------------
# 4. Backoff / burst limit to prevent restart loops
# ---------------------------------------------------------------------------


class TestBackoffLimit:
    @pytest.mark.parametrize("unit", EXPECTED_UNITS)
    def test_start_limit_burst(self, unit: str) -> None:
        content = _read_unit(unit)
        assert re.search(
            r"^StartLimitBurst\s*=\s*\d+", content, re.MULTILINE
        ), f"{unit} missing StartLimitBurst= (restart loop guard)"

    @pytest.mark.parametrize("unit", EXPECTED_UNITS)
    def test_start_limit_interval(self, unit: str) -> None:
        content = _read_unit(unit)
        assert re.search(
            r"^StartLimitIntervalSec\s*=", content, re.MULTILINE
        ), f"{unit} missing StartLimitIntervalSec= (restart loop guard)"

    @pytest.mark.parametrize("unit", EXPECTED_UNITS)
    def test_burst_limit_is_finite(self, unit: str) -> None:
        content = _read_unit(unit)
        m = re.search(r"^StartLimitBurst\s*=\s*(\d+)", content, re.MULTILINE)
        assert m is not None, f"{unit} missing StartLimitBurst"
        burst = int(m.group(1))
        assert (
            1 <= burst <= 20
        ), f"{unit} StartLimitBurst={burst} is not a reasonable finite value (1-20)"


# ---------------------------------------------------------------------------
# 5. ExecStart points to a real entry point
# ---------------------------------------------------------------------------


class TestExecStart:
    @pytest.mark.parametrize("unit", EXPECTED_UNITS)
    def test_has_exec_start(self, unit: str) -> None:
        content = _read_unit(unit)
        assert re.search(r"^ExecStart\s*=", content, re.MULTILINE), f"{unit} missing ExecStart="

    def test_api_exec_start_references_flask_proxy(self) -> None:
        content = _read_unit("rex-api.service")
        assert (
            "flask_proxy.py" in content
        ), "rex-api.service ExecStart must reference flask_proxy.py"

    def test_tts_exec_start_references_speak_api(self) -> None:
        content = _read_unit("rex-tts.service")
        assert "rex-speak-api" in content, "rex-tts.service ExecStart must reference rex-speak-api"

    def test_voice_exec_start_references_rex_loop(self) -> None:
        content = _read_unit("rex-voice.service")
        assert "rex_loop.py" in content, "rex-voice.service ExecStart must reference rex_loop.py"

    def test_agent_exec_start_references_rex_agent(self) -> None:
        content = _read_unit("rex-agent.service")
        assert "rex-agent" in content, "rex-agent.service ExecStart must reference rex-agent"


# ---------------------------------------------------------------------------
# 6. WantedBy=multi-user.target (boot integration)
# ---------------------------------------------------------------------------


class TestInstallSection:
    @pytest.mark.parametrize("unit", EXPECTED_UNITS)
    def test_wanted_by_multi_user(self, unit: str) -> None:
        content = _read_unit(unit)
        assert re.search(
            r"^WantedBy\s*=\s*multi-user\.target", content, re.MULTILINE
        ), f"{unit} WantedBy must be multi-user.target for boot-time startup"


# ---------------------------------------------------------------------------
# 7. Documentation in docs/deployment.md
# ---------------------------------------------------------------------------


class TestDeploymentDocumentation:
    def _doc(self) -> str:
        return DEPLOYMENT_DOC.read_text(encoding="utf-8")

    def test_deployment_doc_exists(self) -> None:
        assert DEPLOYMENT_DOC.is_file(), "docs/deployment.md not found"

    def test_doc_mentions_systemd(self) -> None:
        assert "systemd" in self._doc(), "docs/deployment.md must mention systemd"

    def test_doc_mentions_each_unit(self) -> None:
        doc = self._doc()
        for unit in EXPECTED_UNITS:
            assert unit in doc, f"docs/deployment.md must mention {unit}"

    def test_doc_mentions_restart_policy(self) -> None:
        doc = self._doc()
        assert (
            "Restart" in doc or "restart" in doc
        ), "docs/deployment.md must document the restart policy"

    def test_doc_mentions_start_limit(self) -> None:
        doc = self._doc()
        assert (
            "StartLimit" in doc or "burst" in doc.lower()
        ), "docs/deployment.md must document the burst/backoff limit"

    def test_doc_mentions_liveness_check(self) -> None:
        doc = self._doc()
        assert (
            "/health/live" in doc
        ), "docs/deployment.md must reference /health/live as the liveness check"

    def test_doc_mentions_journalctl(self) -> None:
        doc = self._doc()
        assert "journalctl" in doc, "docs/deployment.md must document how to view logs (journalctl)"

    def test_doc_mentions_systemctl_enable(self) -> None:
        doc = self._doc()
        assert (
            "systemctl enable" in doc
        ), "docs/deployment.md must show the systemctl enable command for boot startup"

    def test_doc_mentions_deploy_systemd_path(self) -> None:
        doc = self._doc()
        assert (
            "deploy/systemd" in doc
        ), "docs/deployment.md must reference the deploy/systemd/ directory"
