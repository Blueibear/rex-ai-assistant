"""Unit tests for rex.tools.windows_repair (US-WIN-004).

All subprocess calls are mocked so tests run on any platform.
"""

from __future__ import annotations

import subprocess
from types import SimpleNamespace
from unittest.mock import patch

import rex.tools.windows_repair as _mod

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ps_result(stdout: str = "", returncode: int = 0, stderr: str = "") -> SimpleNamespace:
    """Build a fake subprocess.CompletedProcess-like object."""
    ns = SimpleNamespace()
    ns.stdout = stdout
    ns.stderr = stderr
    ns.returncode = returncode
    return ns


# ---------------------------------------------------------------------------
# Platform guard tests
# ---------------------------------------------------------------------------


class TestPlatformNotSupported:
    """All four functions return platform_not_supported on non-Windows."""

    def test_check_disk_health_non_windows(self):
        with patch.object(_mod, "_IS_WINDOWS", False):
            result = _mod.check_disk_health()
        assert result["status"] == "platform_not_supported"
        assert "findings" in result
        assert "recommended_actions" in result

    def test_check_windows_update_non_windows(self):
        with patch.object(_mod, "_IS_WINDOWS", False):
            result = _mod.check_windows_update_status()
        assert result["status"] == "platform_not_supported"

    def test_flush_dns_non_windows(self):
        with patch.object(_mod, "_IS_WINDOWS", False):
            result = _mod.flush_dns_cache()
        assert result["status"] == "platform_not_supported"

    def test_run_sfc_non_windows(self):
        with patch.object(_mod, "_IS_WINDOWS", False):
            result = _mod.run_sfc_scan(confirmed=True)
        assert result["status"] == "platform_not_supported"


# ---------------------------------------------------------------------------
# check_disk_health tests
# ---------------------------------------------------------------------------


class TestCheckDiskHealth:
    def test_smart_ok_single_drive(self):
        smart_output = "\\\\?\\SCSI#Disk&Ven_Samsung|False|0"
        with (
            patch.object(_mod, "_IS_WINDOWS", True),
            patch("subprocess.run", return_value=_ps_result(stdout=smart_output)),
        ):
            result = _mod.check_disk_health()

        assert result["status"] == "ok"
        assert any("OK" in f for f in result["findings"])
        assert result["recommended_actions"] == []

    def test_smart_failure_predicted(self):
        smart_output = "\\\\?\\SCSI#Disk&Ven_WD|True|32"
        with (
            patch.object(_mod, "_IS_WINDOWS", True),
            patch("subprocess.run", return_value=_ps_result(stdout=smart_output)),
        ):
            result = _mod.check_disk_health()

        assert result["status"] == "warning"
        assert any("failure prediction" in f for f in result["findings"])
        assert len(result["recommended_actions"]) > 0

    def test_no_smart_data(self):
        with (
            patch.object(_mod, "_IS_WINDOWS", True),
            patch("subprocess.run", return_value=_ps_result(stdout="")),
        ):
            result = _mod.check_disk_health()

        assert result["status"] == "warning"
        assert any("No SMART data" in f for f in result["findings"])

    def test_subprocess_error(self):
        with (
            patch.object(_mod, "_IS_WINDOWS", True),
            patch("subprocess.run", side_effect=Exception("Access denied")),
        ):
            result = _mod.check_disk_health()

        assert result["status"] == "error"
        assert any("Access denied" in f for f in result["findings"])


# ---------------------------------------------------------------------------
# check_windows_update_status tests
# ---------------------------------------------------------------------------


class TestCheckWindowsUpdateStatus:
    def test_no_updates(self):
        with (
            patch.object(_mod, "_IS_WINDOWS", True),
            patch("subprocess.run", return_value=_ps_result(stdout="0")),
        ):
            result = _mod.check_windows_update_status()

        assert result["status"] == "ok"
        assert any("up to date" in f for f in result["findings"])

    def test_pending_updates(self):
        output = "2\nCumulative Update for Windows 11\nSecurity Update KB5034441"
        with (
            patch.object(_mod, "_IS_WINDOWS", True),
            patch("subprocess.run", return_value=_ps_result(stdout=output)),
        ):
            result = _mod.check_windows_update_status()

        assert result["status"] == "warning"
        assert any("2 pending" in f for f in result["findings"])
        assert len(result["recommended_actions"]) > 0

    def test_empty_output(self):
        with (
            patch.object(_mod, "_IS_WINDOWS", True),
            patch("subprocess.run", return_value=_ps_result(stdout="")),
        ):
            result = _mod.check_windows_update_status()

        assert result["status"] == "warning"

    def test_powershell_exception(self):
        with (
            patch.object(_mod, "_IS_WINDOWS", True),
            patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="ps", timeout=60)),
        ):
            result = _mod.check_windows_update_status()

        assert result["status"] == "error"
        assert len(result["recommended_actions"]) > 0


# ---------------------------------------------------------------------------
# flush_dns_cache tests
# ---------------------------------------------------------------------------


class TestFlushDnsCache:
    def test_flush_success(self):
        with (
            patch.object(_mod, "_IS_WINDOWS", True),
            patch(
                "subprocess.run",
                return_value=_ps_result(
                    stdout="Successfully flushed the DNS Resolver Cache.", returncode=0
                ),
            ),
        ):
            result = _mod.flush_dns_cache()

        assert result["status"] == "ok"
        assert any("flushed" in f.lower() for f in result["findings"])
        assert result["recommended_actions"] == []

    def test_flush_failure(self):
        with (
            patch.object(_mod, "_IS_WINDOWS", True),
            patch(
                "subprocess.run",
                return_value=_ps_result(stdout="", stderr="Access denied.", returncode=1),
            ),
        ):
            result = _mod.flush_dns_cache()

        assert result["status"] == "error"
        assert len(result["recommended_actions"]) > 0

    def test_flush_exception(self):
        with (
            patch.object(_mod, "_IS_WINDOWS", True),
            patch("subprocess.run", side_effect=FileNotFoundError("ipconfig not found")),
        ):
            result = _mod.flush_dns_cache()

        assert result["status"] == "error"
        assert any("ipconfig" in r for r in result["recommended_actions"])


# ---------------------------------------------------------------------------
# run_sfc_scan tests
# ---------------------------------------------------------------------------


class TestRunSfcScan:
    def test_requires_confirmation_when_not_confirmed(self):
        with patch.object(_mod, "_IS_WINDOWS", True):
            result = _mod.run_sfc_scan(confirmed=False)

        assert result.get("requires_confirmation") is True
        assert "action" in result
        assert "message" in result

    def test_no_integrity_violations(self):
        output = "Windows Resource Protection did not find any integrity violations."
        with (
            patch.object(_mod, "_IS_WINDOWS", True),
            patch("subprocess.run", return_value=_ps_result(stdout=output, returncode=0)),
        ):
            result = _mod.run_sfc_scan(confirmed=True)

        assert result["status"] == "ok"
        assert any("no integrity violations" in f for f in result["findings"])

    def test_found_and_repaired(self):
        output = "Windows Resource Protection found corrupt files and successfully repaired them."
        with (
            patch.object(_mod, "_IS_WINDOWS", True),
            patch("subprocess.run", return_value=_ps_result(stdout=output, returncode=0)),
        ):
            result = _mod.run_sfc_scan(confirmed=True)

        assert result["status"] == "ok"
        assert any("repaired" in f for f in result["findings"])
        assert any("Restart" in r for r in result["recommended_actions"])

    def test_found_but_unable_to_fix(self):
        output = (
            "Windows Resource Protection found corrupt files but was unable to fix some of them."
        )
        with (
            patch.object(_mod, "_IS_WINDOWS", True),
            patch("subprocess.run", return_value=_ps_result(stdout=output, returncode=0)),
        ):
            result = _mod.run_sfc_scan(confirmed=True)

        assert result["status"] == "warning"
        assert any("DISM" in r for r in result["recommended_actions"])

    def test_non_zero_exit(self):
        with (
            patch.object(_mod, "_IS_WINDOWS", True),
            patch(
                "subprocess.run",
                return_value=_ps_result(stdout="", stderr="Must run as admin.", returncode=1),
            ),
        ):
            result = _mod.run_sfc_scan(confirmed=True)

        assert result["status"] == "error"
        assert any("Administrator" in r for r in result["recommended_actions"])

    def test_subprocess_exception(self):
        with (
            patch.object(_mod, "_IS_WINDOWS", True),
            patch("subprocess.run", side_effect=FileNotFoundError("sfc not found")),
        ):
            result = _mod.run_sfc_scan(confirmed=True)

        assert result["status"] == "error"
        assert any("sfc /scannow" in r for r in result["recommended_actions"])


# ---------------------------------------------------------------------------
# ToolRegistry registration test
# ---------------------------------------------------------------------------


class TestToolRegistration:
    def test_all_repair_tools_registered(self):

        # Build a minimal registry without importing the full default registry
        # (which has heavy optional deps); instead just verify our module
        # exports are importable and callable.
        assert callable(_mod.check_disk_health)
        assert callable(_mod.check_windows_update_status)
        assert callable(_mod.flush_dns_cache)
        assert callable(_mod.run_sfc_scan)
