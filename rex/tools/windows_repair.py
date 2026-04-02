"""Windows issue diagnosis and repair tools (Phase 6 — US-WIN-004).

Each function executes PowerShell or system commands via ``subprocess``.
On non-Windows platforms every function returns ``{"status": "platform_not_supported"}``.

Elevation-required operations (``run_sfc_scan``) check the ``confirmed``
kwarg before proceeding and return a confirmation-request dict when not yet
confirmed.

All functions return a structured dict with keys:
    status              - "ok", "warning", "error", or "platform_not_supported"
    findings            - list of human-readable observation strings
    recommended_actions - list of suggested next-step strings

Functions are designed as tool handlers: they accept ``**kwargs`` so they can
be invoked uniformly by ``ToolDispatcher``.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Platform guard
# ---------------------------------------------------------------------------

_IS_WINDOWS = sys.platform == "win32"

_NOT_SUPPORTED: dict[str, Any] = {
    "status": "platform_not_supported",
    "findings": ["This tool requires Windows."],
    "recommended_actions": [],
}


def _platform_not_supported() -> dict[str, Any]:
    return dict(_NOT_SUPPORTED)


# ---------------------------------------------------------------------------
# PowerShell helper
# ---------------------------------------------------------------------------


def _run_ps(script: str, timeout: float = 30.0) -> str:
    """Execute *script* in PowerShell and return stdout as a stripped string.

    Raises ``RuntimeError`` on non-zero exit.
    """
    result = subprocess.run(
        ["powershell", "-NonInteractive", "-NoProfile", "-Command", script],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"PowerShell error (rc={result.returncode}): {result.stderr.strip()}"
        )
    return result.stdout.strip()


# ---------------------------------------------------------------------------
# Disk health
# ---------------------------------------------------------------------------


def check_disk_health(**kwargs: Any) -> dict[str, Any]:
    """Check disk health via SMART data using PowerShell / WMI.

    Returns:
        Structured dict with ``status``, ``findings``, ``recommended_actions``.
    """
    if not _IS_WINDOWS:
        return _platform_not_supported()

    try:
        script = r"""
Get-WmiObject -Namespace root\wmi -Class MSStorageDriver_FailurePredictStatus |
  Select-Object InstanceName, PredictFailure, Reason |
  ForEach-Object {
    "$($_.InstanceName)|$($_.PredictFailure)|$($_.Reason)"
  }
"""
        output = _run_ps(script, timeout=20.0)
        findings: list[str] = []
        recommended: list[str] = []
        overall = "ok"

        if not output:
            findings.append("No SMART data available — disk may not support SMART reporting.")
            recommended.append(
                "Run 'chkdsk C: /f' from an elevated prompt to check for filesystem errors."
            )
            overall = "warning"
        else:
            for line in output.splitlines():
                parts = line.split("|")
                if len(parts) < 2:
                    continue
                instance = parts[0].strip()
                predict_failure = parts[1].strip().lower() == "true"
                reason = parts[2].strip() if len(parts) > 2 else ""

                if predict_failure:
                    overall = "warning"
                    findings.append(
                        f"Drive '{instance}' reports SMART failure prediction "
                        f"(reason code: {reason or 'N/A'})."
                    )
                    recommended.append(
                        f"Back up data from '{instance}' immediately and consider replacing the drive."
                    )
                else:
                    findings.append(f"Drive '{instance}': SMART status OK.")

        logger.info("windows_repair: check_disk_health ok (status=%s)", overall)
        return {
            "status": overall,
            "findings": findings,
            "recommended_actions": recommended,
        }
    except Exception as exc:
        logger.warning("windows_repair: check_disk_health error: %s", exc)
        return {
            "status": "error",
            "findings": [f"Could not retrieve disk health: {exc}"],
            "recommended_actions": [
                "Ensure you are running as Administrator for full SMART access."
            ],
        }


# ---------------------------------------------------------------------------
# Windows Update status
# ---------------------------------------------------------------------------


def check_windows_update_status(**kwargs: Any) -> dict[str, Any]:
    """Check for pending Windows updates via the Windows Update Agent COM object.

    Returns:
        Structured dict with ``status``, ``findings``, ``recommended_actions``.
    """
    if not _IS_WINDOWS:
        return _platform_not_supported()

    try:
        script = r"""
$session = New-Object -ComObject Microsoft.Update.Session
$searcher = $session.CreateUpdateSearcher()
$results = $searcher.Search("IsInstalled=0 AND Type='Software'")
$results.Updates.Count
foreach ($u in $results.Updates) { $u.Title }
"""
        output = _run_ps(script, timeout=60.0)
        lines = [ln.strip() for ln in output.splitlines() if ln.strip()]
        findings: list[str] = []
        recommended: list[str] = []

        if not lines:
            findings.append("Windows Update query returned no data.")
            recommended.append("Open Windows Update settings to verify update status manually.")
            status = "warning"
        else:
            try:
                count = int(lines[0])
            except ValueError:
                count = len(lines) - 1  # fallback

            if count == 0:
                findings.append("Windows is up to date — no pending updates found.")
                status = "ok"
            else:
                status = "warning"
                findings.append(f"{count} pending update(s) found.")
                for title in lines[1 : count + 1]:
                    findings.append(f"  • {title}")
                recommended.append(
                    "Run Windows Update (Settings → Windows Update → Check for updates) "
                    "to install pending updates."
                )

        logger.info("windows_repair: check_windows_update_status ok (status=%s)", status)
        return {
            "status": status,
            "findings": findings,
            "recommended_actions": recommended,
        }
    except Exception as exc:
        logger.warning("windows_repair: check_windows_update_status error: %s", exc)
        return {
            "status": "error",
            "findings": [f"Could not check Windows Update status: {exc}"],
            "recommended_actions": [
                "Open Windows Update settings manually to check for updates."
            ],
        }


# ---------------------------------------------------------------------------
# DNS cache flush
# ---------------------------------------------------------------------------


def flush_dns_cache(**kwargs: Any) -> dict[str, Any]:
    """Flush the Windows DNS resolver cache via ``ipconfig /flushdns``.

    Returns:
        Structured dict with ``status``, ``findings``, ``recommended_actions``.
    """
    if not _IS_WINDOWS:
        return _platform_not_supported()

    try:
        result = subprocess.run(
            ["ipconfig", "/flushdns"],
            capture_output=True,
            text=True,
            timeout=10.0,
        )
        output = result.stdout.strip()
        stderr = result.stderr.strip()

        if result.returncode == 0:
            logger.info("windows_repair: flush_dns_cache ok")
            return {
                "status": "ok",
                "findings": [output or "DNS resolver cache successfully flushed."],
                "recommended_actions": [],
            }
        else:
            logger.warning("windows_repair: flush_dns_cache failed rc=%d", result.returncode)
            return {
                "status": "error",
                "findings": [f"ipconfig /flushdns failed: {stderr or output}"],
                "recommended_actions": [
                    "Run 'ipconfig /flushdns' from an elevated Command Prompt."
                ],
            }
    except Exception as exc:
        logger.warning("windows_repair: flush_dns_cache error: %s", exc)
        return {
            "status": "error",
            "findings": [f"Could not flush DNS cache: {exc}"],
            "recommended_actions": [
                "Run 'ipconfig /flushdns' from an elevated Command Prompt."
            ],
        }


# ---------------------------------------------------------------------------
# System File Checker (requires elevation)
# ---------------------------------------------------------------------------


def run_sfc_scan(confirmed: bool = False, **kwargs: Any) -> dict[str, Any]:
    """Run System File Checker (``sfc /scannow``) — requires elevation.

    Because SFC requires Administrator privileges, this function returns a
    confirmation-request dict when ``confirmed`` is ``False``.

    Args:
        confirmed: Must be ``True`` for the scan to proceed.

    Returns:
        Structured dict with ``status``, ``findings``, ``recommended_actions``,
        or a ``requires_confirmation`` dict when not yet confirmed.
    """
    if not _IS_WINDOWS:
        return _platform_not_supported()

    if not confirmed:
        return {
            "requires_confirmation": True,
            "action": "run_sfc_scan(confirmed=True)",
            "message": (
                "System File Checker (sfc /scannow) requires Administrator elevation "
                "and may take several minutes. Proceed?"
            ),
            "status": "pending_confirmation",
            "findings": ["SFC scan requires elevation and user confirmation before running."],
            "recommended_actions": ["Confirm to start the scan."],
        }

    try:
        result = subprocess.run(
            ["sfc", "/scannow"],
            capture_output=True,
            text=True,
            timeout=300.0,  # SFC can take several minutes
        )
        # SFC output is UTF-16 on some Windows versions; decode if needed
        raw_out = result.stdout.strip()
        raw_err = result.stderr.strip()
        output = raw_out or raw_err

        findings: list[str] = []
        recommended: list[str] = []

        # Parse known SFC result phrases
        output_lower = output.lower()
        if "did not find any integrity violations" in output_lower:
            status = "ok"
            findings.append("SFC found no integrity violations — system files are intact.")
        elif "found corrupt files and successfully repaired" in output_lower:
            status = "ok"
            findings.append("SFC found and repaired corrupted system files.")
            recommended.append("Restart Windows to complete the repair.")
        elif "found corrupt files but was unable to fix" in output_lower:
            status = "warning"
            findings.append("SFC found corrupted files but could not repair them.")
            recommended.append(
                "Run 'DISM /Online /Cleanup-Image /RestoreHealth' then repeat sfc /scannow."
            )
        elif result.returncode != 0:
            status = "error"
            findings.append(f"SFC exited with code {result.returncode}.")
            if output:
                findings.append(output[:500])
            recommended.append("Ensure you are running as Administrator and try again.")
        else:
            status = "ok"
            findings.append(output[:500] if output else "SFC scan completed.")

        logger.info("windows_repair: run_sfc_scan ok (status=%s)", status)
        return {
            "status": status,
            "findings": findings,
            "recommended_actions": recommended,
        }
    except Exception as exc:
        logger.warning("windows_repair: run_sfc_scan error: %s", exc)
        return {
            "status": "error",
            "findings": [f"Could not run SFC scan: {exc}"],
            "recommended_actions": [
                "Run 'sfc /scannow' from an elevated Command Prompt manually."
            ],
        }
