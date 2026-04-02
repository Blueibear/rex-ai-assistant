"""Windows settings read/write via PowerShell bridge (Phase 6 — US-WIN-003).

Each function executes PowerShell cmdlets via ``subprocess``.  On non-Windows
platforms every function raises ``NotImplementedError`` so callers can degrade
gracefully.

Settings changes (``set_*`` functions) respect ``AppConfig.require_confirm_system_changes``:
when ``True`` the function returns a confirmation-request dict instead of
executing immediately.  The caller is responsible for obtaining confirmation
and re-calling with ``confirmed=True``.

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


def _require_windows(fn_name: str) -> None:
    """Raise ``NotImplementedError`` when not on Windows."""
    if not _IS_WINDOWS:
        raise NotImplementedError(
            f"{fn_name} is only supported on Windows (current platform: {sys.platform})"
        )


# ---------------------------------------------------------------------------
# PowerShell helper
# ---------------------------------------------------------------------------


def _run_ps(script: str, timeout: float = 10.0) -> str:
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
        raise RuntimeError(f"PowerShell error (rc={result.returncode}): {result.stderr.strip()}")
    return result.stdout.strip()


# ---------------------------------------------------------------------------
# Volume
# ---------------------------------------------------------------------------


def get_volume(**kwargs: Any) -> dict[str, Any]:
    """Return the current system master volume level (0–100).

    Returns:
        Dict with key ``volume`` (int 0-100).
    """
    _require_windows("get_volume")
    try:
        script = (
            "Add-Type -TypeDefinition '"
            "using System.Runtime.InteropServices;"
            '[Guid("5CDF2C82-841E-4546-9722-0CF74078229A")]'
            "[InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]"
            "interface IAudioEndpointVolume {"
            "    int _VtblGap1_6();"
            "    int SetMasterVolumeLevelScalar(float fLevel, System.Guid pguidEventContext);"
            "    int _VtblGap2_1();"
            "    int GetMasterVolumeLevelScalar(out float pfLevel);"
            "}' -PassThru | Out-Null;"
            # Simpler approach: use the audio API via COM
            "$wshShell = New-Object -ComObject WScript.Shell;"
            "$volume = [math]::Round((Get-WmiObject -Query 'SELECT * FROM Win32_SoundDevice').Count);"
            # Use nircmd or fallback to WScript approach; safest portable method:
            "Add-Type -AssemblyName System.Windows.Forms;"
            "$vol = [System.Windows.Forms.SendKeys]::SendWait('') ; "
            # Use the simpler VBScript/WMI approach
            "(New-Object -ComObject WScript.Shell).SendKeys([char]174) | Out-Null;"
            # Actually the simplest reliable approach:
        )
        # Use a reliable PowerShell approach via Windows Core Audio API
        script = r"""
$vol = (Get-CimInstance -ClassName Win32_SoundDevice).Count
Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;
[ComImport, Guid("BCDE0395-E52F-467C-8E3D-C4579291692E")]
class MMDeviceEnumerator {}
[ComImport, Guid("A95664D2-9614-4F35-A746-DE8DB63617E6"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
interface IMMDeviceEnumerator {
    int NotImpl1();
    [PreserveSig] int GetDefaultAudioEndpoint(int dataFlow, int role, out IMMDevice ppDevice);
}
[ComImport, Guid("D666063F-1587-4E43-81F1-B948E807363F"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
interface IMMDevice {
    [PreserveSig] int Activate(ref Guid iid, int dwClsCtx, IntPtr pActivationParams, [MarshalAs(UnmanagedType.IUnknown)] out object ppInterface);
}
[ComImport, Guid("5CDF2C82-841E-4546-9722-0CF74078229A"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
interface IAudioEndpointVolume {
    int NotImpl1(); int NotImpl2(); int NotImpl3(); int NotImpl4(); int NotImpl5(); int NotImpl6();
    [PreserveSig] int SetMasterVolumeLevelScalar(float fLevel, Guid pguidEventContext);
    int NotImpl7();
    [PreserveSig] int GetMasterVolumeLevelScalar(out float pfLevel);
}
"@
$enumerator = [Activator]::CreateInstance([Type]::GetTypeFromCLSID([Guid]"BCDE0395-E52F-467C-8E3D-C4579291692E"))
$iEnum = [IMMDeviceEnumerator]$enumerator
$device = $null
$iEnum.GetDefaultAudioEndpoint(0, 1, [ref]$device) | Out-Null
$iid = [Guid]"5CDF2C82-841E-4546-9722-0CF74078229A"
$iVolObj = $null
$device.Activate([ref]$iid, 23, [IntPtr]::Zero, [ref]$iVolObj) | Out-Null
$iVol = [IAudioEndpointVolume]$iVolObj
$level = 0.0
$iVol.GetMasterVolumeLevelScalar([ref]$level) | Out-Null
[math]::Round($level * 100)
"""
        output = _run_ps(script)
        volume = int(output.split()[-1]) if output else 0
        logger.info("windows_settings: get_volume ok (%d)", volume)
        return {"volume": volume}
    except Exception as exc:
        logger.warning("windows_settings: get_volume error: %s", exc)
        return {"error": str(exc)}


def set_volume(level: int = 50, confirmed: bool = False, **kwargs: Any) -> dict[str, Any]:
    """Set the system master volume to *level* (0–100).

    When ``AppConfig.require_confirm_system_changes`` is ``True`` and
    ``confirmed`` is ``False``, returns a confirmation-request dict.

    Args:
        level: Target volume level 0-100.
        confirmed: Set to ``True`` after user confirmation.

    Returns:
        Dict with ``volume`` (int) after change, or ``requires_confirmation`` dict.
    """
    _require_windows("set_volume")
    level = max(0, min(100, int(level)))

    config = kwargs.get("config")
    if (
        config is not None
        and getattr(config, "require_confirm_system_changes", True)
        and not confirmed
    ):
        return {
            "requires_confirmation": True,
            "action": f"set_volume({level})",
            "message": f"Set system volume to {level}%?",
        }

    try:
        script = rf"""
Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;
[ComImport, Guid("BCDE0395-E52F-467C-8E3D-C4579291692E")]
class MMDeviceEnumerator {{}}
[ComImport, Guid("A95664D2-9614-4F35-A746-DE8DB63617E6"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
interface IMMDeviceEnumerator {{
    int NotImpl1();
    [PreserveSig] int GetDefaultAudioEndpoint(int dataFlow, int role, out IMMDevice ppDevice);
}}
[ComImport, Guid("D666063F-1587-4E43-81F1-B948E807363F"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
interface IMMDevice {{
    [PreserveSig] int Activate(ref Guid iid, int dwClsCtx, IntPtr pActivationParams, [MarshalAs(UnmanagedType.IUnknown)] out object ppInterface);
}}
[ComImport, Guid("5CDF2C82-841E-4546-9722-0CF74078229A"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
interface IAudioEndpointVolume {{
    int NotImpl1(); int NotImpl2(); int NotImpl3(); int NotImpl4(); int NotImpl5(); int NotImpl6();
    [PreserveSig] int SetMasterVolumeLevelScalar(float fLevel, Guid pguidEventContext);
    int NotImpl7();
    [PreserveSig] int GetMasterVolumeLevelScalar(out float pfLevel);
}}
"@
$enumerator = [Activator]::CreateInstance([Type]::GetTypeFromCLSID([Guid]"BCDE0395-E52F-467C-8E3D-C4579291692E"))
$iEnum = [IMMDeviceEnumerator]$enumerator
$device = $null
$iEnum.GetDefaultAudioEndpoint(0, 1, [ref]$device) | Out-Null
$iid = [Guid]"5CDF2C82-841E-4546-9722-0CF74078229A"
$iVolObj = $null
$device.Activate([ref]$iid, 23, [IntPtr]::Zero, [ref]$iVolObj) | Out-Null
$iVol = [IAudioEndpointVolume]$iVolObj
$iVol.SetMasterVolumeLevelScalar({level / 100:.4f}, [Guid]::Empty) | Out-Null
$readback = 0.0
$iVol.GetMasterVolumeLevelScalar([ref]$readback) | Out-Null
[math]::Round($readback * 100)
"""
        output = _run_ps(script)
        actual = int(output.split()[-1]) if output else level
        logger.info("windows_settings: set_volume ok (requested=%d, actual=%d)", level, actual)
        return {"volume": actual}
    except Exception as exc:
        logger.warning("windows_settings: set_volume error: %s", exc)
        return {"error": str(exc)}


# ---------------------------------------------------------------------------
# Brightness
# ---------------------------------------------------------------------------


def set_brightness(level: int = 50, confirmed: bool = False, **kwargs: Any) -> dict[str, Any]:
    """Set display brightness to *level* (0–100) via WMI.

    Args:
        level: Target brightness 0-100.
        confirmed: Set to ``True`` after user confirmation.

    Returns:
        Dict with ``brightness`` (int) after change, or ``requires_confirmation`` dict.
    """
    _require_windows("set_brightness")
    level = max(0, min(100, int(level)))

    config = kwargs.get("config")
    if (
        config is not None
        and getattr(config, "require_confirm_system_changes", True)
        and not confirmed
    ):
        return {
            "requires_confirmation": True,
            "action": f"set_brightness({level})",
            "message": f"Set display brightness to {level}%?",
        }

    try:
        script = (
            f"(Get-WmiObject -Namespace root/WMI -Class WmiMonitorBrightnessMethods)"
            f".WmiSetBrightness(1, {level});"
            f"(Get-WmiObject -Namespace root/WMI -Class WmiMonitorBrightness).CurrentBrightness"
        )
        output = _run_ps(script)
        actual = int(output.split()[-1]) if output else level
        logger.info("windows_settings: set_brightness ok (requested=%d, actual=%d)", level, actual)
        return {"brightness": actual}
    except Exception as exc:
        logger.warning("windows_settings: set_brightness error: %s", exc)
        return {"error": str(exc)}


# ---------------------------------------------------------------------------
# Power plan
# ---------------------------------------------------------------------------


def get_power_plan(**kwargs: Any) -> dict[str, Any]:
    """Return the name of the currently active power plan.

    Returns:
        Dict with key ``power_plan`` (str) and ``guid`` (str).
    """
    _require_windows("get_power_plan")
    try:
        script = "powercfg /getactivescheme"
        output = _run_ps(script)
        # Output: "Power Scheme GUID: <guid>  (<name>)"
        guid = ""
        name = ""
        if "GUID:" in output:
            parts = output.split("GUID:")[-1].strip()
            # parts: "<guid>  (<name>)" or just "<guid>"
            guid_part = parts.split()[0] if parts else ""
            guid = guid_part.strip()
            if "(" in parts and ")" in parts:
                name = parts[parts.index("(") + 1 : parts.rindex(")")]
        logger.info("windows_settings: get_power_plan ok (%s)", name)
        return {"power_plan": name, "guid": guid}
    except Exception as exc:
        logger.warning("windows_settings: get_power_plan error: %s", exc)
        return {"error": str(exc)}


def set_power_plan(
    name: str = "Balanced", confirmed: bool = False, **kwargs: Any
) -> dict[str, Any]:
    """Activate a power plan by name or GUID.

    Common names: ``"Balanced"``, ``"High performance"``, ``"Power saver"``.

    Args:
        name: Power plan name or GUID string.
        confirmed: Set to ``True`` after user confirmation.

    Returns:
        Dict with ``power_plan`` (str) after change, or ``requires_confirmation`` dict.
    """
    _require_windows("set_power_plan")

    config = kwargs.get("config")
    if (
        config is not None
        and getattr(config, "require_confirm_system_changes", True)
        and not confirmed
    ):
        return {
            "requires_confirmation": True,
            "action": f"set_power_plan({name!r})",
            "message": f"Switch power plan to '{name}'?",
        }

    try:
        # First find the plan by name or guid
        list_script = "powercfg /list"
        output = _run_ps(list_script)
        guid = None
        for line in output.splitlines():
            if name.lower() in line.lower():
                # Extract GUID from: "Power Scheme GUID: <guid>  (<name>)"
                if "GUID:" in line:
                    parts = line.split("GUID:")[-1].strip()
                    guid = parts.split()[0].strip()
                    break
        if guid is None:
            # Maybe name is already a GUID
            guid = name

        set_script = f"powercfg /setactive {guid}"
        _run_ps(set_script)

        # Read back
        readback = get_power_plan()
        actual_name = readback.get("power_plan", name)
        logger.info("windows_settings: set_power_plan ok (%s)", actual_name)
        return {"power_plan": actual_name, "guid": readback.get("guid", guid)}
    except Exception as exc:
        logger.warning("windows_settings: set_power_plan error: %s", exc)
        return {"error": str(exc)}
