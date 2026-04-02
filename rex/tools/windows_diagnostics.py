"""Windows diagnostics tools for Rex (Phase 6 — US-WIN-002).

Provides system health information via psutil.  Each function returns a
structured dict suitable for LLM context injection.

All functions operate on any platform psutil supports (Windows, macOS, Linux).
Battery status returns a ``platform_not_supported`` status on systems where
battery sensors are unavailable.

Functions are designed as tool handlers: they accept ``**kwargs`` so they can
be invoked uniformly by ``ToolDispatcher``.
"""

from __future__ import annotations

import datetime
import logging
import platform
import socket
from typing import Any

import psutil

logger = logging.getLogger(__name__)


def get_system_info(**kwargs: Any) -> dict[str, Any]:
    """Return basic OS and hardware information.

    Returns:
        Dict with keys: ``platform``, ``platform_version``, ``architecture``,
        ``hostname``, ``cpu_count``, ``total_memory_gb``, ``boot_time``.
    """
    try:
        mem = psutil.virtual_memory()
        boot_str = datetime.datetime.fromtimestamp(psutil.boot_time()).isoformat()

        result: dict[str, Any] = {
            "platform": platform.system(),
            "platform_version": platform.version(),
            "architecture": platform.machine(),
            "hostname": socket.gethostname(),
            "cpu_count": psutil.cpu_count(logical=True),
            "total_memory_gb": round(mem.total / (1024**3), 2),
            "boot_time": boot_str,
        }
        logger.info("windows_diagnostics: get_system_info ok")
        return result
    except Exception as exc:
        logger.warning("windows_diagnostics: get_system_info error: %s", exc)
        return {"error": str(exc)}


def get_cpu_usage(**kwargs: Any) -> dict[str, Any]:
    """Return current CPU usage statistics.

    Returns:
        Dict with keys: ``usage_percent`` (overall), ``per_core_percent``
        (list), ``frequency_mhz``, ``cpu_count_logical``,
        ``cpu_count_physical``.
    """
    try:
        freq = psutil.cpu_freq()
        result: dict[str, Any] = {
            "usage_percent": psutil.cpu_percent(interval=0.1),
            "per_core_percent": psutil.cpu_percent(interval=0.1, percpu=True),
            "frequency_mhz": round(freq.current, 1) if freq else None,
            "cpu_count_logical": psutil.cpu_count(logical=True),
            "cpu_count_physical": psutil.cpu_count(logical=False),
        }
        logger.info("windows_diagnostics: get_cpu_usage ok")
        return result
    except Exception as exc:
        logger.warning("windows_diagnostics: get_cpu_usage error: %s", exc)
        return {"error": str(exc)}


def get_memory_usage(**kwargs: Any) -> dict[str, Any]:
    """Return current RAM and swap usage.

    Returns:
        Dict with keys: ``total_gb``, ``available_gb``, ``used_gb``,
        ``percent``, ``swap_total_gb``, ``swap_used_gb``, ``swap_percent``.
    """
    try:
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()
        result: dict[str, Any] = {
            "total_gb": round(mem.total / (1024**3), 2),
            "available_gb": round(mem.available / (1024**3), 2),
            "used_gb": round(mem.used / (1024**3), 2),
            "percent": mem.percent,
            "swap_total_gb": round(swap.total / (1024**3), 2),
            "swap_used_gb": round(swap.used / (1024**3), 2),
            "swap_percent": swap.percent,
        }
        logger.info("windows_diagnostics: get_memory_usage ok")
        return result
    except Exception as exc:
        logger.warning("windows_diagnostics: get_memory_usage error: %s", exc)
        return {"error": str(exc)}


def get_disk_usage(**kwargs: Any) -> dict[str, Any]:
    """Return disk usage for all mounted partitions.

    Returns:
        Dict with key ``partitions``: a list of dicts each containing
        ``device``, ``mountpoint``, ``fstype``, ``total_gb``, ``used_gb``,
        ``free_gb``, ``percent``.
    """
    try:
        partitions = []
        for part in psutil.disk_partitions(all=False):
            try:
                usage = psutil.disk_usage(part.mountpoint)
                partitions.append(
                    {
                        "device": part.device,
                        "mountpoint": part.mountpoint,
                        "fstype": part.fstype,
                        "total_gb": round(usage.total / (1024**3), 2),
                        "used_gb": round(usage.used / (1024**3), 2),
                        "free_gb": round(usage.free / (1024**3), 2),
                        "percent": usage.percent,
                    }
                )
            except (PermissionError, OSError):
                # Skip inaccessible partitions (e.g. optical drives on Windows)
                continue

        result: dict[str, Any] = {"partitions": partitions}
        logger.info("windows_diagnostics: get_disk_usage ok (%d partitions)", len(partitions))
        return result
    except Exception as exc:
        logger.warning("windows_diagnostics: get_disk_usage error: %s", exc)
        return {"error": str(exc)}


def get_battery_status(**kwargs: Any) -> dict[str, Any]:
    """Return battery status if available.

    Returns:
        Dict with keys ``percent``, ``plugged_in``, ``time_left_seconds``
        when a battery is present; or ``status: platform_not_supported`` when
        no battery sensor is available.
    """
    try:
        battery = psutil.sensors_battery()
        if battery is None:
            return {"status": "platform_not_supported", "detail": "No battery sensor detected"}

        result: dict[str, Any] = {
            "percent": battery.percent,
            "plugged_in": battery.power_plugged,
            "time_left_seconds": battery.secsleft if battery.secsleft >= 0 else None,
        }
        logger.info("windows_diagnostics: get_battery_status ok")
        return result
    except AttributeError:
        # psutil.sensors_battery may not exist on some builds/platforms
        return {"status": "platform_not_supported", "detail": "sensors_battery not available"}
    except Exception as exc:
        logger.warning("windows_diagnostics: get_battery_status error: %s", exc)
        return {"error": str(exc)}


def list_processes(**kwargs: Any) -> dict[str, Any]:
    """Return a list of running processes sorted by CPU usage descending.

    Returns at most 50 processes.

    Returns:
        Dict with key ``processes``: list of dicts each containing ``pid``,
        ``name``, ``cpu_percent``, ``memory_mb``, ``status``.
    """
    try:
        procs = []
        for proc in psutil.process_iter(["pid", "name", "cpu_percent", "memory_info", "status"]):
            try:
                info = proc.info
                mem_mb = (
                    round(info["memory_info"].rss / (1024 * 1024), 1)
                    if info.get("memory_info")
                    else 0.0
                )
                procs.append(
                    {
                        "pid": info["pid"],
                        "name": info["name"] or "",
                        "cpu_percent": info["cpu_percent"] or 0.0,
                        "memory_mb": mem_mb,
                        "status": info["status"] or "",
                    }
                )
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        procs.sort(key=lambda p: p["cpu_percent"], reverse=True)
        result: dict[str, Any] = {"processes": procs[:50]}
        logger.info(
            "windows_diagnostics: list_processes ok (%d returned)", len(result["processes"])
        )
        return result
    except Exception as exc:
        logger.warning("windows_diagnostics: list_processes error: %s", exc)
        return {"error": str(exc)}
