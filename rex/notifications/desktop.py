"""Cross-platform desktop notification helper.

Tries backends in order:
1. ``plyer`` — cross-platform, optional dependency
2. Platform-native fallbacks (Windows toast via PowerShell, macOS osascript,
   Linux notify-send)
3. Log-only fallback — logs a warning so the app never crashes

Usage::

    from rex.notifications.desktop import notify

    notify("Rex Alert", "Your timer has expired.")
    notify("Reminder", "Meeting in 5 minutes.", urgency="high")
"""

from __future__ import annotations

import importlib.util
import logging
import platform
import subprocess
import sys
from typing import Literal

logger = logging.getLogger(__name__)

Urgency = Literal["low", "normal", "high", "critical"]

# Map urgency strings to plyer / notify-send urgency levels
_LIBNOTIFY_URGENCY: dict[str, str] = {
    "low": "low",
    "normal": "normal",
    "high": "critical",
    "critical": "critical",
}


# ---------------------------------------------------------------------------
# Backend implementations
# ---------------------------------------------------------------------------


def _notify_plyer(title: str, message: str, urgency: Urgency) -> bool:
    """Attempt to send a notification via plyer.

    Returns ``True`` on success, ``False`` if plyer is unavailable or fails.
    """
    if importlib.util.find_spec("plyer") is None:
        return False
    try:
        from plyer import notification  # noqa: PLC0415

        notification.notify(title=title, message=message, app_name="Rex")
        return True
    except Exception as exc:  # noqa: BLE001
        logger.debug("plyer notification failed: %s", exc)
        return False


def _notify_windows(title: str, message: str) -> bool:
    """Send a Windows toast notification via PowerShell.

    Returns ``True`` on success, ``False`` otherwise.
    """
    if sys.platform != "win32":
        return False
    try:
        # Use the Windows 10/11 toast notification API via PowerShell
        script = (
            "[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, "
            "ContentType = WindowsRuntime] | Out-Null; "
            "[Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom.XmlDocument, "
            "ContentType = WindowsRuntime] | Out-Null; "
            "$xml = [Windows.UI.Notifications.ToastNotificationManager]"
            "::GetTemplateContent([Windows.UI.Notifications.ToastTemplateType]::ToastText02); "
            "$nodes = $xml.GetElementsByTagName('text'); "
            f"$nodes[0].AppendChild($xml.CreateTextNode('{title}')) | Out-Null; "
            f"$nodes[1].AppendChild($xml.CreateTextNode('{message}')) | Out-Null; "
            "$toast = [Windows.UI.Notifications.ToastNotification]::new($xml); "
            "[Windows.UI.Notifications.ToastNotificationManager]"
            "::CreateToastNotifier('Rex').Show($toast)"
        )
        result = subprocess.run(
            ["powershell", "-NonInteractive", "-Command", script],
            capture_output=True,
            timeout=10,
        )
        return result.returncode == 0
    except Exception as exc:  # noqa: BLE001
        logger.debug("Windows toast notification failed: %s", exc)
        return False


def _notify_macos(title: str, message: str) -> bool:
    """Send a macOS notification via osascript.

    Returns ``True`` on success, ``False`` otherwise.
    """
    if sys.platform != "darwin":
        return False
    try:
        safe_title = title.replace('"', '\\"')
        safe_message = message.replace('"', '\\"')
        script = f'display notification "{safe_message}" with title "{safe_title}"'
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            timeout=10,
        )
        return result.returncode == 0
    except Exception as exc:  # noqa: BLE001
        logger.debug("macOS osascript notification failed: %s", exc)
        return False


def _notify_linux(title: str, message: str, urgency: Urgency) -> bool:
    """Send a Linux desktop notification via notify-send.

    Returns ``True`` on success, ``False`` otherwise.
    """
    if not sys.platform.startswith("linux"):
        return False
    try:
        libnotify_urgency = _LIBNOTIFY_URGENCY.get(urgency, "normal")
        result = subprocess.run(
            ["notify-send", f"--urgency={libnotify_urgency}", "--app-name=Rex", title, message],
            capture_output=True,
            timeout=10,
        )
        return result.returncode == 0
    except Exception as exc:  # noqa: BLE001
        logger.debug("Linux notify-send notification failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def notify(title: str, message: str, urgency: Urgency = "normal") -> None:
    """Show a desktop notification.

    Tries multiple backends in priority order:
    1. plyer (cross-platform, optional dependency)
    2. Platform-native fallback (PowerShell on Windows, osascript on macOS,
       notify-send on Linux)
    3. Log-only fallback (warns instead of raising)

    Args:
        title: Short heading shown in the notification popup.
        message: Body text of the notification.
        urgency: Urgency level — ``"low"``, ``"normal"``, ``"high"``, or
            ``"critical"``.  Defaults to ``"normal"``.
    """
    _os = platform.system()

    if _notify_plyer(title, message, urgency):
        return

    if _os == "Windows" and _notify_windows(title, message):
        return

    if _os == "Darwin" and _notify_macos(title, message):
        return

    if _os == "Linux" and _notify_linux(title, message, urgency):
        return

    logger.warning(
        "Desktop notification system unavailable — could not display: [%s] %s",
        title,
        message,
    )


__all__ = ["Urgency", "notify"]
