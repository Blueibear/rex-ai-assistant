"""Desktop application launcher for Rex (US-054).

Resolves an app name from ``config/app_registry.json`` and launches it using
the appropriate platform command:

- Windows: ``os.startfile()`` (preferred) or ``subprocess`` with ``start``
- macOS:   ``open <executable>``
- Linux:   ``xdg-open <executable>``

Public API
----------
- :func:`launch_app`        — launch an app by its registry name
- :func:`load_app_registry` — load the registry dict from a JSON file
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_REGISTRY_PATH = Path("config") / "app_registry.json"

# Sentinel returned when an app is not found in the registry.
APP_NOT_FOUND_MESSAGE = (
    "I don't know how to open that. You can add it in settings."
)


# ---------------------------------------------------------------------------
# Registry helpers
# ---------------------------------------------------------------------------


def _default_registry_path() -> Path:
    """Return the path to the app registry, preferring the repo root."""
    try:
        from rex.bridge_utils import repo_root

        return repo_root() / "config" / "app_registry.json"
    except Exception:
        return _DEFAULT_REGISTRY_PATH


def load_app_registry(registry_path: "str | Path | None" = None) -> dict[str, str]:
    """Load the app registry from *registry_path*.

    The registry is a JSON object mapping lowercase app names to executable
    paths or commands, e.g.::

        {
          "notepad":  "notepad.exe",
          "chrome":   "C:/Program Files/Google/Chrome/Application/chrome.exe",
          "terminal": "/usr/bin/gnome-terminal"
        }

    Args:
        registry_path: Path to the JSON registry file.
                       ``None`` uses the default config path.

    Returns:
        Dict mapping app name → executable path/command.
        Returns an empty dict if the file does not exist or is invalid.
    """
    path = Path(registry_path) if registry_path is not None else _default_registry_path()
    if not path.exists():
        logger.debug("App registry not found at %s; returning empty registry", path)
        return {}
    try:
        raw: Any = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            logger.warning("App registry at %s is not a JSON object; ignoring", path)
            return {}
        return {k.lower(): str(v) for k, v in raw.items()}
    except Exception as exc:
        logger.warning("Failed to load app registry from %s: %s", path, exc)
        return {}


# ---------------------------------------------------------------------------
# Platform launch
# ---------------------------------------------------------------------------


def _launch_windows(executable: str) -> None:
    """Launch *executable* on Windows using os.startfile."""
    os.startfile(executable)  # noqa: SC200


def _launch_macos(executable: str) -> None:
    """Launch *executable* on macOS using the ``open`` command."""
    subprocess.Popen(["open", executable])


def _launch_linux(executable: str) -> None:
    """Launch *executable* on Linux using ``xdg-open``."""
    subprocess.Popen(["xdg-open", executable])


def _platform_launch(executable: str) -> None:
    """Dispatch to the right platform launcher."""
    platform = sys.platform
    if platform == "win32":
        _launch_windows(executable)
    elif platform == "darwin":
        _launch_macos(executable)
    else:
        _launch_linux(executable)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def launch_app(
    name: str,
    registry_path: "str | Path | None" = None,
) -> str:
    """Launch a desktop application by its registry name.

    Looks up *name* (case-insensitive) in the app registry.  If found,
    launches it using the platform-appropriate method and returns a
    confirmation message.  If not found, returns :data:`APP_NOT_FOUND_MESSAGE`.

    Args:
        name:          The human-friendly app name to launch (e.g. ``"notepad"``).
        registry_path: Path to the registry JSON.  ``None`` uses the default.

    Returns:
        A status string suitable for TTS / display.
    """
    registry = load_app_registry(registry_path)
    key = name.strip().lower()

    if key not in registry:
        logger.info("launch_app: '%s' not found in registry", name)
        return APP_NOT_FOUND_MESSAGE

    executable = registry[key]
    logger.info("launch_app: launching '%s' → %s", name, executable)

    try:
        _platform_launch(executable)
        return f"Opening {name}."
    except Exception as exc:
        logger.error("launch_app: failed to launch '%s': %s", name, exc)
        return f"I couldn't open {name}. {exc}"


__all__ = [
    "APP_NOT_FOUND_MESSAGE",
    "launch_app",
    "load_app_registry",
]
