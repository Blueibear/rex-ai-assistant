"""Application launcher for OS-level app launching.

Provides a safe, registry-backed mechanism for launching desktop applications
by name. Applications must be explicitly registered before they can be launched.

Usage::

    launcher = AppLauncher()
    launcher.register("notepad", "notepad.exe")
    result = launcher.launch("notepad")
    print(result.pid)
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


class AppNotRegisteredError(Exception):
    """Raised when an app name is not in the registry."""


class AppLaunchError(Exception):
    """Raised when an application fails to launch."""


@dataclass
class LaunchResult:
    """Result of an application launch attempt."""

    success: bool
    app_name: str
    pid: Optional[int] = None
    error: Optional[str] = None


class AppLauncher:
    """Registry-backed application launcher.

    Applications must be registered by name before they can be launched.
    Launching is non-blocking (uses ``subprocess.Popen``).

    Args:
        apps: Optional pre-populated mapping of name -> executable path.
    """

    def __init__(self, apps: Optional[dict[str, str]] = None) -> None:
        self._registry: dict[str, str] = dict(apps or {})

    # ------------------------------------------------------------------
    # Registry management
    # ------------------------------------------------------------------

    def register(self, name: str, executable: str) -> None:
        """Register an application by name.

        Args:
            name: Logical name for the application (e.g. ``"notepad"``).
            executable: Path or command name to execute (e.g. ``"notepad.exe"``).

        Raises:
            ValueError: If name or executable is empty.
        """
        if not name or not name.strip():
            raise ValueError("App name must not be empty")
        if not executable or not executable.strip():
            raise ValueError("Executable must not be empty")
        self._registry[name.strip()] = executable.strip()
        logger.debug("Registered app %r -> %r", name, executable)

    def unregister(self, name: str) -> bool:
        """Remove an application from the registry.

        Args:
            name: Logical name to remove.

        Returns:
            ``True`` if the app was registered and removed, ``False`` otherwise.
        """
        if name in self._registry:
            del self._registry[name]
            logger.debug("Unregistered app %r", name)
            return True
        return False

    def list_apps(self) -> list[str]:
        """Return the names of all registered applications."""
        return sorted(self._registry.keys())

    def is_registered(self, name: str) -> bool:
        """Return ``True`` if *name* is a registered application."""
        return name in self._registry

    # ------------------------------------------------------------------
    # Launch
    # ------------------------------------------------------------------

    def launch(self, name: str, args: Optional[list[str]] = None) -> LaunchResult:
        """Launch a registered application.

        The launch is non-blocking. The returned :class:`LaunchResult` contains
        the PID of the launched process.

        Args:
            name: Logical name of the application to launch.
            args: Optional list of arguments to pass to the executable.

        Returns:
            :class:`LaunchResult` with ``success=True`` and ``pid`` set on
            success, or ``success=False`` and ``error`` set on failure.

        Raises:
            AppNotRegisteredError: If *name* is not in the registry.
        """
        if name not in self._registry:
            raise AppNotRegisteredError(
                f"App {name!r} is not registered. "
                f"Register it first with launcher.register(name, executable). "
                f"Known apps: {sorted(self._registry.keys())}"
            )

        executable = self._registry[name]
        argv = [executable] + (args or [])

        try:
            proc = subprocess.Popen(  # noqa: S603
                argv,
                shell=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            pid = proc.pid
            logger.info("Launched app %r (pid=%d)", name, pid)
            return LaunchResult(success=True, app_name=name, pid=pid)
        except FileNotFoundError as exc:
            error = f"Executable not found: {executable!r}"
            logger.warning("Failed to launch %r: %s", name, error)
            return LaunchResult(success=False, app_name=name, error=error)
        except PermissionError as exc:
            error = f"Permission denied launching {executable!r}: {exc}"
            logger.warning("Failed to launch %r: %s", name, error)
            return LaunchResult(success=False, app_name=name, error=error)
        except Exception as exc:  # noqa: BLE001
            error = f"Launch failed: {exc}"
            logger.warning("Failed to launch %r: %s", name, error)
            return LaunchResult(success=False, app_name=name, error=error)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_launcher: Optional[AppLauncher] = None


def get_app_launcher() -> AppLauncher:
    """Return the module-level :class:`AppLauncher` singleton."""
    global _launcher  # noqa: PLW0603
    if _launcher is None:
        _launcher = AppLauncher()
    return _launcher


def set_app_launcher(launcher: Optional[AppLauncher]) -> None:
    """Replace the module-level singleton (for testing)."""
    global _launcher  # noqa: PLW0603
    _launcher = launcher


__all__ = [
    "AppLaunchError",
    "AppLauncher",
    "AppNotRegisteredError",
    "LaunchResult",
    "get_app_launcher",
    "set_app_launcher",
]
