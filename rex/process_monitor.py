"""Process monitor for OS-level launched applications.

Tracks launched processes, detects crashes, and supports restart.

Usage::

    import subprocess
    from rex.process_monitor import ProcessMonitor

    monitor = ProcessMonitor()
    proc = subprocess.Popen(["notepad.exe"], shell=False,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    monitor.watch(proc.pid, "notepad", proc, executable="notepad.exe")

    status = monitor.check(proc.pid)
    if status.crashed:
        restart_result = monitor.restart(proc.pid)
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ProcessRecord:
    """Internal record of a watched process."""

    pid: int
    app_name: str
    executable: str
    process: subprocess.Popen[bytes]
    auto_restart: bool = False
    restart_count: int = 0
    max_restarts: int = 5


@dataclass
class ProcessStatus:
    """Status snapshot for a watched process."""

    pid: int
    app_name: str
    running: bool
    crashed: bool
    returncode: Optional[int] = None
    restart_count: int = 0


@dataclass
class RestartResult:
    """Result of a restart attempt."""

    success: bool
    app_name: str
    old_pid: int
    new_pid: Optional[int] = None
    error: Optional[str] = None


class ProcessNotWatchedError(Exception):
    """Raised when a PID is not registered with the monitor."""


class RestartLimitExceededError(Exception):
    """Raised when a process has exceeded its maximum restart count."""


class ProcessMonitor:
    """Monitor launched processes for crashes and restart them on failure.

    Processes must be registered with :meth:`watch` after launch.  Call
    :meth:`check` or :meth:`check_all` to detect crashes.  Call
    :meth:`restart` to re-launch a crashed process.
    """

    def __init__(self) -> None:
        self._records: dict[int, ProcessRecord] = {}

    # ------------------------------------------------------------------
    # Watch / unwatch
    # ------------------------------------------------------------------

    def watch(
        self,
        pid: int,
        app_name: str,
        process: subprocess.Popen[bytes],
        *,
        executable: str = "",
        auto_restart: bool = False,
        max_restarts: int = 5,
    ) -> None:
        """Register a process for monitoring.

        Args:
            pid: PID of the running process.
            app_name: Logical name for the application.
            process: The :class:`subprocess.Popen` handle.
            executable: Executable path used to restart the process.
            auto_restart: Whether to restart the process automatically on crash.
            max_restarts: Maximum number of automatic restart attempts.
        """
        self._records[pid] = ProcessRecord(
            pid=pid,
            app_name=app_name,
            executable=executable,
            process=process,
            auto_restart=auto_restart,
            max_restarts=max_restarts,
        )
        logger.debug("Watching process %r (pid=%d)", app_name, pid)

    def unwatch(self, pid: int) -> bool:
        """Stop monitoring a process.

        Args:
            pid: PID to remove.

        Returns:
            ``True`` if the process was watched and removed, ``False`` otherwise.
        """
        if pid in self._records:
            del self._records[pid]
            logger.debug("Stopped watching pid=%d", pid)
            return True
        return False

    def list_watched(self) -> list[int]:
        """Return PIDs of all currently watched processes."""
        return list(self._records.keys())

    # ------------------------------------------------------------------
    # Status checks
    # ------------------------------------------------------------------

    def is_running(self, pid: int) -> bool:
        """Return ``True`` if the process with the given PID is still running.

        Args:
            pid: PID to check.

        Returns:
            ``True`` if the process is alive (poll() returns None).

        Raises:
            ProcessNotWatchedError: If *pid* is not registered.
        """
        record = self._records.get(pid)
        if record is None:
            raise ProcessNotWatchedError(
                f"PID {pid} is not watched. Register it with watch() first."
            )
        return record.process.poll() is None

    def check(self, pid: int) -> ProcessStatus:
        """Return the current status of a watched process.

        A process is considered *crashed* when it has exited with a non-zero
        return code (``returncode != 0`` and ``returncode is not None``).

        Args:
            pid: PID to check.

        Returns:
            :class:`ProcessStatus` snapshot.

        Raises:
            ProcessNotWatchedError: If *pid* is not registered.
        """
        record = self._records.get(pid)
        if record is None:
            raise ProcessNotWatchedError(
                f"PID {pid} is not watched. Register it with watch() first."
            )
        returncode = record.process.poll()
        running = returncode is None
        crashed = not running and returncode != 0
        return ProcessStatus(
            pid=pid,
            app_name=record.app_name,
            running=running,
            crashed=crashed,
            returncode=returncode,
            restart_count=record.restart_count,
        )

    def check_all(self) -> list[ProcessStatus]:
        """Return status snapshots for all watched processes.

        Returns:
            List of :class:`ProcessStatus` objects.
        """
        return [self.check(pid) for pid in list(self._records.keys())]

    # ------------------------------------------------------------------
    # Restart
    # ------------------------------------------------------------------

    def restart(self, pid: int) -> RestartResult:
        """Restart a watched (usually crashed) process.

        The process is re-launched using the executable stored in the process
        record.  The old PID entry is replaced with the new one.

        Args:
            pid: PID of the process to restart.

        Returns:
            :class:`RestartResult`.

        Raises:
            ProcessNotWatchedError: If *pid* is not registered.
            RestartLimitExceededError: If the process has reached *max_restarts*.
        """
        record = self._records.get(pid)
        if record is None:
            raise ProcessNotWatchedError(
                f"PID {pid} is not watched. Register it with watch() first."
            )

        if record.restart_count >= record.max_restarts:
            raise RestartLimitExceededError(
                f"Process {record.app_name!r} (pid={pid}) has exceeded "
                f"its maximum restart count of {record.max_restarts}."
            )

        if not record.executable:
            error = (
                f"Cannot restart {record.app_name!r}: no executable path stored. "
                "Pass executable=<path> when calling watch()."
            )
            logger.warning(error)
            return RestartResult(
                success=False,
                app_name=record.app_name,
                old_pid=pid,
                error=error,
            )

        try:
            new_proc = subprocess.Popen(  # noqa: S603
                [record.executable],
                shell=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as exc:  # noqa: BLE001
            error = f"Re-launch of {record.executable!r} failed: {exc}"
            logger.warning("Restart of %r failed: %s", record.app_name, error)
            return RestartResult(
                success=False,
                app_name=record.app_name,
                old_pid=pid,
                error=error,
            )

        new_pid = new_proc.pid
        new_restart_count = record.restart_count + 1

        logger.info(
            "Restarted %r: old pid=%d -> new pid=%d (attempt %d/%d)",
            record.app_name,
            pid,
            new_pid,
            new_restart_count,
            record.max_restarts,
        )

        # Replace old record with new one
        del self._records[pid]
        self._records[new_pid] = ProcessRecord(
            pid=new_pid,
            app_name=record.app_name,
            executable=record.executable,
            process=new_proc,
            auto_restart=record.auto_restart,
            restart_count=new_restart_count,
            max_restarts=record.max_restarts,
        )

        return RestartResult(
            success=True,
            app_name=record.app_name,
            old_pid=pid,
            new_pid=new_pid,
        )


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_monitor: Optional[ProcessMonitor] = None


def get_process_monitor() -> ProcessMonitor:
    """Return the module-level :class:`ProcessMonitor` singleton."""
    global _monitor  # noqa: PLW0603
    if _monitor is None:
        _monitor = ProcessMonitor()
    return _monitor


def set_process_monitor(monitor: Optional[ProcessMonitor]) -> None:
    """Replace the module-level singleton (for testing)."""
    global _monitor  # noqa: PLW0603
    _monitor = monitor


__all__ = [
    "ProcessMonitor",
    "ProcessNotWatchedError",
    "ProcessRecord",
    "ProcessStatus",
    "RestartLimitExceededError",
    "RestartResult",
    "get_process_monitor",
    "set_process_monitor",
]
