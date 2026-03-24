"""
US-072: Process Monitoring

Acceptance Criteria:
- launched processes monitored
- crashes detected
- process restart supported
- Typecheck passes
"""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from rex.process_monitor import (
    ProcessMonitor,
    ProcessNotWatchedError,
    ProcessStatus,
    RestartLimitExceededError,
    get_process_monitor,
    set_process_monitor,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_proc(pid: int = 1000, returncode: int | None = None) -> MagicMock:
    """Return a mock Popen process."""
    proc = MagicMock(spec=subprocess.Popen)
    proc.pid = pid
    proc.poll.return_value = returncode
    return proc


def _running_proc(pid: int = 1000) -> MagicMock:
    return _make_proc(pid=pid, returncode=None)


def _crashed_proc(pid: int = 1000, returncode: int = 1) -> MagicMock:
    return _make_proc(pid=pid, returncode=returncode)


def _clean_exit_proc(pid: int = 1000) -> MagicMock:
    return _make_proc(pid=pid, returncode=0)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_singleton() -> None:
    set_process_monitor(None)
    yield
    set_process_monitor(None)


# ---------------------------------------------------------------------------
# Launched processes monitored
# ---------------------------------------------------------------------------


def test_watch_registers_pid() -> None:
    monitor = ProcessMonitor()
    proc = _running_proc(pid=100)
    monitor.watch(100, "notepad", proc, executable="notepad.exe")
    assert 100 in monitor.list_watched()


def test_watch_multiple_processes() -> None:
    monitor = ProcessMonitor()
    for pid in [101, 102, 103]:
        monitor.watch(pid, f"app{pid}", _running_proc(pid=pid), executable="app.exe")
    watched = set(monitor.list_watched())
    assert {101, 102, 103}.issubset(watched)


def test_unwatch_removes_pid() -> None:
    monitor = ProcessMonitor()
    proc = _running_proc(pid=200)
    monitor.watch(200, "app", proc, executable="app.exe")
    removed = monitor.unwatch(200)
    assert removed is True
    assert 200 not in monitor.list_watched()


def test_unwatch_unknown_returns_false() -> None:
    monitor = ProcessMonitor()
    assert monitor.unwatch(9999) is False


def test_is_running_returns_true_for_live_process() -> None:
    monitor = ProcessMonitor()
    proc = _running_proc(pid=300)
    monitor.watch(300, "app", proc, executable="app.exe")
    assert monitor.is_running(300) is True


def test_is_running_returns_false_for_stopped_process() -> None:
    monitor = ProcessMonitor()
    proc = _clean_exit_proc(pid=301)
    monitor.watch(301, "app", proc, executable="app.exe")
    assert monitor.is_running(301) is False


def test_is_running_raises_for_unwatched_pid() -> None:
    monitor = ProcessMonitor()
    with pytest.raises(ProcessNotWatchedError):
        monitor.is_running(9999)


def test_check_raises_for_unwatched_pid() -> None:
    monitor = ProcessMonitor()
    with pytest.raises(ProcessNotWatchedError):
        monitor.check(9999)


def test_check_all_returns_all_statuses() -> None:
    monitor = ProcessMonitor()
    for pid in [400, 401, 402]:
        monitor.watch(pid, f"app{pid}", _running_proc(pid=pid), executable="app.exe")
    statuses = monitor.check_all()
    assert len(statuses) == 3
    pids = {s.pid for s in statuses}
    assert pids == {400, 401, 402}


def test_check_all_empty_monitor_returns_empty_list() -> None:
    monitor = ProcessMonitor()
    assert monitor.check_all() == []


def test_check_returns_process_status_type() -> None:
    monitor = ProcessMonitor()
    proc = _running_proc(pid=500)
    monitor.watch(500, "app", proc, executable="app.exe")
    status = monitor.check(500)
    assert isinstance(status, ProcessStatus)


def test_check_status_includes_app_name() -> None:
    monitor = ProcessMonitor()
    proc = _running_proc(pid=501)
    monitor.watch(501, "myapp", proc, executable="myapp.exe")
    status = monitor.check(501)
    assert status.app_name == "myapp"


# ---------------------------------------------------------------------------
# Crashes detected
# ---------------------------------------------------------------------------


def test_check_running_process_not_crashed() -> None:
    monitor = ProcessMonitor()
    proc = _running_proc(pid=600)
    monitor.watch(600, "app", proc, executable="app.exe")
    status = monitor.check(600)
    assert status.running is True
    assert status.crashed is False
    assert status.returncode is None


def test_check_crashed_process_has_crashed_true() -> None:
    monitor = ProcessMonitor()
    proc = _crashed_proc(pid=601, returncode=1)
    monitor.watch(601, "app", proc, executable="app.exe")
    status = monitor.check(601)
    assert status.crashed is True
    assert status.running is False
    assert status.returncode == 1


def test_check_nonzero_returncode_means_crashed() -> None:
    monitor = ProcessMonitor()
    for rc in [-1, 2, 255]:
        proc = _make_proc(pid=700 + rc, returncode=rc)
        monitor.watch(700 + rc, "app", proc, executable="app.exe")
        status = monitor.check(700 + rc)
        assert status.crashed is True, f"Expected crashed=True for returncode={rc}"


def test_check_clean_exit_is_not_crashed() -> None:
    monitor = ProcessMonitor()
    proc = _clean_exit_proc(pid=800)
    monitor.watch(800, "app", proc, executable="app.exe")
    status = monitor.check(800)
    assert status.crashed is False
    assert status.running is False
    assert status.returncode == 0


def test_check_all_detects_mixed_statuses() -> None:
    monitor = ProcessMonitor()
    monitor.watch(900, "alive", _running_proc(pid=900), executable="alive.exe")
    monitor.watch(901, "dead", _crashed_proc(pid=901), executable="dead.exe")
    statuses = {s.pid: s for s in monitor.check_all()}
    assert statuses[900].running is True
    assert statuses[901].crashed is True


# ---------------------------------------------------------------------------
# Process restart supported
# ---------------------------------------------------------------------------


def test_restart_replaces_old_pid_with_new_pid() -> None:
    monitor = ProcessMonitor()
    old_proc = _crashed_proc(pid=1001)
    monitor.watch(1001, "app", old_proc, executable="app.exe")

    new_fake_proc = _running_proc(pid=2001)
    with patch("rex.process_monitor.subprocess.Popen", return_value=new_fake_proc):
        result = monitor.restart(1001)

    assert result.success is True
    assert result.old_pid == 1001
    assert result.new_pid == 2001
    assert 1001 not in monitor.list_watched()
    assert 2001 in monitor.list_watched()


def test_restart_increments_restart_count() -> None:
    monitor = ProcessMonitor()
    old_proc = _crashed_proc(pid=1100)
    monitor.watch(1100, "app", old_proc, executable="app.exe")

    new_fake_proc = _running_proc(pid=2100)
    with patch("rex.process_monitor.subprocess.Popen", return_value=new_fake_proc):
        monitor.restart(1100)

    status = monitor.check(2100)
    assert status.restart_count == 1


def test_restart_new_process_is_running() -> None:
    monitor = ProcessMonitor()
    old_proc = _crashed_proc(pid=1200)
    monitor.watch(1200, "app", old_proc, executable="app.exe")

    new_fake_proc = _running_proc(pid=2200)
    with patch("rex.process_monitor.subprocess.Popen", return_value=new_fake_proc):
        result = monitor.restart(1200)

    assert monitor.is_running(result.new_pid) is True  # type: ignore[arg-type]


def test_restart_raises_for_unwatched_pid() -> None:
    monitor = ProcessMonitor()
    with pytest.raises(ProcessNotWatchedError):
        monitor.restart(9999)


def test_restart_raises_when_limit_exceeded() -> None:
    monitor = ProcessMonitor()
    proc = _crashed_proc(pid=1300)
    monitor.watch(1300, "app", proc, executable="app.exe", max_restarts=2)
    # Simulate already at max
    monitor._records[1300].restart_count = 2
    with pytest.raises(RestartLimitExceededError):
        monitor.restart(1300)


def test_restart_fails_without_executable() -> None:
    monitor = ProcessMonitor()
    proc = _crashed_proc(pid=1400)
    monitor.watch(1400, "app", proc)  # no executable
    result = monitor.restart(1400)
    assert result.success is False
    assert result.error is not None


def test_restart_handles_popen_failure() -> None:
    monitor = ProcessMonitor()
    proc = _crashed_proc(pid=1500)
    monitor.watch(1500, "app", proc, executable="app.exe")
    with patch(
        "rex.process_monitor.subprocess.Popen",
        side_effect=FileNotFoundError("not found"),
    ):
        result = monitor.restart(1500)
    assert result.success is False
    assert result.error is not None
    assert "not found" in result.error


def test_restart_result_contains_app_name() -> None:
    monitor = ProcessMonitor()
    proc = _crashed_proc(pid=1600)
    monitor.watch(1600, "myapp", proc, executable="myapp.exe")
    new_fake_proc = _running_proc(pid=2600)
    with patch("rex.process_monitor.subprocess.Popen", return_value=new_fake_proc):
        result = monitor.restart(1600)
    assert result.app_name == "myapp"


def test_multiple_restarts_accumulate_count() -> None:
    monitor = ProcessMonitor()
    proc = _crashed_proc(pid=1700)
    monitor.watch(1700, "app", proc, executable="app.exe", max_restarts=5)

    current_pid = 1700
    for i in range(1, 4):
        new_pid = 2700 + i
        new_proc = _running_proc(pid=new_pid)
        with patch("rex.process_monitor.subprocess.Popen", return_value=new_proc):
            monitor.restart(current_pid)
        current_pid = new_pid
        # Mark as crashed for the next iteration
        monitor._records[current_pid].process = _crashed_proc(pid=current_pid)

    status = monitor.check(current_pid)
    assert status.restart_count == 3


# ---------------------------------------------------------------------------
# Singleton tests
# ---------------------------------------------------------------------------


def test_get_process_monitor_returns_instance() -> None:
    monitor = get_process_monitor()
    assert isinstance(monitor, ProcessMonitor)


def test_get_process_monitor_returns_same_instance() -> None:
    a = get_process_monitor()
    b = get_process_monitor()
    assert a is b


def test_set_process_monitor_replaces_singleton() -> None:
    custom = ProcessMonitor()
    set_process_monitor(custom)
    assert get_process_monitor() is custom


def test_set_process_monitor_none_resets_singleton() -> None:
    m1 = get_process_monitor()
    set_process_monitor(None)
    m2 = get_process_monitor()
    assert m1 is not m2
