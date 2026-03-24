"""
US-038: Application Launching

Acceptance Criteria:
- applications launch
- execution verified
- failures handled
- Typecheck passes
"""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from rex.app_launcher import (
    AppLauncher,
    AppNotRegisteredError,
    get_app_launcher,
    set_app_launcher,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_singleton() -> None:
    """Isolate the global singleton between tests."""
    set_app_launcher(None)
    yield
    set_app_launcher(None)


def _make_launcher(*registered: tuple[str, str]) -> AppLauncher:
    """Return an AppLauncher pre-loaded with (name, exe) pairs."""
    launcher = AppLauncher()
    for name, exe in registered:
        launcher.register(name, exe)
    return launcher


def _fake_popen(pid: int = 12345) -> MagicMock:
    """Return a mock Popen process with a fixed PID."""
    proc = MagicMock()
    proc.pid = pid
    return proc


# ---------------------------------------------------------------------------
# Registration tests
# ---------------------------------------------------------------------------


def test_register_stores_app() -> None:
    launcher = _make_launcher(("notepad", "notepad.exe"))
    assert launcher.is_registered("notepad")


def test_register_multiple_apps() -> None:
    launcher = _make_launcher(("a", "a.exe"), ("b", "b.exe"), ("c", "c.exe"))
    assert launcher.list_apps() == ["a", "b", "c"]


def test_register_overwrites_existing() -> None:
    launcher = AppLauncher()
    launcher.register("app", "old.exe")
    launcher.register("app", "new.exe")
    assert launcher.list_apps() == ["app"]


def test_register_empty_name_raises() -> None:
    launcher = AppLauncher()
    with pytest.raises(ValueError, match="name"):
        launcher.register("", "something.exe")


def test_register_empty_executable_raises() -> None:
    launcher = AppLauncher()
    with pytest.raises(ValueError, match="[Ee]xecutable"):
        launcher.register("app", "")


def test_unregister_removes_app() -> None:
    launcher = _make_launcher(("notepad", "notepad.exe"))
    removed = launcher.unregister("notepad")
    assert removed is True
    assert not launcher.is_registered("notepad")


def test_unregister_unknown_returns_false() -> None:
    launcher = AppLauncher()
    assert launcher.unregister("ghost") is False


def test_is_registered_false_for_unknown() -> None:
    launcher = AppLauncher()
    assert not launcher.is_registered("ghost")


# ---------------------------------------------------------------------------
# Application launch tests
# ---------------------------------------------------------------------------


def test_launch_unknown_app_raises() -> None:
    launcher = AppLauncher()
    with pytest.raises(AppNotRegisteredError, match="not registered"):
        launcher.launch("notepad")


def test_launch_returns_success_with_pid() -> None:
    launcher = _make_launcher(("notepad", "notepad.exe"))
    fake_proc = _fake_popen(pid=9999)

    with patch("rex.app_launcher.subprocess.Popen", return_value=fake_proc) as mock_popen:
        result = launcher.launch("notepad")

    assert result.success is True
    assert result.pid == 9999
    assert result.app_name == "notepad"
    assert result.error is None
    mock_popen.assert_called_once_with(
        ["notepad.exe"],
        shell=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def test_launch_passes_args() -> None:
    launcher = _make_launcher(("editor", "editor.exe"))
    fake_proc = _fake_popen(pid=1)

    with patch("rex.app_launcher.subprocess.Popen", return_value=fake_proc) as mock_popen:
        launcher.launch("editor", args=["--new-window", "/path/to/file"])

    mock_popen.assert_called_once_with(
        ["editor.exe", "--new-window", "/path/to/file"],
        shell=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


# ---------------------------------------------------------------------------
# Execution verified tests
# ---------------------------------------------------------------------------


def test_pid_is_positive_integer_on_success() -> None:
    launcher = _make_launcher(("app", "app.exe"))
    fake_proc = _fake_popen(pid=42)

    with patch("rex.app_launcher.subprocess.Popen", return_value=fake_proc):
        result = launcher.launch("app")

    assert isinstance(result.pid, int)
    assert result.pid > 0


def test_launch_result_success_flag_true() -> None:
    launcher = _make_launcher(("app", "app.exe"))
    fake_proc = _fake_popen(pid=1)

    with patch("rex.app_launcher.subprocess.Popen", return_value=fake_proc):
        result = launcher.launch("app")

    assert result.success is True


def test_launch_result_contains_app_name() -> None:
    launcher = _make_launcher(("my_app", "my_app.exe"))
    fake_proc = _fake_popen()

    with patch("rex.app_launcher.subprocess.Popen", return_value=fake_proc):
        result = launcher.launch("my_app")

    assert result.app_name == "my_app"


# ---------------------------------------------------------------------------
# Failure handling tests
# ---------------------------------------------------------------------------


def test_launch_file_not_found_returns_failure() -> None:
    launcher = _make_launcher(("bad_app", "nonexistent_binary_xyz"))

    with patch(
        "rex.app_launcher.subprocess.Popen",
        side_effect=FileNotFoundError("not found"),
    ):
        result = launcher.launch("bad_app")

    assert result.success is False
    assert result.pid is None
    assert result.error is not None
    assert "not found" in result.error.lower() or "nonexistent_binary_xyz" in result.error


def test_launch_permission_error_returns_failure() -> None:
    launcher = _make_launcher(("locked_app", "/locked/app"))

    with patch(
        "rex.app_launcher.subprocess.Popen",
        side_effect=PermissionError("access denied"),
    ):
        result = launcher.launch("locked_app")

    assert result.success is False
    assert result.error is not None
    assert "permission" in result.error.lower() or "denied" in result.error.lower()


def test_launch_generic_exception_returns_failure() -> None:
    launcher = _make_launcher(("crash_app", "crash.exe"))

    with patch(
        "rex.app_launcher.subprocess.Popen",
        side_effect=OSError("something went wrong"),
    ):
        result = launcher.launch("crash_app")

    assert result.success is False
    assert result.error is not None


def test_failure_result_has_no_pid() -> None:
    launcher = _make_launcher(("app", "app.exe"))

    with patch(
        "rex.app_launcher.subprocess.Popen",
        side_effect=FileNotFoundError(),
    ):
        result = launcher.launch("app")

    assert result.pid is None


# ---------------------------------------------------------------------------
# Singleton tests
# ---------------------------------------------------------------------------


def test_get_app_launcher_returns_instance() -> None:
    launcher = get_app_launcher()
    assert isinstance(launcher, AppLauncher)


def test_get_app_launcher_returns_same_instance() -> None:
    a = get_app_launcher()
    b = get_app_launcher()
    assert a is b


def test_set_app_launcher_replaces_singleton() -> None:
    custom = AppLauncher(apps={"vim": "vim"})
    set_app_launcher(custom)
    assert get_app_launcher() is custom


def test_set_app_launcher_none_resets_singleton() -> None:
    launcher1 = get_app_launcher()
    set_app_launcher(None)
    launcher2 = get_app_launcher()
    assert launcher1 is not launcher2


# ---------------------------------------------------------------------------
# Pre-populated registry test
# ---------------------------------------------------------------------------


def test_constructor_with_preloaded_apps() -> None:
    launcher = AppLauncher(apps={"firefox": "firefox", "terminal": "xterm"})
    assert launcher.is_registered("firefox")
    assert launcher.is_registered("terminal")
    assert launcher.list_apps() == ["firefox", "terminal"]
