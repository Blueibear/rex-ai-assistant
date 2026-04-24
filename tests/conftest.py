"""Pytest configuration for Rex Assistant tests."""

from __future__ import annotations

import os
import socket
import sys
from pathlib import Path

import pytest

# rex.cli raises SystemExit on Python versions other than 3.11 (its supported
# runtime).  Any test module that imports rex.cli at module level will cause
# pytest INTERNALERROR during collection.  We detect this here once and
# gracefully skip those files so the rest of the suite can run.
try:
    import rex.cli  # noqa: F401

    _REX_CLI_AVAILABLE = True
except SystemExit:
    _REX_CLI_AVAILABLE = False

# Scan for test files that import rex.cli at module scope so we can skip
# collection when the CLI Python-version guard fires.
_TESTS_DIR = Path(__file__).resolve().parent


def _find_cli_dependent_tests() -> list[str]:
    """Return absolute paths of test files that import rex.cli at module scope."""
    results: list[str] = []
    for path in _TESTS_DIR.glob("test_*.py"):
        if path.name == "conftest.py":
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if "from rex.cli import" in text or "import rex.cli" in text:
            results.append(str(path))
    return results


try:
    import numpy  # noqa: F401

    _NUMPY_AVAILABLE = True
except ImportError:
    _NUMPY_AVAILABLE = False

try:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM):
        _SOCKETS_AVAILABLE = True
except OSError:
    _SOCKETS_AVAILABLE = False

# Numpy-dependent test files (ML/audio pipeline tests).
_NUMPY_TEST_GLOBS = [
    "test_us020_full_voice_loop.py",
    "test_us138_voice_roundtrip.py",
    "test_voice_enrollment.py",
    "test_voice_enrollment_ui_service.py",
    "test_voice_identifier.py",
    "test_wakeword_model_selection.py",
    "test_ww002_wakeword_train.py",
]

_SOCKET_TEST_GLOBS = [
    "test_computers.py",
    "test_service_supervisor.py",
]

_ignored: list[str] = []
if not _REX_CLI_AVAILABLE:
    _ignored.extend(_find_cli_dependent_tests())
if not _NUMPY_AVAILABLE:
    _ignored.extend(str(_TESTS_DIR / name) for name in _NUMPY_TEST_GLOBS)
if not _SOCKETS_AVAILABLE:
    _ignored.extend(str(_TESTS_DIR / name) for name in _SOCKET_TEST_GLOBS)

collect_ignore: list[str] = _ignored

# ---------------------------------------------------------------------------
# Async test support detection
# ---------------------------------------------------------------------------
try:
    import anyio  # noqa: F401

    _ASYNC_RUNNER = "anyio"
except ImportError:
    try:
        import pytest_asyncio  # noqa: F401

        _ASYNC_RUNNER = "asyncio"
    except ImportError:
        _ASYNC_RUNNER = None

# Resolve root of the project
ROOT = Path(__file__).resolve().parents[1]

# Ensure root path is in sys.path for module imports
root_str = str(ROOT)
if root_str not in sys.path:
    sys.path.insert(0, root_str)

# Ensure tests directory is in sys.path so test helpers are directly importable
tests_str = str(ROOT / "tests")
if tests_str not in sys.path:
    sys.path.insert(0, tests_str)

# Ensure subprocesses spawned by tests receive the same startup compatibility
# shims as the parent pytest process.
startup_str = str(ROOT / "tests" / "python_startup")
if startup_str not in sys.path:
    sys.path.insert(0, startup_str)

pythonpath_parts = [startup_str]
existing_pythonpath = os.environ.get("PYTHONPATH")
if existing_pythonpath:
    pythonpath_parts.extend(
        part for part in existing_pythonpath.split(os.pathsep) if part and part != startup_str
    )
os.environ["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)

try:
    from tests.python_startup.sitecustomize import _install_ssl_fallback
except Exception:
    pass
else:
    _install_ssl_fallback()

# Signal that tests are running (some modules might check this)
os.environ["REX_TESTING"] = "true"

# Optional: Directory for shared test fixtures
FIXTURES_DIR = ROOT / "tests" / "fixtures"
FIXTURES_DIR.mkdir(parents=True, exist_ok=True)


# Optional: Register custom pytest plugins
# pytest_plugins = ["tests.fixtures.custom_plugin"]


def _tracked_modified_files() -> set[str]:
    from tests.git_helpers import get_dirty_files  # noqa: PLC0415

    return {
        line[3:]
        for line in get_dirty_files(exclude_coverage=False)
        if line[0:2].strip()
    }


@pytest.fixture(scope="session")
def tracked_modifications_baseline() -> set[str]:
    """Tracked files already modified before tests started."""
    return _tracked_modified_files()


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Map async test markers to the available runner, or skip if none installed."""
    del config
    import inspect

    skip_no_async = pytest.mark.skip(
        reason="No async test runner installed (anyio or pytest-asyncio required)"
    )

    for item in items:
        is_async = inspect.iscoroutinefunction(getattr(item, "function", None))
        has_asyncio = bool(item.get_closest_marker("asyncio"))

        if is_async or has_asyncio:
            if _ASYNC_RUNNER is None:
                item.add_marker(skip_no_async)
            elif _ASYNC_RUNNER == "anyio" and not item.get_closest_marker("anyio"):
                item.add_marker(pytest.mark.anyio)
