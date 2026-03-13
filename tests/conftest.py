"""Pytest configuration for Rex Assistant tests."""

from __future__ import annotations

import asyncio
import os
import sys
import warnings
from pathlib import Path

import pytest

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

# Signal that tests are running (some modules might check this)
os.environ["REX_TESTING"] = "true"

# Optional: Directory for shared test fixtures
FIXTURES_DIR = ROOT / "tests" / "fixtures"
FIXTURES_DIR.mkdir(parents=True, exist_ok=True)


@pytest.fixture(autouse=True)
def ensure_event_loop_for_sync_tests():
    """Ensure sync tests relying on asyncio.get_event_loop() always have a loop.

    Python 3.12 no longer auto-creates a main-thread loop in all situations,
    and some tests still call ``asyncio.get_event_loop()`` directly.
    """
    created_loop = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        try:
            asyncio.get_event_loop()
        except RuntimeError:
            created_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(created_loop)

    try:
        yield
    finally:
        if created_loop is not None and not created_loop.is_closed():
            created_loop.close()
            asyncio.set_event_loop(None)


# Optional: Register custom pytest plugins
# pytest_plugins = ["tests.fixtures.custom_plugin"]


def _tracked_modified_files() -> set[str]:
    from git_helpers import get_dirty_files  # noqa: PLC0415

    return {
        line[3:]
        for line in get_dirty_files(exclude_coverage=False)
        if line[0:2].strip().startswith("M")
    }


@pytest.fixture(scope="session")
def tracked_modifications_baseline() -> set[str]:
    """Tracked files already modified before tests started."""
    return _tracked_modified_files()
