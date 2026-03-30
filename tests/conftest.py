"""Pytest configuration for Rex Assistant tests."""

from __future__ import annotations

import os
import sys
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


# Optional: Register custom pytest plugins
# pytest_plugins = ["tests.fixtures.custom_plugin"]


def _tracked_modified_files() -> set[str]:
    from tests.git_helpers import get_dirty_files  # noqa: PLC0415

    return {
        line[3:]
        for line in get_dirty_files(exclude_coverage=False)
        if line[0:2].strip().startswith("M")
    }


@pytest.fixture(scope="session")
def tracked_modifications_baseline() -> set[str]:
    """Tracked files already modified before tests started."""
    return _tracked_modified_files()


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Run legacy asyncio-marked tests through the installed anyio plugin."""
    del config
    for item in items:
        if item.get_closest_marker("asyncio") and not item.get_closest_marker("anyio"):
            item.add_marker(pytest.mark.anyio)
