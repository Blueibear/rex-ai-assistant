"""US-018: Bridge path resolution tests.

Parses the BRIDGE_REGISTRY from bridgeResolver.ts and asserts every referenced
bridge script exists in the bridge/ source directory.
"""

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
BRIDGE_RESOLVER_TS = REPO_ROOT / "gui" / "src" / "main" / "bridgeResolver.ts"
BRIDGE_DIR = REPO_ROOT / "bridge"

# Parse the registry at collection time so individual scripts are parametrized.
_ts_source: str = (
    BRIDGE_RESOLVER_TS.read_text(encoding="utf-8") if BRIDGE_RESOLVER_TS.is_file() else ""
)
# Matches value entries like:   rex_tasks_bridge: 'rex_tasks_bridge.py',
_REGISTRY_SCRIPTS: list[str] = re.findall(r":\s*'(rex_\w+\.py)'", _ts_source)


def test_bridge_resolver_ts_exists() -> None:
    assert BRIDGE_RESOLVER_TS.is_file(), f"bridgeResolver.ts not found at {BRIDGE_RESOLVER_TS}"


def test_bridge_dir_exists() -> None:
    assert BRIDGE_DIR.is_dir(), f"bridge/ directory not found at {BRIDGE_DIR}"


def test_registry_is_nonempty() -> None:
    assert (
        _REGISTRY_SCRIPTS
    ), "No bridge scripts parsed from BRIDGE_REGISTRY — check regex or file path."


@pytest.mark.parametrize("script", _REGISTRY_SCRIPTS)
def test_registry_script_exists_in_bridge_dir(script: str) -> None:
    """Each script listed in BRIDGE_REGISTRY must exist under bridge/."""
    assert (BRIDGE_DIR / script).is_file(), f"Missing in bridge/: {script}"
