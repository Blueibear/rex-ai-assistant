"""Pytest configuration for Rex Assistant tests."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Resolve root of the project
ROOT = Path(__file__).resolve().parents[1]

# Ensure root path is in sys.path for module imports
root_str = str(ROOT)
if root_str not in sys.path:
    sys.path.insert(0, root_str)

# Signal that tests are running (some modules might check this)
os.environ["REX_TESTING"] = "true"

# Optional: Directory for shared test fixtures
FIXTURES_DIR = ROOT / "tests" / "fixtures"
FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

# Optional: Register custom pytest plugins (if any)
# pytest_plugins = ["tests.fixtures.custom_plugin"]

