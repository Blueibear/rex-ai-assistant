"""Pytest configuration for Rex Assistant tests."""

from __future__ import annotations

import sys
import os
from pathlib import Path

# Resolve root of the project
ROOT = Path(__file__).resolve().parents[1]

# Ensure root path is in sys.path for module imports
root_str = str(ROOT)
if root_str not in sys.path:
    sys.path.insert(0, root_str)

# Optional: signal to the app that tests are running
os.environ["REX_TESTING"] = "true"

# Optional: directory for test fixtures
FIXTURES_DIR = ROOT / "tests" / "fixtures"
os.makedirs(FIXTURES_DIR, exist_ok=True)

# Optional future usage: dynamically load pytest plugins
# pytest_plugins = ["tests.fixtures.some_plugin"]
