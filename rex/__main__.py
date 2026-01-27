"""Command-line entry point for the Rex AI Assistant package.

This module allows running Rex as a Python module:
    python -m rex
    python -m rex doctor
    python -m rex chat
    python -m rex version

It also enables installation as a console script via pyproject.toml.
"""

from __future__ import annotations

import sys

# Load .env before accessing any environment variables
try:
    from utils.env_loader import load as _load_env

    _load_env()
except ImportError:
    # utils may not be available if running from installed package
    pass

from rex.cli import main

if __name__ == "__main__":
    sys.exit(main())
