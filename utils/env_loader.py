"""Environment variable loader for Rex AI Assistant.

This module ensures .env files are loaded before any environment variables
are accessed, preventing initialization issues.

The .env file is automatically loaded when this module is first imported.
"""

from __future__ import annotations

from pathlib import Path

try:
    from dotenv import find_dotenv, load_dotenv
except ImportError:
    # Gracefully handle missing python-dotenv
    def load_dotenv(*args, **kwargs):  # type: ignore[misc]
        return False

    def find_dotenv(*args, **kwargs):  # type: ignore[misc]
        return ""


_loaded = False


def load() -> None:
    """Load environment variables from .env file.

    Uses find_dotenv(usecwd=True) first, then falls back to <repo_root>/.env.
    Only loads once per process. Uses override=False to respect existing env vars.
    """
    global _loaded
    if _loaded:
        return

    # Try to find .env using find_dotenv with usecwd=True
    dotenv_path = find_dotenv(usecwd=True)

    # Fallback to <repo_root>/.env if not found
    if not dotenv_path:
        # Determine repository root (parent of utils/ directory)
        repo_root = Path(__file__).resolve().parent.parent
        fallback_path = repo_root / ".env"
        if fallback_path.exists():
            dotenv_path = str(fallback_path)

    # Load the .env file if found
    if dotenv_path:
        load_dotenv(dotenv_path, override=False)

    _loaded = True


# Automatically load .env on module import
load()
