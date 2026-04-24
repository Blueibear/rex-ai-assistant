"""Pytest subprocess startup compatibility when repo root is on PYTHONPATH."""

from __future__ import annotations

try:
    from tests.python_startup.sitecustomize import _install_asyncio_fallback
except Exception:
    pass
else:
    _install_asyncio_fallback()
