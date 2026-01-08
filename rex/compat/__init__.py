"""Compatibility shims for third-party library changes.

This package contains compatibility layers that allow the application
to work with newer versions of dependencies that have breaking changes.
"""

from rex.compat.transformers_shims import ensure_transformers_compatibility

__all__ = ["ensure_transformers_compatibility"]
