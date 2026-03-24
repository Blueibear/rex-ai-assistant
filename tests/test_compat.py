"""Tests for rex.compat module (transformers compatibility shim)."""

from __future__ import annotations

import importlib.util
import sys
from unittest.mock import MagicMock, patch


def test_import():
    """Module imports without error."""
    import rex.compat  # noqa: F401


def test_ensure_transformers_compatibility_callable():
    """ensure_transformers_compatibility is a callable function."""
    from rex.compat import ensure_transformers_compatibility

    assert callable(ensure_transformers_compatibility)


def test_ensure_transformers_compatibility_no_error_without_transformers():
    """Function runs without error when transformers is not installed."""
    from rex.compat import ensure_transformers_compatibility

    # Simulate transformers not being installed.
    with patch.dict(sys.modules, {"transformers": None}):
        ensure_transformers_compatibility()  # must not raise


def test_ensure_transformers_compatibility_when_already_patched():
    """Function returns early when BeamSearchScorer already present on transformers."""
    from rex.compat import ensure_transformers_compatibility

    mock_transformers = MagicMock()
    mock_transformers.BeamSearchScorer = MagicMock()  # already present

    # Patch find_spec to report transformers as installed, then put the mock in
    # sys.modules so the subsequent `import transformers` statement resolves it.
    with patch.object(importlib.util, "find_spec", return_value=MagicMock()):
        with patch.dict(sys.modules, {"transformers": mock_transformers}):
            ensure_transformers_compatibility()  # should early-return, no error


def test_all_exports():
    """__all__ contains the public API."""
    import rex.compat as m

    assert "ensure_transformers_compatibility" in m.__all__
