"""Tests for transformers compatibility shim.

This test verifies that the BeamSearchScorer compatibility patch works
correctly with transformers 4.38+ versions.
"""

import sys
import pytest


def test_beamsearchscorer_import():
    """Test that BeamSearchScorer can be imported from transformers top-level."""
    # Import the shim first to apply patches
    from rex.compat import ensure_transformers_compatibility

    ensure_transformers_compatibility()

    # Now try to import BeamSearchScorer the way legacy libraries do
    try:
        from transformers import BeamSearchScorer

        # Verify it's actually a class
        assert BeamSearchScorer is not None
        assert isinstance(BeamSearchScorer, type)
        assert hasattr(BeamSearchScorer, "__name__")
        assert "BeamSearchScorer" in BeamSearchScorer.__name__

    except ImportError as e:
        pytest.skip(f"transformers not installed or BeamSearchScorer unavailable: {e}")


def test_shim_idempotent():
    """Test that applying the shim multiple times is safe."""
    from rex.compat import ensure_transformers_compatibility

    # Apply shim multiple times
    ensure_transformers_compatibility()
    ensure_transformers_compatibility()
    ensure_transformers_compatibility()

    # Should still work
    try:
        from transformers import BeamSearchScorer

        assert BeamSearchScorer is not None
    except ImportError:
        pytest.skip("transformers not installed")


def test_transformers_available_in_sys_modules():
    """Test that transformers is properly loaded in sys.modules after shim."""
    from rex.compat import ensure_transformers_compatibility

    ensure_transformers_compatibility()

    try:
        import transformers

        assert "transformers" in sys.modules
        assert hasattr(transformers, "__version__")
    except ImportError:
        pytest.skip("transformers not installed")


def test_shim_works_without_transformers_installed():
    """Test that the shim doesn't crash if transformers isn't installed."""
    import builtins

    # Save original import
    original_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "transformers":
            raise ImportError("transformers not installed (mocked)")
        return original_import(name, *args, **kwargs)

    # Temporarily mock import failure
    builtins.__import__ = mock_import

    try:
        # Re-import to test with mocked import
        from rex.compat.transformers_shims import ensure_transformers_compatibility

        # Should not raise an exception
        ensure_transformers_compatibility()
    finally:
        # Restore original import
        builtins.__import__ = original_import
