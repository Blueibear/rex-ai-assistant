"""Smoke tests for rex.windows_service (Windows only, no pywin32 required)."""

from __future__ import annotations

import sys

import pytest

if sys.platform != "win32":
    pytest.skip("rex.windows_service is Windows-only", allow_module_level=True)


def test_import():
    """Module imports without error on Windows."""
    import rex.windows_service as ws

    assert ws is not None


def test_constants():
    """Default constants are present and non-empty."""
    from rex.windows_service import DEFAULT_PORT, DEFAULT_SERVICES

    assert DEFAULT_SERVICES
    assert DEFAULT_PORT


def test_pywin32_flag_is_bool():
    """_PYWIN32_SERVICE_AVAILABLE is a bool."""
    from rex.windows_service import _PYWIN32_SERVICE_AVAILABLE

    assert isinstance(_PYWIN32_SERVICE_AVAILABLE, bool)


def test_service_base_is_type():
    """_ServiceBase is a class (object when pywin32 absent)."""
    from rex.windows_service import _ServiceBase

    assert isinstance(_ServiceBase, type)


def test_service_class_attributes():
    """RexNodeService has the expected Windows service name attributes."""
    from rex.windows_service import RexNodeService

    assert RexNodeService._svc_name_ == "RexLeanNode"
    assert RexNodeService._svc_display_name_ == "Rex Lean Node"
    assert RexNodeService._svc_description_


def test_main_raises_when_pywin32_unavailable(monkeypatch):
    """main() raises ImportError when pywin32 service components are missing."""
    import rex.windows_service as ws

    monkeypatch.setattr(ws, "_PYWIN32_SERVICE_AVAILABLE", False)

    with pytest.raises(ImportError, match="pywin32"):
        ws.main()
