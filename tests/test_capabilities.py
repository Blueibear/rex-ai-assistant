"""Tests for rex.capabilities module (capability registry and loader)."""

from __future__ import annotations

from rex.capabilities import CAPABILITY_REGISTRY, load_capabilities


def test_import():
    """Module imports without error."""
    import rex.capabilities  # noqa: F401


def test_capability_registry_has_expected_keys():
    """CAPABILITY_REGISTRY contains the four built-in capability keys."""
    for key in ("local_commands", "ha_router", "web_search", "plugins"):
        assert key in CAPABILITY_REGISTRY, f"Missing key: {key}"


def test_capability_registry_values_are_callable():
    """Every loader in CAPABILITY_REGISTRY is callable."""
    for name, loader in CAPABILITY_REGISTRY.items():
        assert callable(loader), f"Loader for {name!r} is not callable"


def test_load_capabilities_empty_profile():
    """Returns empty dict when profile has no capabilities list."""
    assert load_capabilities({}, {}) == {}


def test_load_capabilities_with_known_capability():
    """Loads a capability that exists in the registry."""
    profile = {"capabilities": ["local_commands"]}
    result = load_capabilities(profile, {})
    assert "local_commands" in result


def test_load_capabilities_all_known():
    """Loads all four built-in capabilities."""
    profile = {"capabilities": ["local_commands", "ha_router", "web_search", "plugins"]}
    result = load_capabilities(profile, {})
    assert set(result.keys()) == {"local_commands", "ha_router", "web_search", "plugins"}


def test_load_capabilities_ignores_unknown_names():
    """Unknown capability names are silently skipped."""
    profile = {"capabilities": ["nonexistent_capability"]}
    result = load_capabilities(profile, {})
    assert result == {}


def test_load_capabilities_non_dict_profile():
    """Handles non-dict profile gracefully (returns empty dict)."""
    assert load_capabilities(None, {}) == {}
    assert load_capabilities("string", {}) == {}
    assert load_capabilities([], {}) == {}


def test_all_exports():
    """__all__ contains the public API."""
    import rex.capabilities as m

    assert "CAPABILITY_REGISTRY" in m.__all__
    assert "load_capabilities" in m.__all__
