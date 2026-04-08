"""Tests for rex.capabilities.registry (US-037)."""

from __future__ import annotations

import pytest

from rex.capabilities.registry import (
    Capability,
    CapabilityRegistry,
    get_capability_registry,
    populate_from_config,
    reset_capability_registry,
)

# ---------------------------------------------------------------------------
# Capability dataclass
# ---------------------------------------------------------------------------


def test_capability_creation():
    cap = Capability(
        name="test_cap",
        description="A test capability",
        inputs=["query"],
        outputs=["result"],
        triggers=["test"],
    )
    assert cap.name == "test_cap"
    assert cap.description == "A test capability"
    assert cap.inputs == ["query"]
    assert cap.outputs == ["result"]
    assert cap.triggers == ["test"]
    assert cap.enabled is True
    assert cap.category == "General"


def test_capability_empty_name_raises():
    with pytest.raises(ValueError, match="name"):
        Capability(name="", description="desc")


def test_capability_empty_description_raises():
    with pytest.raises(ValueError, match="description"):
        Capability(name="cap", description="")


def test_capability_defaults():
    cap = Capability(name="cap", description="desc")
    assert cap.inputs == []
    assert cap.outputs == []
    assert cap.triggers == []
    assert cap.enabled is True
    assert cap.category == "General"


# ---------------------------------------------------------------------------
# CapabilityRegistry
# ---------------------------------------------------------------------------


def _make_registry() -> CapabilityRegistry:
    registry = CapabilityRegistry()
    registry.register(
        Capability(
            name="chat",
            description="Chat with assistant",
            triggers=["chat", "talk"],
            category="General",
        )
    )
    registry.register(
        Capability(
            name="web_search",
            description="Search the web",
            triggers=["search", "find"],
            category="Search",
            enabled=True,
        )
    )
    registry.register(
        Capability(
            name="home_assistant",
            description="Control home devices",
            triggers=["turn on", "lights"],
            category="Home",
            enabled=False,
        )
    )
    return registry


def test_registry_list_returns_enabled_only():
    registry = _make_registry()
    caps = registry.list()
    names = [c.name for c in caps]
    assert "home_assistant" not in names
    assert "chat" in names
    assert "web_search" in names


def test_registry_list_include_disabled():
    registry = _make_registry()
    caps = registry.list(include_disabled=True)
    names = [c.name for c in caps]
    assert "home_assistant" in names


def test_registry_list_sorted_by_name():
    registry = _make_registry()
    caps = registry.list(include_disabled=True)
    names = [c.name for c in caps]
    assert names == sorted(names)


def test_registry_search_by_name():
    registry = _make_registry()
    results = registry.search("chat")
    assert len(results) == 1
    assert results[0].name == "chat"


def test_registry_search_by_description():
    registry = _make_registry()
    results = registry.search("web")
    assert any(c.name == "web_search" for c in results)


def test_registry_search_by_trigger():
    registry = _make_registry()
    results = registry.search("search")
    names = [c.name for c in results]
    assert "web_search" in names


def test_registry_search_case_insensitive():
    registry = _make_registry()
    results = registry.search("CHAT")
    assert any(c.name == "chat" for c in results)


def test_registry_search_excludes_disabled():
    registry = _make_registry()
    results = registry.search("home")
    assert all(c.name != "home_assistant" for c in results)


def test_registry_search_no_match():
    registry = _make_registry()
    results = registry.search("zzznomatch")
    assert results == []


def test_registry_unregister():
    registry = _make_registry()
    assert registry.unregister("chat") is True
    assert registry.get("chat") is None


def test_registry_unregister_nonexistent():
    registry = CapabilityRegistry()
    assert registry.unregister("nothing") is False


def test_registry_get():
    registry = _make_registry()
    cap = registry.get("chat")
    assert cap is not None
    assert cap.name == "chat"


def test_registry_get_missing():
    registry = CapabilityRegistry()
    assert registry.get("missing") is None


# ---------------------------------------------------------------------------
# populate_from_config
# ---------------------------------------------------------------------------


class _FakeConfig:
    def __init__(self, **kwargs: object) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)


def test_populate_enables_weather():
    reset_capability_registry()
    registry = get_capability_registry()
    cap = registry.get("weather_now")
    assert cap is not None
    assert cap.enabled is False  # not configured yet

    cfg = _FakeConfig(openweathermap_api_key="abc123")
    populate_from_config(registry, cfg)
    assert registry.get("weather_now").enabled is True


def test_populate_enables_home_assistant():
    reset_capability_registry()
    registry = get_capability_registry()
    cfg = _FakeConfig(ha_token="token123", ha_base_url=None)
    populate_from_config(registry, cfg)
    assert registry.get("home_assistant").enabled is True


def test_populate_enables_home_assistant_via_url():
    reset_capability_registry()
    registry = get_capability_registry()
    cfg = _FakeConfig(ha_token=None, ha_base_url="http://homeassistant.local:8123")
    populate_from_config(registry, cfg)
    assert registry.get("home_assistant").enabled is True


def test_populate_enables_music_assistant():
    reset_capability_registry()
    registry = get_capability_registry()
    cfg = _FakeConfig(music_assistant_url="http://ma.local:8095")
    populate_from_config(registry, cfg)
    assert registry.get("music_assistant").enabled is True


def test_populate_enables_email():
    reset_capability_registry()
    registry = get_capability_registry()
    cfg = _FakeConfig(email_provider="gmail", email_accounts=[])
    populate_from_config(registry, cfg)
    assert registry.get("send_email").enabled is True


def test_populate_no_config_leaves_disabled():
    reset_capability_registry()
    registry = get_capability_registry()
    cfg = _FakeConfig(
        openweathermap_api_key=None,
        brave_api_key=None,
        ha_token=None,
        ha_base_url=None,
        email_provider="none",
        email_accounts=[],
        music_assistant_url=None,
        search_providers="duckduckgo",
    )
    populate_from_config(registry, cfg)
    # web_search stays enabled because duckduckgo is a default fallback
    weather = registry.get("weather_now")
    assert weather is not None
    assert weather.enabled is False


# ---------------------------------------------------------------------------
# Global singleton
# ---------------------------------------------------------------------------


def test_get_capability_registry_singleton():
    reset_capability_registry()
    r1 = get_capability_registry()
    r2 = get_capability_registry()
    assert r1 is r2


def test_get_capability_registry_populates_on_config():
    reset_capability_registry()
    cfg = _FakeConfig(openweathermap_api_key="key")
    registry = get_capability_registry(config=cfg)
    assert registry.get("weather_now").enabled is True


def test_reset_capability_registry():
    r1 = get_capability_registry()
    reset_capability_registry()
    r2 = get_capability_registry()
    assert r1 is not r2


# ---------------------------------------------------------------------------
# Built-in capabilities present at startup
# ---------------------------------------------------------------------------


def test_builtin_capabilities_present():
    reset_capability_registry()
    registry = get_capability_registry()
    names = [c.name for c in registry.list(include_disabled=True)]
    for expected in ("chat", "time_now", "weather_now", "web_search", "home_assistant"):
        assert expected in names, f"Missing built-in capability: {expected}"


def test_chat_and_time_enabled_by_default():
    reset_capability_registry()
    registry = get_capability_registry()
    chat = registry.get("chat")
    time_cap = registry.get("time_now")
    assert chat is not None and chat.enabled is True
    assert time_cap is not None and time_cap.enabled is True


# ---------------------------------------------------------------------------
# __all__ exports
# ---------------------------------------------------------------------------


def test_module_all_exports():
    import rex.capabilities.registry as m

    for name in (
        "Capability",
        "CapabilityRegistry",
        "get_capability_registry",
        "populate_from_config",
        "reset_capability_registry",
    ):
        assert name in m.__all__, f"Missing from __all__: {name}"
