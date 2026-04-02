"""Unit tests for rex.tools.registry (US-TD-001)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from rex.tools.registry import Tool, ToolRegistry

# ---------------------------------------------------------------------------
# Minimal config stub used across tests
# ---------------------------------------------------------------------------


@dataclass
class _Config:
    openweathermap_api_key: str | None = None
    brave_api_key: str | None = None
    email_accounts: list[Any] = field(default_factory=list)
    ha_base_url: str | None = None
    ha_token: str | None = None


def _dummy_handler(**kwargs: Any) -> dict[str, Any]:
    return {"ok": True}


# ---------------------------------------------------------------------------
# Tool construction
# ---------------------------------------------------------------------------


def test_tool_construction_valid() -> None:
    tool = Tool(
        name="test_tool",
        description="A test tool.",
        capability_tags=["test"],
        requires_config=[],
        handler=_dummy_handler,
    )
    assert tool.name == "test_tool"
    assert tool.description == "A test tool."
    assert tool.capability_tags == ["test"]
    assert tool.requires_config == []
    assert tool.handler is _dummy_handler


def test_tool_empty_name_raises() -> None:
    with pytest.raises(ValueError, match="name cannot be empty"):
        Tool(
            name="",
            description="desc",
            capability_tags=[],
            requires_config=[],
            handler=_dummy_handler,
        )


def test_tool_empty_description_raises() -> None:
    with pytest.raises(ValueError, match="description cannot be empty"):
        Tool(
            name="t",
            description="",
            capability_tags=[],
            requires_config=[],
            handler=_dummy_handler,
        )


# ---------------------------------------------------------------------------
# ToolRegistry — registration
# ---------------------------------------------------------------------------


def test_register_and_get() -> None:
    registry = ToolRegistry()
    tool = Tool("t1", "desc", ["tag"], [], _dummy_handler)
    registry.register(tool)
    assert registry.get("t1") is tool


def test_get_unknown_returns_none() -> None:
    registry = ToolRegistry()
    assert registry.get("nonexistent") is None


def test_register_replaces_existing() -> None:
    registry = ToolRegistry()
    t1 = Tool("t", "first", [], [], _dummy_handler)
    t2 = Tool("t", "second", [], [], _dummy_handler)
    registry.register(t1)
    registry.register(t2)
    assert registry.get("t") is t2


def test_all_tools_returns_all() -> None:
    registry = ToolRegistry()
    registry.register(Tool("a", "desc a", [], [], _dummy_handler))
    registry.register(Tool("b", "desc b", [], [], _dummy_handler))
    names = {t.name for t in registry.all_tools()}
    assert names == {"a", "b"}


# ---------------------------------------------------------------------------
# ToolRegistry — availability filtering
# ---------------------------------------------------------------------------


def test_available_tools_no_requirements_always_included() -> None:
    registry = ToolRegistry()
    registry.register(Tool("always", "always available", ["x"], [], _dummy_handler))
    config = _Config()
    available = registry.available_tools(config)
    assert any(t.name == "always" for t in available)


def test_available_tools_missing_config_excluded() -> None:
    registry = ToolRegistry()
    registry.register(
        Tool(
            name="weather",
            description="Weather tool.",
            capability_tags=["weather"],
            requires_config=["openweathermap_api_key"],
            handler=_dummy_handler,
        )
    )
    config = _Config(openweathermap_api_key=None)
    available = registry.available_tools(config)
    assert not any(t.name == "weather" for t in available)


def test_available_tools_present_config_included() -> None:
    registry = ToolRegistry()
    registry.register(
        Tool(
            name="weather",
            description="Weather tool.",
            capability_tags=["weather"],
            requires_config=["openweathermap_api_key"],
            handler=_dummy_handler,
        )
    )
    config = _Config(openweathermap_api_key="test-key")
    available = registry.available_tools(config)
    assert any(t.name == "weather" for t in available)


def test_available_tools_all_requirements_must_be_satisfied() -> None:
    registry = ToolRegistry()
    registry.register(
        Tool(
            name="ha",
            description="Home Assistant.",
            capability_tags=["iot"],
            requires_config=["ha_base_url", "ha_token"],
            handler=_dummy_handler,
        )
    )
    # Only one of the two fields set → not available
    config = _Config(ha_base_url="http://homeassistant.local", ha_token=None)
    available = registry.available_tools(config)
    assert not any(t.name == "ha" for t in available)

    # Both fields set → available
    config2 = _Config(ha_base_url="http://homeassistant.local", ha_token="token123")
    available2 = registry.available_tools(config2)
    assert any(t.name == "ha" for t in available2)


def test_available_tools_empty_list_field_excluded() -> None:
    """email_accounts=[] (falsy list) must exclude send_email."""
    registry = ToolRegistry()
    registry.register(
        Tool(
            name="send_email",
            description="Send email.",
            capability_tags=["email"],
            requires_config=["email_accounts"],
            handler=_dummy_handler,
        )
    )
    config_empty = _Config(email_accounts=[])
    assert not any(t.name == "send_email" for t in registry.available_tools(config_empty))

    config_with_account = _Config(email_accounts=[object()])
    assert any(t.name == "send_email" for t in registry.available_tools(config_with_account))


def test_available_tools_unknown_config_attr_treated_as_missing() -> None:
    registry = ToolRegistry()
    registry.register(
        Tool(
            name="exotic",
            description="Tool needing obscure setting.",
            capability_tags=[],
            requires_config=["exotic_api_key"],
            handler=_dummy_handler,
        )
    )
    config = _Config()  # no exotic_api_key attribute
    available = registry.available_tools(config)
    assert not any(t.name == "exotic" for t in available)


# ---------------------------------------------------------------------------
# get_default_registry smoke tests (no network calls)
# ---------------------------------------------------------------------------


def test_get_default_registry_contains_expected_tools() -> None:
    """Smoke test: default registry has all documented tools."""
    from rex.tools.registry import get_default_registry

    registry = get_default_registry()
    names = {t.name for t in registry.all_tools()}
    expected = {
        "time_now",
        "weather_now",
        "web_search",
        "send_email",
        "calendar_create",
        "home_assistant_call_service",
        "send_sms",
        "file_ops",
    }
    assert expected.issubset(names), f"Missing tools: {expected - names}"


def test_get_default_registry_availability_filter_works() -> None:
    """With empty config only unconditional tools are returned."""
    from rex.tools.registry import get_default_registry

    registry = get_default_registry()

    class _EmptyConfig:
        pass

    available_names = {t.name for t in registry.available_tools(_EmptyConfig())}
    # Tools that require no config keys must be present
    assert "time_now" in available_names
    assert "calendar_create" in available_names
    assert "send_sms" in available_names
    assert "file_ops" in available_names
    # Tools that require config keys must be absent
    assert "weather_now" not in available_names
    assert "web_search" not in available_names
    assert "home_assistant_call_service" not in available_names
    assert "send_email" not in available_names
