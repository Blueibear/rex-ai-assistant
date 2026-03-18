"""US-023: Capability discovery tests.

Acceptance criteria:
- capabilities enumerated
- tools discoverable
- capability metadata exposed
- Typecheck passes
"""

from __future__ import annotations

import pytest

from rex.tool_registry import (
    ToolMeta,
    ToolNotFoundError,
    ToolRegistry,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def registry() -> ToolRegistry:
    """Fresh registry with a few test tools."""
    reg = ToolRegistry()

    reg.register_tool(
        ToolMeta(
            name="alpha",
            description="Alpha tool",
            capabilities=["read", "network"],
            version="1.0.0",
        )
    )
    reg.register_tool(
        ToolMeta(
            name="beta",
            description="Beta tool",
            capabilities=["write", "email"],
            version="2.0.0",
        )
    )
    reg.register_tool(
        ToolMeta(
            name="gamma",
            description="Gamma tool",
            capabilities=["read", "write", "iot"],
            version="1.5.0",
            enabled=False,
        )
    )
    return reg


# ---------------------------------------------------------------------------
# Capabilities enumerated
# ---------------------------------------------------------------------------


def test_list_tools_returns_enabled_tools(registry: ToolRegistry) -> None:
    """list_tools() returns only enabled tools by default."""
    tools = registry.list_tools()
    names = [t.name for t in tools]
    assert "alpha" in names
    assert "beta" in names
    assert "gamma" not in names


def test_list_tools_include_disabled(registry: ToolRegistry) -> None:
    """list_tools(include_disabled=True) includes disabled tools."""
    tools = registry.list_tools(include_disabled=True)
    names = [t.name for t in tools]
    assert "gamma" in names


def test_list_tools_sorted_by_name(registry: ToolRegistry) -> None:
    """list_tools() returns tools sorted alphabetically by name."""
    tools = registry.list_tools(include_disabled=True)
    names = [t.name for t in tools]
    assert names == sorted(names)


def test_enumerate_all_capabilities(registry: ToolRegistry) -> None:
    """All capability tags across all tools can be collected."""
    all_caps: set[str] = set()
    for tool in registry.list_tools(include_disabled=True):
        all_caps.update(tool.capabilities)
    assert "read" in all_caps
    assert "write" in all_caps
    assert "network" in all_caps
    assert "email" in all_caps
    assert "iot" in all_caps


def test_empty_registry_list_tools() -> None:
    """Empty registry returns empty list."""
    reg = ToolRegistry()
    assert reg.list_tools() == []


# ---------------------------------------------------------------------------
# Tools discoverable
# ---------------------------------------------------------------------------


def test_has_tool_registered(registry: ToolRegistry) -> None:
    """has_tool() returns True for a registered tool."""
    assert registry.has_tool("alpha") is True


def test_has_tool_unregistered(registry: ToolRegistry) -> None:
    """has_tool() returns False for an unregistered tool."""
    assert registry.has_tool("nonexistent") is False


def test_get_tool_returns_meta(registry: ToolRegistry) -> None:
    """get_tool() returns ToolMeta for a registered tool."""
    tool = registry.get_tool("beta")
    assert tool is not None
    assert tool.name == "beta"
    assert tool.description == "Beta tool"


def test_get_tool_returns_none_for_missing(registry: ToolRegistry) -> None:
    """get_tool() returns None for an unregistered tool."""
    assert registry.get_tool("missing") is None


def test_register_and_discover_new_tool(registry: ToolRegistry) -> None:
    """Newly registered tools are immediately discoverable."""
    registry.register_tool(
        ToolMeta(name="delta", description="Delta tool", capabilities=["search"])
    )
    assert registry.has_tool("delta") is True
    tool = registry.get_tool("delta")
    assert tool is not None
    assert "search" in tool.capabilities


# ---------------------------------------------------------------------------
# Capability metadata exposed
# ---------------------------------------------------------------------------


def test_get_tool_status_exposes_capabilities(registry: ToolRegistry) -> None:
    """get_tool_status() exposes capability list in metadata."""
    status = registry.get_tool_status("alpha")
    assert "capabilities" in status
    assert "read" in status["capabilities"]
    assert "network" in status["capabilities"]


def test_get_tool_status_exposes_version(registry: ToolRegistry) -> None:
    """get_tool_status() exposes version in metadata."""
    status = registry.get_tool_status("beta")
    assert status["version"] == "2.0.0"


def test_get_tool_status_exposes_enabled(registry: ToolRegistry) -> None:
    """get_tool_status() exposes enabled flag."""
    enabled_status = registry.get_tool_status("alpha")
    disabled_status = registry.get_tool_status("gamma")
    assert enabled_status["enabled"] is True
    assert disabled_status["enabled"] is False


def test_get_tool_status_exposes_description(registry: ToolRegistry) -> None:
    """get_tool_status() exposes tool description."""
    status = registry.get_tool_status("beta")
    assert status["description"] == "Beta tool"


def test_get_tool_status_exposes_ready_flag(registry: ToolRegistry) -> None:
    """get_tool_status() exposes ready flag (enabled + creds + health)."""
    status = registry.get_tool_status("alpha")
    assert "ready" in status
    # alpha has no required credentials and default health check passes
    assert status["ready"] is True


def test_get_tool_status_disabled_not_ready(registry: ToolRegistry) -> None:
    """Disabled tool shows ready=False."""
    status = registry.get_tool_status("gamma")
    assert status["ready"] is False


def test_get_all_status_returns_all_tools(registry: ToolRegistry) -> None:
    """get_all_status() returns status for all registered tools."""
    all_status = registry.get_all_status()
    names = [s["name"] for s in all_status]
    assert "alpha" in names
    assert "beta" in names
    assert "gamma" in names


def test_get_all_status_sorted(registry: ToolRegistry) -> None:
    """get_all_status() returns tools sorted by name."""
    all_status = registry.get_all_status()
    names = [s["name"] for s in all_status]
    assert names == sorted(names)


def test_get_tool_status_raises_for_missing(registry: ToolRegistry) -> None:
    """get_tool_status() raises ToolNotFoundError for unknown tool."""
    with pytest.raises(ToolNotFoundError):
        registry.get_tool_status("nonexistent")


def test_capabilities_field_is_list(registry: ToolRegistry) -> None:
    """ToolMeta capabilities field is a list."""
    tool = registry.get_tool("alpha")
    assert tool is not None
    assert isinstance(tool.capabilities, list)


def test_get_all_status_metadata_complete(registry: ToolRegistry) -> None:
    """Each status dict contains all required metadata keys."""
    required_keys = {
        "name",
        "description",
        "version",
        "enabled",
        "capabilities",
        "required_credentials",
        "credentials_available",
        "missing_credentials",
        "health_ok",
        "health_message",
        "ready",
    }
    for status in registry.get_all_status():
        assert required_keys.issubset(status.keys()), (
            f"Missing keys in status for {status.get('name')}: " f"{required_keys - status.keys()}"
        )
