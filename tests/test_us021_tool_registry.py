"""US-021: Tool registry tests.

Acceptance criteria:
- tools register correctly
- tool metadata stored
- duplicate tools prevented
- Typecheck passes
"""

from __future__ import annotations

import pytest

from rex.tool_registry import ToolMeta, ToolNotFoundError, ToolRegistry

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_registry() -> ToolRegistry:
    registry = ToolRegistry()
    return registry


def _make_tool(name: str = "my_tool", description: str = "A test tool") -> ToolMeta:
    return ToolMeta(name=name, description=description)


# ---------------------------------------------------------------------------
# 1. Tools register correctly
# ---------------------------------------------------------------------------


class TestToolsRegisterCorrectly:
    def test_register_single_tool(self):
        """A tool registered with register_tool() is immediately retrievable."""
        registry = _make_registry()
        tool = _make_tool("ping", "Ping a host")
        registry.register_tool(tool)
        assert registry.has_tool("ping")

    def test_register_multiple_tools(self):
        """Multiple tools can be registered independently."""
        registry = _make_registry()
        for name in ("alpha", "beta", "gamma"):
            registry.register_tool(_make_tool(name, f"{name} tool"))
        for name in ("alpha", "beta", "gamma"):
            assert registry.has_tool(name)

    def test_has_tool_returns_false_for_unregistered(self):
        """has_tool() returns False for a name never registered."""
        registry = _make_registry()
        assert not registry.has_tool("nonexistent")

    def test_unregister_tool_removes_it(self):
        """unregister_tool() removes a previously registered tool."""
        registry = _make_registry()
        registry.register_tool(_make_tool("temp"))
        registry.unregister_tool("temp")
        assert not registry.has_tool("temp")

    def test_unregister_nonexistent_returns_false(self):
        """unregister_tool() returns False when the tool does not exist."""
        registry = _make_registry()
        result = registry.unregister_tool("ghost")
        assert result is False


# ---------------------------------------------------------------------------
# 2. Tool metadata stored
# ---------------------------------------------------------------------------


class TestToolMetadataStored:
    def test_metadata_name_preserved(self):
        """Tool name is stored verbatim."""
        registry = _make_registry()
        registry.register_tool(_make_tool("weather", "Get weather"))
        assert registry.get_tool("weather").name == "weather"  # type: ignore[union-attr]

    def test_metadata_description_preserved(self):
        """Tool description is stored verbatim."""
        registry = _make_registry()
        registry.register_tool(ToolMeta(name="foo", description="Does foo things"))
        assert registry.get_tool("foo").description == "Does foo things"  # type: ignore[union-attr]

    def test_metadata_capabilities_stored(self):
        """Capability tags are stored in tool metadata."""
        registry = _make_registry()
        registry.register_tool(
            ToolMeta(name="writer", description="Write files", capabilities=["write", "fs"])
        )
        stored = registry.get_tool("writer")
        assert stored is not None
        assert "write" in stored.capabilities
        assert "fs" in stored.capabilities

    def test_metadata_required_credentials_stored(self):
        """required_credentials list is stored in tool metadata."""
        registry = _make_registry()
        registry.register_tool(
            ToolMeta(
                name="emailer",
                description="Send email",
                required_credentials=["email"],
            )
        )
        stored = registry.get_tool("emailer")
        assert stored is not None
        assert "email" in stored.required_credentials

    def test_metadata_version_stored(self):
        """Custom version string is preserved."""
        registry = _make_registry()
        registry.register_tool(ToolMeta(name="v_tool", description="versioned", version="2.3.1"))
        assert registry.get_tool("v_tool").version == "2.3.1"  # type: ignore[union-attr]

    def test_list_tools_returns_all_registered(self):
        """list_tools() returns every enabled tool that was registered."""
        registry = _make_registry()
        for name in ("one", "two", "three"):
            registry.register_tool(_make_tool(name, f"{name} desc"))
        names = {t.name for t in registry.list_tools()}
        assert {"one", "two", "three"} <= names

    def test_get_tool_returns_none_for_missing(self):
        """get_tool() returns None for an unregistered name."""
        registry = _make_registry()
        assert registry.get_tool("missing") is None

    def test_tool_meta_empty_name_raises(self):
        """ToolMeta rejects an empty name."""
        with pytest.raises(ValueError, match="name"):
            ToolMeta(name="", description="oops")

    def test_tool_meta_empty_description_raises(self):
        """ToolMeta rejects an empty description."""
        with pytest.raises(ValueError, match="description"):
            ToolMeta(name="ok", description="")


# ---------------------------------------------------------------------------
# 3. Duplicate tools prevented / overwrite behaviour
# ---------------------------------------------------------------------------


class TestDuplicateToolHandling:
    def test_registering_same_name_twice_overwrites(self):
        """Re-registering a tool with the same name replaces the old entry."""
        registry = _make_registry()
        registry.register_tool(ToolMeta(name="dup", description="version one"))
        registry.register_tool(ToolMeta(name="dup", description="version two"))
        # Only one entry under the name
        matches = [t for t in registry.list_tools(include_disabled=True) if t.name == "dup"]
        assert len(matches) == 1

    def test_overwrite_updates_description(self):
        """The second registration's description replaces the first."""
        registry = _make_registry()
        registry.register_tool(ToolMeta(name="dup", description="old"))
        registry.register_tool(ToolMeta(name="dup", description="new"))
        assert registry.get_tool("dup").description == "new"  # type: ignore[union-attr]

    def test_overwrite_updates_capabilities(self):
        """Re-registering with new capabilities overwrites previous ones."""
        registry = _make_registry()
        registry.register_tool(ToolMeta(name="cap_tool", description="cap", capabilities=["read"]))
        registry.register_tool(
            ToolMeta(name="cap_tool", description="cap", capabilities=["read", "write"])
        )
        stored = registry.get_tool("cap_tool")
        assert stored is not None
        assert "write" in stored.capabilities

    def test_list_tools_contains_no_duplicates(self):
        """list_tools() never returns the same tool name more than once."""
        registry = _make_registry()
        for _ in range(5):
            registry.register_tool(ToolMeta(name="repeated", description="again"))
        repeated = [t for t in registry.list_tools(include_disabled=True) if t.name == "repeated"]
        assert len(repeated) == 1

    def test_check_credentials_raises_for_unregistered(self):
        """check_credentials() raises ToolNotFoundError for unknown tools."""
        registry = _make_registry()
        with pytest.raises(ToolNotFoundError):
            registry.check_credentials("unknown_tool")
