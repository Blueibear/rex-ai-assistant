"""Tests for the tool registry module."""

from __future__ import annotations

from pathlib import Path

import pytest

from rex.credentials import CredentialManager
from rex.openclaw.tool_registry import (
    MissingCredentialError,
    ToolMeta,
    ToolNotFoundError,
    ToolRegistry,
    get_tool_registry,
    register_tool,
    set_tool_registry,
)


class TestToolMeta:
    """Tests for the ToolMeta dataclass."""

    def test_toolmeta_creation(self):
        """Test creating tool metadata."""
        tool = ToolMeta(
            name="test_tool",
            description="A test tool",
            required_credentials=["api_key"],
            capabilities=["read", "write"],
            version="1.2.3",
        )
        assert tool.name == "test_tool"
        assert tool.description == "A test tool"
        assert tool.required_credentials == ["api_key"]
        assert tool.capabilities == ["read", "write"]
        assert tool.version == "1.2.3"
        assert tool.enabled is True

    def test_toolmeta_default_values(self):
        """Test tool metadata default values."""
        tool = ToolMeta(name="minimal", description="Minimal tool")
        assert tool.required_credentials == []
        assert tool.capabilities == []
        assert tool.version == "1.0.0"
        assert tool.enabled is True

    def test_toolmeta_default_health_check(self):
        """Test default health check passes."""
        tool = ToolMeta(name="test", description="Test")
        ok, message = tool.health_check()
        assert ok is True
        assert message == "OK"

    def test_toolmeta_custom_health_check(self):
        """Test custom health check function."""

        def custom_check() -> tuple[bool, str]:
            return False, "Service unavailable"

        tool = ToolMeta(
            name="unhealthy",
            description="Unhealthy tool",
            health_check=custom_check,
        )
        ok, message = tool.health_check()
        assert ok is False
        assert message == "Service unavailable"

    def test_toolmeta_empty_name_raises(self):
        """Test that empty name raises ValueError."""
        with pytest.raises(ValueError, match="name cannot be empty"):
            ToolMeta(name="", description="Description")

    def test_toolmeta_empty_description_raises(self):
        """Test that empty description raises ValueError."""
        with pytest.raises(ValueError, match="description cannot be empty"):
            ToolMeta(name="tool", description="")


class TestToolRegistry:
    """Tests for the ToolRegistry class."""

    def test_register_and_get_tool(self):
        """Test registering and retrieving a tool."""
        registry = ToolRegistry()
        tool = ToolMeta(name="test_tool", description="Test tool")
        registry.register_tool(tool)

        retrieved = registry.get_tool("test_tool")
        assert retrieved is not None
        assert retrieved.name == "test_tool"

    def test_get_tool_returns_none_for_unknown(self):
        """Test that get_tool returns None for unknown tools."""
        registry = ToolRegistry()
        assert registry.get_tool("unknown") is None

    def test_has_tool(self):
        """Test has_tool method."""
        registry = ToolRegistry()
        registry.register_tool(ToolMeta(name="exists", description="Exists"))

        assert registry.has_tool("exists")
        assert not registry.has_tool("missing")

    def test_unregister_tool(self):
        """Test unregistering a tool."""
        registry = ToolRegistry()
        registry.register_tool(ToolMeta(name="removable", description="Removable"))

        assert registry.unregister_tool("removable")
        assert not registry.has_tool("removable")

    def test_unregister_nonexistent_tool(self):
        """Test unregistering a nonexistent tool."""
        registry = ToolRegistry()
        assert not registry.unregister_tool("never_existed")

    def test_list_tools_sorted(self):
        """Test that list_tools returns sorted list."""
        registry = ToolRegistry()
        registry.register_tool(ToolMeta(name="zebra", description="Last"))
        registry.register_tool(ToolMeta(name="alpha", description="First"))
        registry.register_tool(ToolMeta(name="middle", description="Middle"))

        tools = registry.list_tools()
        names = [t.name for t in tools]
        assert names == ["alpha", "middle", "zebra"]

    def test_list_tools_excludes_disabled(self):
        """Test that list_tools excludes disabled tools by default."""
        registry = ToolRegistry()
        registry.register_tool(ToolMeta(name="enabled", description="Enabled"))
        registry.register_tool(ToolMeta(name="disabled", description="Disabled", enabled=False))

        tools = registry.list_tools()
        names = [t.name for t in tools]
        assert "enabled" in names
        assert "disabled" not in names

    def test_list_tools_includes_disabled(self):
        """Test that list_tools includes disabled when requested."""
        registry = ToolRegistry()
        registry.register_tool(ToolMeta(name="enabled", description="Enabled"))
        registry.register_tool(ToolMeta(name="disabled", description="Disabled", enabled=False))

        tools = registry.list_tools(include_disabled=True)
        names = [t.name for t in tools]
        assert "enabled" in names
        assert "disabled" in names

    def test_check_health(self):
        """Test health check for a tool."""
        registry = ToolRegistry()
        registry.register_tool(
            ToolMeta(
                name="healthy",
                description="Healthy tool",
                health_check=lambda: (True, "All good"),
            )
        )

        ok, message = registry.check_health("healthy")
        assert ok is True
        assert message == "All good"

    def test_check_health_unknown_tool(self):
        """Test health check for unknown tool raises error."""
        registry = ToolRegistry()
        with pytest.raises(ToolNotFoundError):
            registry.check_health("unknown")

    def test_check_health_disabled_tool(self):
        """Test health check for disabled tool returns disabled."""
        registry = ToolRegistry()
        registry.register_tool(
            ToolMeta(
                name="disabled",
                description="Disabled",
                enabled=False,
            )
        )

        ok, message = registry.check_health("disabled")
        assert ok is False
        assert "disabled" in message.lower()

    def test_check_health_exception_handling(self):
        """Test that health check exceptions are handled."""

        def bad_check() -> tuple[bool, str]:
            raise RuntimeError("Check failed!")

        registry = ToolRegistry()
        registry.register_tool(
            ToolMeta(
                name="broken",
                description="Broken tool",
                health_check=bad_check,
            )
        )

        ok, message = registry.check_health("broken")
        assert ok is False
        assert "error" in message.lower()

    def test_check_all_health(self):
        """Test checking health of all tools."""
        registry = ToolRegistry()
        registry.register_tool(
            ToolMeta(
                name="healthy",
                description="Healthy",
                health_check=lambda: (True, "OK"),
            )
        )
        registry.register_tool(
            ToolMeta(
                name="unhealthy",
                description="Unhealthy",
                health_check=lambda: (False, "Bad"),
            )
        )

        results = registry.check_all_health()
        assert results["healthy"] == (True, "OK")
        assert results["unhealthy"] == (False, "Bad")


class TestToolRegistryCredentials:
    """Tests for credential checking in ToolRegistry."""

    def setup_method(self):
        """Set up test fixtures."""
        self.credential_manager = CredentialManager(config_path=Path("/nonexistent/path.json"))

    def test_check_credentials_all_available(self):
        """Test credential check when all are available."""
        self.credential_manager.set_token("api_key", "secret")
        registry = ToolRegistry(credential_manager=self.credential_manager)
        registry.register_tool(
            ToolMeta(
                name="needs_key",
                description="Needs API key",
                required_credentials=["api_key"],
            )
        )

        all_available, missing = registry.check_credentials("needs_key")
        assert all_available is True
        assert missing == []

    def test_check_credentials_some_missing(self):
        """Test credential check when some are missing."""
        self.credential_manager.set_token("key_a", "secret_a")
        registry = ToolRegistry(credential_manager=self.credential_manager)
        registry.register_tool(
            ToolMeta(
                name="needs_both",
                description="Needs both keys",
                required_credentials=["key_a", "key_b"],
            )
        )

        all_available, missing = registry.check_credentials("needs_both")
        assert all_available is False
        assert missing == ["key_b"]

    def test_check_credentials_none_required(self):
        """Test credential check when none are required."""
        registry = ToolRegistry(credential_manager=self.credential_manager)
        registry.register_tool(
            ToolMeta(
                name="no_creds",
                description="No credentials needed",
                required_credentials=[],
            )
        )

        all_available, missing = registry.check_credentials("no_creds")
        assert all_available is True
        assert missing == []

    def test_check_credentials_unknown_tool(self):
        """Test credential check for unknown tool raises error."""
        registry = ToolRegistry(credential_manager=self.credential_manager)
        with pytest.raises(ToolNotFoundError):
            registry.check_credentials("unknown")

    def test_validate_credentials_raises_on_missing(self):
        """Test validate_credentials raises MissingCredentialError."""
        registry = ToolRegistry(credential_manager=self.credential_manager)
        registry.register_tool(
            ToolMeta(
                name="needs_creds",
                description="Needs credentials",
                required_credentials=["missing_cred"],
            )
        )

        with pytest.raises(MissingCredentialError) as exc_info:
            registry.validate_credentials_for_tool("needs_creds")

        assert exc_info.value.tool_name == "needs_creds"
        assert "missing_cred" in exc_info.value.missing_credentials


class TestToolRegistryStatus:
    """Tests for tool status functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.credential_manager = CredentialManager(config_path=Path("/nonexistent/path.json"))

    def test_get_tool_status_ready(self):
        """Test status for a ready tool."""
        self.credential_manager.set_token("required_key", "value")
        registry = ToolRegistry(credential_manager=self.credential_manager)
        registry.register_tool(
            ToolMeta(
                name="ready_tool",
                description="Ready tool",
                required_credentials=["required_key"],
                capabilities=["read"],
                health_check=lambda: (True, "OK"),
            )
        )

        status = registry.get_tool_status("ready_tool")
        assert status["name"] == "ready_tool"
        assert status["description"] == "Ready tool"
        assert status["enabled"] is True
        assert status["credentials_available"] is True
        assert status["missing_credentials"] == []
        assert status["health_ok"] is True
        assert status["ready"] is True

    def test_get_tool_status_missing_creds(self):
        """Test status for tool with missing credentials."""
        registry = ToolRegistry(credential_manager=self.credential_manager)
        registry.register_tool(
            ToolMeta(
                name="missing_creds_tool",
                description="Missing creds",
                required_credentials=["missing_key"],
            )
        )

        status = registry.get_tool_status("missing_creds_tool")
        assert status["credentials_available"] is False
        assert "missing_key" in status["missing_credentials"]
        assert status["ready"] is False

    def test_get_tool_status_unhealthy(self):
        """Test status for unhealthy tool."""
        registry = ToolRegistry(credential_manager=self.credential_manager)
        registry.register_tool(
            ToolMeta(
                name="unhealthy_tool",
                description="Unhealthy",
                health_check=lambda: (False, "Down for maintenance"),
            )
        )

        status = registry.get_tool_status("unhealthy_tool")
        assert status["health_ok"] is False
        assert status["health_message"] == "Down for maintenance"
        assert status["ready"] is False

    def test_get_tool_status_disabled(self):
        """Test status for disabled tool."""
        registry = ToolRegistry(credential_manager=self.credential_manager)
        registry.register_tool(
            ToolMeta(
                name="disabled_tool",
                description="Disabled",
                enabled=False,
            )
        )

        status = registry.get_tool_status("disabled_tool")
        assert status["enabled"] is False
        assert status["ready"] is False

    def test_get_tool_status_unknown_raises(self):
        """Test status for unknown tool raises error."""
        registry = ToolRegistry(credential_manager=self.credential_manager)
        with pytest.raises(ToolNotFoundError):
            registry.get_tool_status("unknown")

    def test_get_all_status(self):
        """Test getting status for all tools."""
        registry = ToolRegistry(credential_manager=self.credential_manager)
        registry.register_tool(ToolMeta(name="alpha", description="Alpha"))
        registry.register_tool(ToolMeta(name="beta", description="Beta"))

        statuses = registry.get_all_status()
        assert len(statuses) == 2
        names = [s["name"] for s in statuses]
        assert names == ["alpha", "beta"]  # Sorted


class TestGlobalToolRegistry:
    """Tests for global tool registry functions."""

    def setup_method(self):
        """Reset global state before each test."""
        set_tool_registry(None)  # type: ignore

    def test_get_tool_registry_returns_singleton(self):
        """Test that get_tool_registry returns singleton with builtin tools."""
        registry1 = get_tool_registry()
        registry2 = get_tool_registry()

        assert registry1 is registry2

    def test_get_tool_registry_has_builtin_tools(self):
        """Test that singleton has built-in tools registered."""
        registry = get_tool_registry()

        # Check for built-in tools
        assert registry.has_tool("time_now")
        assert registry.has_tool("weather_now")
        assert registry.has_tool("web_search")

    def test_set_tool_registry_replaces_singleton(self):
        """Test that set_tool_registry replaces the singleton."""
        custom_registry = ToolRegistry()
        custom_registry.register_tool(ToolMeta(name="custom", description="Custom"))

        set_tool_registry(custom_registry)

        assert get_tool_registry() is custom_registry
        assert get_tool_registry().has_tool("custom")

    def test_register_tool_convenience_function(self):
        """Test the register_tool convenience function."""
        # Get fresh registry
        registry = get_tool_registry()

        register_tool(ToolMeta(name="convenience_tool", description="Convenience"))

        assert registry.has_tool("convenience_tool")


class TestToolRegistryIntegration:
    """Integration tests with CredentialManager."""

    def test_tool_with_credential_check_integration(self):
        """Test tool credential checking with real CredentialManager."""
        # Create a fresh credential manager with some tokens
        cred_manager = CredentialManager(config_path=Path("/nonexistent/path.json"))
        cred_manager.set_token("available_service", "token123")

        # Create registry with the credential manager
        registry = ToolRegistry(credential_manager=cred_manager)

        # Register tools with different credential requirements
        registry.register_tool(
            ToolMeta(
                name="tool_with_creds",
                description="Needs available service",
                required_credentials=["available_service"],
            )
        )
        registry.register_tool(
            ToolMeta(
                name="tool_without_creds",
                description="Needs unavailable service",
                required_credentials=["unavailable_service"],
            )
        )
        registry.register_tool(
            ToolMeta(
                name="tool_no_requirements",
                description="No credential requirements",
                required_credentials=[],
            )
        )

        # Check statuses
        status1 = registry.get_tool_status("tool_with_creds")
        assert status1["credentials_available"] is True

        status2 = registry.get_tool_status("tool_without_creds")
        assert status2["credentials_available"] is False

        status3 = registry.get_tool_status("tool_no_requirements")
        assert status3["credentials_available"] is True


class TestBuiltinTools:
    """Tests for built-in tool registrations."""

    def setup_method(self):
        """Reset global state before each test."""
        set_tool_registry(None)  # type: ignore

    def test_time_now_tool_registered(self):
        """Test that time_now is registered correctly."""
        registry = get_tool_registry()
        tool = registry.get_tool("time_now")

        assert tool is not None
        assert tool.name == "time_now"
        assert "time" in tool.description.lower()
        assert tool.required_credentials == []

        # Health check should pass
        ok, msg = tool.health_check()
        assert ok is True

    def test_weather_now_tool_registered(self):
        """Test that weather_now is registered correctly."""
        registry = get_tool_registry()
        tool = registry.get_tool("weather_now")

        assert tool is not None
        assert "weather" in tool.description.lower()
        assert "openweathermap" in tool.required_credentials

    def test_web_search_tool_registered(self):
        """Test that web_search is registered correctly."""
        registry = get_tool_registry()
        tool = registry.get_tool("web_search")

        assert tool is not None
        assert "search" in tool.description.lower()

    def test_home_assistant_tool_registered(self):
        """Test that home_assistant is registered correctly."""
        registry = get_tool_registry()
        tool = registry.get_tool("home_assistant")

        assert tool is not None
        assert "home" in tool.description.lower()
        assert "home_assistant" in tool.required_credentials
