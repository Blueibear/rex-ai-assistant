"""Tests for rex.tool_router — dispatch skeleton and UnknownToolError."""

from __future__ import annotations

import pytest

from rex.tool_catalog import EXECUTABLE_TOOLS
from rex.tool_router import UnknownToolError, execute_tool


class TestUnknownToolError:
    def test_raises_for_unknown_tool(self):
        with pytest.raises(UnknownToolError) as exc_info:
            execute_tool("totally_fake_tool", {})
        assert "totally_fake_tool" in str(exc_info.value)

    def test_error_carries_tool_name(self):
        try:
            execute_tool("nonexistent", {})
        except UnknownToolError as exc:
            assert exc.tool_name == "nonexistent"

    def test_error_message_lists_catalog(self):
        try:
            execute_tool("bogus", {})
        except UnknownToolError as exc:
            # Message should mention known tools
            for tool in EXECUTABLE_TOOLS:
                assert tool in str(exc)

    def test_unknown_tool_is_not_generic_exception(self):
        """UnknownToolError must be its own class, not a generic KeyError/ValueError."""
        with pytest.raises(UnknownToolError):
            execute_tool("not_in_catalog", {})
        # Ensure it is NOT just a ValueError or KeyError
        try:
            execute_tool("not_in_catalog", {})
        except (ValueError, KeyError):
            pytest.fail("Should be UnknownToolError, not ValueError/KeyError")
        except UnknownToolError:
            pass  # correct


class TestExecuteToolCatalogCoverage:
    """Every tool in EXECUTABLE_TOOLS must be callable without raising."""

    @pytest.mark.parametrize("tool_name", sorted(EXECUTABLE_TOOLS))
    def test_all_catalog_tools_return_string(self, tool_name: str):
        """Each tool must return a non-empty string (may be '[integration not configured]')."""
        result = execute_tool(tool_name, {})
        assert isinstance(result, str), f"{tool_name} did not return str"
        assert len(result) > 0, f"{tool_name} returned empty string"

    @pytest.mark.parametrize("tool_name", sorted(EXECUTABLE_TOOLS))
    def test_catalog_tools_do_not_raise(self, tool_name: str):
        """No catalog tool should raise an exception when called with empty args."""
        try:
            execute_tool(tool_name, {})
        except UnknownToolError:
            pytest.fail(f"{tool_name} is in EXECUTABLE_TOOLS but raised UnknownToolError")


class TestTimeNow:
    def test_returns_time_string(self):
        result = execute_tool("time_now", {"location": "UTC"})
        assert "UTC" in result or "time" in result.lower()

    def test_empty_args_ok(self):
        result = execute_tool("time_now", {})
        assert isinstance(result, str) and len(result) > 0

    def test_location_reflected_in_output(self):
        result = execute_tool("time_now", {"location": "New York"})
        assert "New York" in result


class TestStubHandlers:
    """Stubs (US-186/187) must return the not-configured sentinel string."""

    @pytest.mark.parametrize(
        "tool_name",
        ["weather_now", "web_search", "send_email", "calendar_create_event"],
    )
    def test_stub_returns_not_configured(self, tool_name: str):
        result = execute_tool(tool_name, {})
        assert result == "[integration not configured]"
