"""Tests for rex.openclaw.tools.time_tool — US-P2-006."""

from __future__ import annotations

from unittest.mock import patch


class TestTimeTool:
    def test_time_now_with_known_location_returns_dict(self):
        """time_now returns a dict with local_time, date, timezone keys."""
        from rex.openclaw.tools.time_tool import time_now

        result = time_now("London")
        assert isinstance(result, dict)
        # Either a successful result or an error — either way it's a dict
        if "error" not in result:
            assert "local_time" in result
            assert "date" in result
            assert "timezone" in result

    def test_time_now_success_structure(self):
        """On success, time_now returns correctly shaped result."""
        from rex.openclaw.tools.time_tool import time_now

        fake_result = {
            "local_time": "2026-03-22 14:30",
            "date": "2026-03-22",
            "timezone": "Europe/London",
        }
        with patch(
            "rex.openclaw.tools.time_tool.execute_tool", return_value=fake_result
        ) as mock_exec:
            result = time_now("London")

        assert result == fake_result
        call_args = mock_exec.call_args
        request = call_args.args[0]
        assert request["tool"] == "time_now"
        assert request["args"]["location"] == "London"

    def test_time_now_passes_location_in_args(self):
        """time_now forwards location to execute_tool args."""
        from rex.openclaw.tools.time_tool import time_now

        with patch("rex.openclaw.tools.time_tool.execute_tool", return_value={}) as mock_exec:
            time_now("Edinburgh, Scotland")

        call_args = mock_exec.call_args
        assert call_args.args[0]["args"]["location"] == "Edinburgh, Scotland"

    def test_time_now_no_location_omits_from_args(self):
        """When location is None, args dict has no 'location' key."""
        from rex.openclaw.tools.time_tool import time_now

        with patch("rex.openclaw.tools.time_tool.execute_tool", return_value={}) as mock_exec:
            time_now()

        call_args = mock_exec.call_args
        assert "location" not in call_args.args[0]["args"]

    def test_time_now_passes_context(self):
        """Context dict is forwarded as default_context to execute_tool."""
        from rex.openclaw.tools.time_tool import time_now

        ctx = {"location": "Paris", "timezone": "Europe/Paris"}
        with patch("rex.openclaw.tools.time_tool.execute_tool", return_value={}) as mock_exec:
            time_now(context=ctx)

        call_args = mock_exec.call_args
        # second positional arg is default_context
        assert call_args.args[1] == ctx

    def test_time_now_skips_policy_and_audit(self):
        """time_now disables policy/credential/audit checks."""
        from rex.openclaw.tools.time_tool import time_now

        with patch("rex.openclaw.tools.time_tool.execute_tool", return_value={}) as mock_exec:
            time_now("Tokyo")

        call_kwargs = mock_exec.call_args.kwargs
        assert call_kwargs.get("skip_policy_check") is True
        assert call_kwargs.get("skip_credential_check") is True
        assert call_kwargs.get("skip_audit_log") is True

    def test_register_returns_none_without_openclaw(self):
        """register() returns None and logs a warning when openclaw is absent."""
        from rex.openclaw.tools.time_tool import register

        # openclaw is not installed in the test environment
        result = register()
        assert result is None

    def test_register_accepts_agent_arg(self):
        """register() accepts an agent argument without error."""
        from rex.openclaw.tools.time_tool import register

        result = register(agent=object())
        assert result is None

    def test_tool_name_constant(self):
        from rex.openclaw.tools import time_tool

        assert time_tool.TOOL_NAME == "time_now"

    def test_tool_description_is_string(self):
        from rex.openclaw.tools import time_tool

        assert isinstance(time_tool.TOOL_DESCRIPTION, str)
        assert len(time_tool.TOOL_DESCRIPTION) > 0


class TestTimeToolPackage:
    def test_tools_package_importable(self):
        import rex.openclaw.tools  # noqa: F401

    def test_time_tool_importable(self):
        import rex.openclaw.tools.time_tool  # noqa: F401
