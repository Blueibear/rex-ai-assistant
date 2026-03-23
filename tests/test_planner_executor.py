"""Tests for the planner and autonomy modes modules.

This module tests:
- Planner: Goal parsing, workflow generation, validation
- Autonomy modes: Mode detection, configuration
"""

import tempfile
from pathlib import Path

import pytest

from rex.autonomy_modes import (
    AutonomyConfig,
    AutonomyMode,
    _infer_category,
    create_default_config,
    get_mode,
)
from rex.contracts import ToolCall
from rex.planner import Planner, UnableToPlanError
from rex.policy_engine import reset_policy_engine
from rex.tool_registry import ToolMeta, ToolRegistry, reset_tool_registry
from rex.workflow import Workflow, WorkflowStep


class TestPlanner:
    """Tests for the Planner class."""

    @pytest.fixture
    def tool_registry(self):
        """Create a tool registry with test tools."""
        reset_tool_registry()
        registry = ToolRegistry()

        # Register test tools
        registry.register_tool(
            ToolMeta(
                name="time_now",
                description="Get current time",
                required_credentials=[],
                health_check=lambda: (True, "OK"),
            )
        )
        registry.register_tool(
            ToolMeta(
                name="weather_now",
                description="Get weather",
                required_credentials=[],
                health_check=lambda: (True, "OK"),
            )
        )
        registry.register_tool(
            ToolMeta(
                name="send_email",
                description="Send email",
                required_credentials=[],
                health_check=lambda: (True, "OK"),
            )
        )
        registry.register_tool(
            ToolMeta(
                name="web_search",
                description="Search the web",
                required_credentials=[],
                health_check=lambda: (True, "OK"),
            )
        )
        registry.register_tool(
            ToolMeta(
                name="home_assistant_call_service",
                description="Control home devices",
                required_credentials=[],
                health_check=lambda: (True, "OK"),
            )
        )

        return registry

    @pytest.fixture
    def policy_engine(self):
        """Create a policy engine for testing."""
        reset_policy_engine()
        from rex.policy_engine import get_policy_engine

        return get_policy_engine()

    @pytest.fixture
    def planner(self, tool_registry, policy_engine):
        """Create a planner for testing."""
        return Planner(tool_registry=tool_registry, policy_engine=policy_engine)

    def test_plan_check_weather(self, planner):
        """Test planning for weather check."""
        workflow = planner.plan("check weather in Dallas")

        assert workflow.title == "check weather in Dallas"
        assert len(workflow.steps) == 1
        assert workflow.steps[0].tool_call.tool == "weather_now"
        assert workflow.steps[0].tool_call.args["location"] == "Dallas"

    def test_plan_check_time(self, planner):
        """Test planning for time check."""
        workflow = planner.plan("what's the time in New York")

        assert len(workflow.steps) == 1
        assert workflow.steps[0].tool_call.tool == "time_now"
        assert "New York" in workflow.steps[0].tool_call.args["location"]

    def test_plan_send_email(self, planner):
        """Test planning for sending email."""
        workflow = planner.plan("send email to alice@example.com")

        assert len(workflow.steps) == 1
        assert workflow.steps[0].tool_call.tool == "send_email"
        assert workflow.steps[0].tool_call.args["to"] == "alice@example.com"

    def test_plan_newsletter(self, planner):
        """Test planning for newsletter."""
        workflow = planner.plan("send monthly newsletter")

        assert len(workflow.steps) >= 1
        # Should have email step
        email_steps = [
            s for s in workflow.steps if s.tool_call and s.tool_call.tool == "send_email"
        ]
        assert len(email_steps) > 0

    def test_plan_web_search(self, planner):
        """Test planning for web search."""
        workflow = planner.plan("search for Python tutorials")

        assert len(workflow.steps) == 1
        assert workflow.steps[0].tool_call.tool == "web_search"
        assert "Python tutorials" in workflow.steps[0].tool_call.args["query"]

    def test_plan_home_control(self, planner):
        """Test planning for home automation."""
        workflow = planner.plan("turn on living room lights")

        assert len(workflow.steps) == 1
        assert workflow.steps[0].tool_call.tool == "home_assistant_call_service"
        assert workflow.steps[0].tool_call.args["service"] == "turn_on"

    def test_plan_unknown_goal(self, planner):
        """Test planning with unknown goal."""
        with pytest.raises(UnableToPlanError):
            planner.plan("do something completely unknown and unrecognizable")

    def test_validate_workflow_success(self, planner):
        """Test workflow validation with valid workflow."""
        workflow = planner.plan("check weather in Dallas")

        assert planner.validate_workflow(workflow) is True

    def test_validate_workflow_missing_tool(self, planner):
        """Test workflow validation with missing tool."""
        workflow = Workflow(
            title="Test",
            steps=[
                WorkflowStep(
                    description="Use missing tool",
                    tool_call=ToolCall(tool="nonexistent_tool", args={}),
                )
            ],
        )

        assert planner.validate_workflow(workflow) is False

    def test_validate_workflow_marks_requires_approval(self, planner):
        """Test that validation sets requires_approval based on policy."""
        workflow = planner.plan("send email to test@example.com")
        planner.validate_workflow(workflow)

        # Email tool should require approval
        email_step = workflow.steps[0]
        assert email_step.requires_approval is True


class TestAutonomyModes:
    """Tests for autonomy modes and configuration."""

    def test_autonomy_mode_enum(self):
        """Test AutonomyMode enum values."""
        assert AutonomyMode.OFF.value == "off"
        assert AutonomyMode.SUGGEST.value == "suggest"
        assert AutonomyMode.AUTO.value == "auto"

    def test_create_default_config(self):
        """Test creating default autonomy config."""
        config = create_default_config()

        # Check some defaults
        assert config.get_mode("info.time") == AutonomyMode.AUTO
        assert config.get_mode("email.send") == AutonomyMode.SUGGEST
        assert config.get_mode("os.command") == AutonomyMode.OFF

    def test_autonomy_config_exact_match(self):
        """Test exact category matching."""
        config = AutonomyConfig()
        config.set_mode("email.newsletter", AutonomyMode.AUTO)

        assert config.get_mode("email.newsletter") == AutonomyMode.AUTO

    def test_autonomy_config_wildcard_match(self):
        """Test wildcard category matching."""
        config = AutonomyConfig()
        config.set_mode("email.*", AutonomyMode.SUGGEST)

        assert config.get_mode("email.newsletter") == AutonomyMode.SUGGEST
        assert config.get_mode("email.send") == AutonomyMode.SUGGEST

    def test_autonomy_config_prefix_match(self):
        """Test prefix matching without wildcard."""
        config = AutonomyConfig()
        config.set_mode("email", AutonomyMode.SUGGEST)

        # Exact match should work
        assert config.get_mode("email") == AutonomyMode.SUGGEST

    def test_autonomy_config_default(self):
        """Test default mode for unmatched categories."""
        config = AutonomyConfig(default_mode=AutonomyMode.OFF)

        assert config.get_mode("unknown.category") == AutonomyMode.OFF

    def test_autonomy_config_save_load(self):
        """Test saving and loading config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "autonomy.json"

            config = AutonomyConfig(default_mode=AutonomyMode.SUGGEST)
            config.set_mode("email.*", AutonomyMode.AUTO)
            config.save(config_path)

            loaded = AutonomyConfig.load(config_path)
            assert loaded.default_mode == AutonomyMode.SUGGEST
            assert loaded.get_mode("email.newsletter") == AutonomyMode.AUTO

    def test_infer_category_email(self):
        """Test category inference for email workflows."""
        workflow = Workflow(
            title="send newsletter",
            steps=[
                WorkflowStep(
                    description="Send email",
                    tool_call=ToolCall(tool="send_email", args={}),
                )
            ],
        )

        category = _infer_category(workflow)
        assert category == "email.newsletter"

    def test_infer_category_weather(self):
        """Test category inference for weather workflows."""
        workflow = Workflow(
            title="check weather",
            steps=[
                WorkflowStep(
                    description="Get weather",
                    tool_call=ToolCall(tool="weather_now", args={}),
                )
            ],
        )

        category = _infer_category(workflow)
        assert category == "info.weather"

    def test_infer_category_home(self):
        """Test category inference for home automation."""
        workflow = Workflow(
            title="control lights",
            steps=[
                WorkflowStep(
                    description="Turn on lights",
                    tool_call=ToolCall(tool="home_assistant_call_service", args={}),
                )
            ],
        )

        category = _infer_category(workflow)
        assert category == "home.control"

    def test_get_mode_for_workflow(self):
        """Test getting mode for a workflow."""
        config = create_default_config()

        # Reset and set global config
        from rex.autonomy_modes import set_autonomy_config

        set_autonomy_config(config)

        workflow = Workflow(
            title="check weather",
            steps=[
                WorkflowStep(
                    description="Get weather",
                    tool_call=ToolCall(tool="weather_now", args={}),
                )
            ],
        )

        mode = get_mode(workflow)
        # Weather should be AUTO
        assert mode == AutonomyMode.AUTO


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
