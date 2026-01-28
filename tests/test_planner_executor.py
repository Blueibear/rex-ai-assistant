"""Tests for the planner and executor modules.

This module tests:
- Planner: Goal parsing, workflow generation, validation
- Executor: Budget enforcement, evidence collection, execution flow
- Autonomy modes: Mode detection, configuration
"""

import tempfile
from pathlib import Path

import pytest

from rex.contracts import ToolCall
from rex.planner import Planner, UnableToPlanError
from rex.executor import Executor, ExecutionBudget, ExecutionResult, BudgetExceededError
from rex.autonomy_modes import (
    AutonomyMode,
    AutonomyConfig,
    get_mode,
    create_default_config,
    _infer_category,
)
from rex.workflow import Workflow, WorkflowStep
from rex.policy_engine import PolicyEngine, reset_policy_engine
from rex.tool_registry import ToolRegistry, ToolMeta, reset_tool_registry
from rex.audit import AuditLogger


class TestPlanner:
    """Tests for the Planner class."""

    @pytest.fixture
    def tool_registry(self):
        """Create a tool registry with test tools."""
        reset_tool_registry()
        registry = ToolRegistry()

        # Register test tools
        registry.register_tool(ToolMeta(
            name="time_now",
            description="Get current time",
            required_credentials=[],
            health_check=lambda: (True, "OK"),
        ))
        registry.register_tool(ToolMeta(
            name="weather_now",
            description="Get weather",
            required_credentials=[],
            health_check=lambda: (True, "OK"),
        ))
        registry.register_tool(ToolMeta(
            name="send_email",
            description="Send email",
            required_credentials=[],
            health_check=lambda: (True, "OK"),
        ))
        registry.register_tool(ToolMeta(
            name="web_search",
            description="Search the web",
            required_credentials=[],
            health_check=lambda: (True, "OK"),
        ))
        registry.register_tool(ToolMeta(
            name="home_assistant_call_service",
            description="Control home devices",
            required_credentials=[],
            health_check=lambda: (True, "OK"),
        ))

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
        email_steps = [s for s in workflow.steps if s.tool_call and s.tool_call.tool == "send_email"]
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


class TestExecutor:
    """Tests for the Executor class."""

    @pytest.fixture
    def temp_dirs(self):
        """Create temporary directories for testing."""
        with tempfile.TemporaryDirectory() as workflow_dir, \
             tempfile.TemporaryDirectory() as approval_dir, \
             tempfile.TemporaryDirectory() as log_dir:
            yield {
                "workflow_dir": Path(workflow_dir),
                "approval_dir": Path(approval_dir),
                "log_dir": Path(log_dir),
            }

    @pytest.fixture
    def simple_workflow(self):
        """Create a simple workflow for testing."""
        return Workflow(
            title="Test workflow",
            steps=[
                WorkflowStep(
                    description="Step 1",
                    tool_call=ToolCall(tool="time_now", args={}),
                ),
                WorkflowStep(
                    description="Step 2",
                    tool_call=ToolCall(tool="time_now", args={}),
                ),
            ],
        )

    def test_create_executor(self, simple_workflow):
        """Test creating an executor."""
        executor = Executor(simple_workflow)

        assert executor.workflow == simple_workflow
        assert executor.budget.is_unlimited()

    def test_create_executor_with_dict_budget(self, simple_workflow):
        """Test creating executor with dict budget."""
        budget = {"max_actions": 10, "max_messages": 5, "max_time_seconds": 60}
        executor = Executor(simple_workflow, budget)

        assert executor.budget.max_actions == 10
        assert executor.budget.max_messages == 5
        assert executor.budget.max_time_seconds == 60

    def test_create_executor_with_budget_object(self, simple_workflow):
        """Test creating executor with ExecutionBudget object."""
        budget = ExecutionBudget(max_actions=10, max_messages=5, max_time_seconds=60)
        executor = Executor(simple_workflow, budget)

        assert executor.budget.max_actions == 10

    def test_execution_budget_repr(self):
        """Test ExecutionBudget string representation."""
        budget = ExecutionBudget(max_actions=10)
        assert "actions=10" in repr(budget)

        unlimited = ExecutionBudget()
        assert "unlimited" in repr(unlimited)

    def test_execution_result_str(self, simple_workflow):
        """Test ExecutionResult string formatting."""
        result = ExecutionResult(
            workflow_id="wf_test",
            status="completed",
            actions_taken=2,
            messages_sent=0,
            elapsed_seconds=1.5,
            budget=ExecutionBudget(),
            remaining_budget=ExecutionBudget(),
            summary="Test completed",
        )

        result_str = str(result)
        assert "wf_test" in result_str
        assert "completed" in result_str
        assert "2" in result_str

    def test_collect_evidence(self, simple_workflow, temp_dirs):
        """Test evidence collection from workflow steps."""
        # Mark steps as executed with results
        simple_workflow.steps[0].result = type('obj', (), {
            'success': True,
            'output': {'result': 'ok'},
            'executed_at': None,
        })()

        executor = Executor(
            simple_workflow,
            workflow_dir=temp_dirs["workflow_dir"],
            approval_dir=temp_dirs["approval_dir"],
        )
        executor._collect_evidence()

        assert executor.actions_taken == 1
        assert len(executor.evidence) == 1

    def test_calculate_remaining_budget(self, simple_workflow):
        """Test remaining budget calculation."""
        budget = ExecutionBudget(max_actions=10, max_messages=5, max_time_seconds=60)
        executor = Executor(simple_workflow, budget)
        executor.actions_taken = 3
        executor.messages_sent = 1

        remaining = executor._calculate_remaining_budget(elapsed=10)

        assert remaining.max_actions == 7
        assert remaining.max_messages == 4
        assert remaining.max_time_seconds == 50

    def test_generate_summary(self, simple_workflow, temp_dirs):
        """Test summary generation."""
        executor = Executor(
            simple_workflow,
            workflow_dir=temp_dirs["workflow_dir"],
            approval_dir=temp_dirs["approval_dir"],
        )

        from rex.workflow_runner import RunResult
        run_result = RunResult(
            workflow_id=simple_workflow.workflow_id,
            status="completed",
            steps_executed=2,
            steps_total=2,
            error=None,
            blocking_approval_id=None,
        )

        summary = executor._generate_summary(run_result)

        assert "completed successfully" in summary
        assert "2 of 2 steps" in summary


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
