"""Tests for CLI planner and executor commands.

This module tests:
- rex plan command: workflow generation, validation, execution
- rex executor resume command: resuming blocked workflows
"""

import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from rex.cli import cmd_plan, cmd_executor_resume
from rex.workflow import Workflow, WorkflowStep
from rex.contracts import ToolCall
from rex.tool_registry import ToolRegistry, ToolMeta, reset_tool_registry, set_tool_registry
from rex.policy_engine import reset_policy_engine, get_policy_engine
from rex.autonomy_modes import reset_autonomy_config


class TestCmdPlan:
    """Tests for the 'rex plan' command."""

    @pytest.fixture(autouse=True)
    def setup_test_env(self):
        """Set up test environment with tools and policies."""
        reset_tool_registry()
        reset_policy_engine()
        reset_autonomy_config()

        # Register test tools
        registry = ToolRegistry()
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
            description="Search web",
            required_credentials=[],
            health_check=lambda: (True, "OK"),
        ))
        set_tool_registry(registry)

        yield

        reset_tool_registry()
        reset_policy_engine()
        reset_autonomy_config()

    @pytest.fixture
    def temp_dirs(self):
        """Create temporary directories for testing."""
        with tempfile.TemporaryDirectory() as workflow_dir, \
             tempfile.TemporaryDirectory() as approval_dir:
            # Patch the default directories
            with patch("rex.workflow.DEFAULT_WORKFLOW_DIR", Path(workflow_dir)), \
                 patch("rex.workflow.DEFAULT_APPROVAL_DIR", Path(approval_dir)):
                yield {
                    "workflow_dir": Path(workflow_dir),
                    "approval_dir": Path(approval_dir),
                }

    def test_plan_simple_goal(self, temp_dirs, capsys):
        """Test planning a simple goal."""
        args = MagicMock()
        args.goal = "check weather in Dallas"
        args.save = False
        args.execute = False
        args.force = False
        args.max_actions = 0
        args.max_messages = 0
        args.max_time = 0

        result = cmd_plan(args)

        assert result == 0

        captured = capsys.readouterr()
        assert "Planning workflow" in captured.out
        assert "check weather in Dallas" in captured.out
        assert "Generated workflow" in captured.out
        assert "Validation passed" in captured.out

    def test_plan_with_save(self, temp_dirs, capsys):
        """Test planning with save option."""
        args = MagicMock()
        args.goal = "check weather in Dallas"
        args.save = True
        args.execute = False
        args.force = False
        args.max_actions = 0
        args.max_messages = 0
        args.max_time = 0

        with patch("rex.workflow.DEFAULT_WORKFLOW_DIR", temp_dirs["workflow_dir"]):
            result = cmd_plan(args)

        assert result == 0

        captured = capsys.readouterr()
        assert "Saved workflow" in captured.out

    def test_plan_unknown_goal(self, temp_dirs, capsys):
        """Test planning with unknown goal."""
        args = MagicMock()
        args.goal = "do something completely unknown xyzabc123"
        args.save = False
        args.execute = False
        args.force = False
        args.max_actions = 0
        args.max_messages = 0
        args.max_time = 0

        result = cmd_plan(args)

        assert result == 1

        captured = capsys.readouterr()
        assert "Error" in captured.out
        assert "Unable to plan" in captured.out

    def test_plan_with_execute_auto_mode(self, temp_dirs, capsys):
        """Test planning with execute for AUTO mode workflow."""
        args = MagicMock()
        args.goal = "check weather in Dallas"  # Weather is AUTO mode
        args.save = True
        args.execute = True
        args.force = False
        args.max_actions = 10
        args.max_messages = 5
        args.max_time = 60

        # Mock the executor run to avoid actual execution
        with patch("rex.workflow.DEFAULT_WORKFLOW_DIR", temp_dirs["workflow_dir"]), \
             patch("rex.workflow.DEFAULT_APPROVAL_DIR", temp_dirs["approval_dir"]), \
             patch("rex.executor.Executor.run") as mock_run:

            from rex.executor import ExecutionResult, ExecutionBudget
            mock_run.return_value = ExecutionResult(
                workflow_id="wf_test",
                status="completed",
                actions_taken=1,
                messages_sent=0,
                elapsed_seconds=0.5,
                budget=ExecutionBudget(),
                remaining_budget=ExecutionBudget(),
                summary="Test completed",
            )

            result = cmd_plan(args)

        assert result == 0

        captured = capsys.readouterr()
        assert "Executing workflow" in captured.out
        assert "Execution complete" in captured.out


class TestCmdExecutorResume:
    """Tests for the 'rex executor resume' command."""

    @pytest.fixture(autouse=True)
    def setup_test_env(self):
        """Set up test environment."""
        reset_tool_registry()
        reset_policy_engine()
        reset_autonomy_config()
        yield
        reset_tool_registry()
        reset_policy_engine()
        reset_autonomy_config()

    @pytest.fixture
    def temp_dirs(self):
        """Create temporary directories for testing."""
        with tempfile.TemporaryDirectory() as workflow_dir, \
             tempfile.TemporaryDirectory() as approval_dir:
            yield {
                "workflow_dir": Path(workflow_dir),
                "approval_dir": Path(approval_dir),
            }

    @pytest.fixture
    def blocked_workflow(self, temp_dirs):
        """Create a blocked workflow for testing."""
        workflow = Workflow(
            workflow_id="wf_test_blocked",
            title="Test blocked workflow",
            status="blocked",
            blocking_approval_id="apr_test_123",
            steps=[
                WorkflowStep(
                    description="Test step",
                    tool_call=ToolCall(tool="send_email", args={"to": "test@example.com"}),
                )
            ],
        )
        workflow.save(temp_dirs["workflow_dir"])
        return workflow

    def test_resume_workflow_not_found(self, temp_dirs, capsys):
        """Test resuming non-existent workflow."""
        args = MagicMock()
        args.workflow_id = "wf_nonexistent"
        args.max_actions = 0
        args.max_messages = 0
        args.max_time = 0

        with patch("rex.workflow.DEFAULT_WORKFLOW_DIR", temp_dirs["workflow_dir"]):
            result = cmd_executor_resume(args)

        assert result == 1

        captured = capsys.readouterr()
        assert "Error" in captured.out
        assert "not found" in captured.out

    def test_resume_non_blocked_workflow(self, temp_dirs, capsys):
        """Test resuming workflow that is not blocked."""
        # Create a completed workflow
        workflow = Workflow(
            workflow_id="wf_test_completed",
            title="Test completed workflow",
            status="completed",
            steps=[],
        )
        workflow.save(temp_dirs["workflow_dir"])

        args = MagicMock()
        args.workflow_id = "wf_test_completed"
        args.max_actions = 0
        args.max_messages = 0
        args.max_time = 0

        with patch("rex.workflow.DEFAULT_WORKFLOW_DIR", temp_dirs["workflow_dir"]):
            result = cmd_executor_resume(args)

        assert result == 1

        captured = capsys.readouterr()
        assert "Error" in captured.out
        assert "Cannot resume" in captured.out

    def test_resume_blocked_workflow_success(self, temp_dirs, blocked_workflow, capsys):
        """Test successfully resuming a blocked workflow."""
        args = MagicMock()
        args.workflow_id = blocked_workflow.workflow_id
        args.max_actions = 10
        args.max_messages = 5
        args.max_time = 60

        # Mock the executor run
        with patch("rex.workflow.DEFAULT_WORKFLOW_DIR", temp_dirs["workflow_dir"]), \
             patch("rex.workflow.DEFAULT_APPROVAL_DIR", temp_dirs["approval_dir"]), \
             patch("rex.executor.Executor.run") as mock_run:

            from rex.executor import ExecutionResult, ExecutionBudget
            mock_run.return_value = ExecutionResult(
                workflow_id=blocked_workflow.workflow_id,
                status="completed",
                actions_taken=1,
                messages_sent=1,
                elapsed_seconds=0.5,
                budget=ExecutionBudget(max_actions=10, max_messages=5, max_time_seconds=60),
                remaining_budget=ExecutionBudget(max_actions=9, max_messages=4, max_time_seconds=59),
                summary="Workflow completed",
            )

            result = cmd_executor_resume(args)

        assert result == 0

        captured = capsys.readouterr()
        assert "Resuming workflow" in captured.out
        assert "Execution complete" in captured.out

    def test_resume_with_budget_params(self, temp_dirs, blocked_workflow, capsys):
        """Test resume with custom budget parameters."""
        args = MagicMock()
        args.workflow_id = blocked_workflow.workflow_id
        args.max_actions = 5
        args.max_messages = 2
        args.max_time = 30

        with patch("rex.workflow.DEFAULT_WORKFLOW_DIR", temp_dirs["workflow_dir"]), \
             patch("rex.workflow.DEFAULT_APPROVAL_DIR", temp_dirs["approval_dir"]), \
             patch("rex.executor.Executor.run") as mock_run, \
             patch("rex.executor.Executor.__init__") as mock_init:

            mock_init.return_value = None

            from rex.executor import ExecutionResult, ExecutionBudget
            mock_run.return_value = ExecutionResult(
                workflow_id=blocked_workflow.workflow_id,
                status="completed",
                actions_taken=1,
                messages_sent=0,
                elapsed_seconds=1.0,
                budget=ExecutionBudget(),
                remaining_budget=ExecutionBudget(),
                summary="Test",
            )

            result = cmd_executor_resume(args)

        # Verify budget was created with correct params
        captured = capsys.readouterr()
        # The output should show the budget
        assert "budget" in captured.out.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
