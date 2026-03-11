"""US-025: Planner task execution.

Acceptance criteria:
- tasks accepted
- task plan generated
- tool calls executed
- Typecheck passes
"""

from unittest.mock import MagicMock, patch

import pytest

from rex.contracts import ToolCall
from rex.executor import Executor, ExecutionBudget, ExecutionResult
from rex.planner import Planner, UnableToPlanError
from rex.policy_engine import PolicyEngine, reset_policy_engine
from rex.tool_registry import ToolMeta, ToolRegistry, reset_tool_registry
from rex.workflow import Workflow, WorkflowStep


@pytest.fixture(autouse=True)
def isolated_registries():
    """Reset global singletons before each test."""
    reset_tool_registry()
    reset_policy_engine()
    yield
    reset_tool_registry()
    reset_policy_engine()


@pytest.fixture
def registry_with_tools():
    """ToolRegistry pre-populated with common test tools."""
    registry = ToolRegistry()
    for name, desc in [
        ("web_search", "Search the web"),
        ("time_now", "Get current time"),
        ("weather_now", "Get weather"),
        ("send_email", "Send email"),
        ("home_assistant_call_service", "Control home devices"),
        ("calendar_create_event", "Create calendar event"),
    ]:
        registry.register_tool(
            ToolMeta(
                name=name,
                description=desc,
                version="1.0",
                enabled=True,
                capabilities=[],
                required_credentials=[],
            )
        )
    return registry


@pytest.fixture
def planner(registry_with_tools):
    return Planner(tool_registry=registry_with_tools, policy_engine=PolicyEngine())


# ---------------------------------------------------------------------------
# Tasks accepted
# ---------------------------------------------------------------------------


class TestTasksAccepted:
    """tasks accepted."""

    def test_plan_accepts_string_goal(self, planner):
        """plan() accepts a plain string goal without raising."""
        workflow = planner.plan("search for open source projects")
        assert workflow is not None

    def test_plan_accepts_goal_with_extra_whitespace(self, planner):
        """plan() strips whitespace and still processes the goal."""
        workflow = planner.plan("  search for python tips  ")
        assert workflow is not None

    def test_plan_accepts_various_known_patterns(self, planner):
        """plan() accepts all supported goal patterns."""
        goals = [
            "search for news",
            "check weather in London",
            "what's the time in Tokyo",
            "turn on bedroom lights",
            "send email to bob@example.com",
        ]
        for goal in goals:
            workflow = planner.plan(goal)
            assert workflow is not None, f"plan() rejected known goal: {goal!r}"

    def test_plan_rejects_empty_goal(self, planner):
        """plan() raises UnableToPlanError for empty goal."""
        with pytest.raises(UnableToPlanError):
            planner.plan("")

    def test_plan_rejects_unknown_goal(self, planner):
        """plan() raises UnableToPlanError for unrecognized goal."""
        with pytest.raises(UnableToPlanError):
            planner.plan("wibble frobnicate xyzzy")


# ---------------------------------------------------------------------------
# Task plan generated
# ---------------------------------------------------------------------------


class TestTaskPlanGenerated:
    """task plan generated."""

    def test_plan_returns_workflow_object(self, planner):
        """plan() returns a Workflow instance."""
        workflow = planner.plan("search for climate data")
        assert isinstance(workflow, Workflow)

    def test_workflow_has_steps(self, planner):
        """Generated workflow has at least one step."""
        workflow = planner.plan("search for climate data")
        assert len(workflow.steps) >= 1

    def test_workflow_steps_have_tool_calls(self, planner):
        """Generated workflow steps contain ToolCall objects."""
        workflow = planner.plan("search for climate data")
        tool_steps = [s for s in workflow.steps if s.tool_call is not None]
        assert len(tool_steps) >= 1

    def test_workflow_title_matches_goal(self, planner):
        """Workflow title equals the goal string."""
        goal = "search for open source AI tools"
        workflow = planner.plan(goal)
        assert workflow.title == goal

    def test_workflow_has_unique_id(self, planner):
        """Each generated workflow has a non-empty unique ID."""
        w1 = planner.plan("search for news")
        w2 = planner.plan("check weather in Paris")
        assert w1.workflow_id
        assert w2.workflow_id
        assert w1.workflow_id != w2.workflow_id

    def test_weather_plan_uses_correct_tool(self, planner):
        """Weather goal generates a step calling weather_now."""
        workflow = planner.plan("check weather in Berlin")
        tools = [s.tool_call.tool for s in workflow.steps if s.tool_call]
        assert "weather_now" in tools

    def test_search_plan_uses_correct_tool(self, planner):
        """Search goal generates a step calling web_search."""
        workflow = planner.plan("search for machine learning papers")
        tools = [s.tool_call.tool for s in workflow.steps if s.tool_call]
        assert "web_search" in tools

    def test_email_plan_uses_correct_tool(self, planner):
        """Email goal generates a step calling send_email."""
        workflow = planner.plan("send email to carol@example.com")
        tools = [s.tool_call.tool for s in workflow.steps if s.tool_call]
        assert "send_email" in tools

    def test_home_control_plan_uses_correct_tool(self, planner):
        """Home control goal generates a step calling home_assistant_call_service."""
        workflow = planner.plan("turn off kitchen lights")
        tools = [s.tool_call.tool for s in workflow.steps if s.tool_call]
        assert "home_assistant_call_service" in tools


# ---------------------------------------------------------------------------
# Tool calls executed
# ---------------------------------------------------------------------------


class TestToolCallsExecuted:
    """tool calls executed."""

    def _make_workflow_with_tool(self, tool_name: str) -> Workflow:
        """Build a minimal workflow with one tool step."""
        return Workflow(
            title="Test execution",
            steps=[
                WorkflowStep(
                    description=f"Call {tool_name}",
                    tool_call=ToolCall(tool=tool_name, args={"query": "test"}),
                )
            ],
        )

    def test_executor_runs_workflow_from_planner(self, planner):
        """Executor.run() executes a planner-generated workflow."""
        workflow = planner.plan("search for test query")

        with patch("rex.workflow_runner.execute_tool", return_value={"result": "ok"}):
            executor = Executor(workflow, budget=ExecutionBudget())
            result = executor.run()

        assert isinstance(result, ExecutionResult)

    def test_executor_reports_actions_taken(self, planner):
        """Executor reports at least one action taken after execution."""
        workflow = planner.plan("search for news")

        with patch("rex.workflow_runner.execute_tool", return_value={"result": "search result"}):
            executor = Executor(workflow, budget=ExecutionBudget())
            result = executor.run()

        assert result.actions_taken >= 1

    def test_executor_status_completed_on_success(self, planner):
        """Executor reports completed status when all steps succeed."""
        workflow = planner.plan("search for facts")

        with patch("rex.workflow_runner.execute_tool", return_value={"result": "ok"}):
            executor = Executor(workflow, budget=ExecutionBudget())
            result = executor.run()

        assert result.status == "completed"

    def test_executor_captures_tool_call_args(self, planner):
        """Tool call args from the plan reach the executor unchanged."""
        workflow = planner.plan("check weather in Amsterdam")
        step = next(s for s in workflow.steps if s.tool_call and s.tool_call.tool == "weather_now")
        assert step.tool_call.args.get("location") == "Amsterdam"

    def test_tool_call_result_stored_on_step(self, planner):
        """After execution, step.result is populated."""
        workflow = planner.plan("search for python docs")

        with patch("rex.workflow_runner.execute_tool", return_value={"result": "python docs found"}):
            executor = Executor(workflow, budget=ExecutionBudget())
            executor.run()

        executed = [s for s in workflow.steps if s.result is not None]
        assert len(executed) >= 1
