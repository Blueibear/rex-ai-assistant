"""US-024: Planner initialization.

Acceptance criteria:
- planner loads successfully
- dependencies resolved
- planner callable
- Typecheck passes
"""

import pytest

from rex.planner import InvalidWorkflowError, Planner, PlannerError, UnableToPlanError
from rex.policy_engine import PolicyEngine, reset_policy_engine
from rex.tool_registry import ToolMeta, ToolRegistry, reset_tool_registry


@pytest.fixture(autouse=True)
def isolated_registries():
    """Reset global singletons before each test."""
    reset_tool_registry()
    reset_policy_engine()
    yield
    reset_tool_registry()
    reset_policy_engine()


class TestPlannerLoadsSuccessfully:
    """planner loads successfully."""

    def test_planner_import(self):
        """Planner can be imported from rex.planner."""
        from rex.planner import Planner  # noqa: F401 — import is the test

        assert Planner is not None

    def test_planner_instantiates_with_defaults(self):
        """Planner instantiates without arguments using global defaults."""
        planner = Planner()
        assert planner is not None

    def test_planner_has_tool_registry(self):
        """Planner stores a tool registry after initialization."""
        planner = Planner()
        assert planner.tool_registry is not None

    def test_planner_has_policy_engine(self):
        """Planner stores a policy engine after initialization."""
        planner = Planner()
        assert planner.policy_engine is not None


class TestPlannerDependenciesResolved:
    """dependencies resolved."""

    def test_custom_tool_registry_accepted(self):
        """Planner accepts a custom ToolRegistry instance."""
        registry = ToolRegistry()
        planner = Planner(tool_registry=registry)
        assert planner.tool_registry is registry

    def test_custom_policy_engine_accepted(self):
        """Planner accepts a custom PolicyEngine instance."""
        policy_engine = PolicyEngine()
        planner = Planner(policy_engine=policy_engine)
        assert planner.policy_engine is policy_engine

    def test_both_dependencies_injectable(self):
        """Planner accepts both custom registry and policy engine together."""
        registry = ToolRegistry()
        policy_engine = PolicyEngine()
        planner = Planner(tool_registry=registry, policy_engine=policy_engine)
        assert planner.tool_registry is registry
        assert planner.policy_engine is policy_engine

    def test_exception_hierarchy(self):
        """PlannerError, UnableToPlanError, InvalidWorkflowError are importable."""
        assert issubclass(UnableToPlanError, PlannerError)
        assert issubclass(InvalidWorkflowError, PlannerError)
        assert issubclass(PlannerError, Exception)

    def test_planner_rules_initialized(self):
        """Planner rule list is populated after initialization."""
        planner = Planner()
        assert len(planner._rules) > 0


class TestPlannerCallable:
    """planner callable."""

    def test_plan_raises_on_empty_goal(self):
        """plan() raises UnableToPlanError for empty goal."""
        planner = Planner()
        with pytest.raises(UnableToPlanError):
            planner.plan("")

    def test_plan_raises_on_whitespace_goal(self):
        """plan() raises UnableToPlanError for whitespace-only goal."""
        planner = Planner()
        with pytest.raises(UnableToPlanError):
            planner.plan("   ")

    def test_plan_raises_on_unknown_goal(self):
        """plan() raises UnableToPlanError when no rule matches."""
        planner = Planner()
        with pytest.raises(UnableToPlanError):
            planner.plan("xyzzy frobnicate the wumpus")

    def test_plan_returns_workflow_for_known_goal(self):
        """plan() returns a Workflow when a goal matches a rule."""
        registry = ToolRegistry()
        registry.register_tool(
            ToolMeta(
                name="web_search",
                description="Search the web",
                version="1.0",
                enabled=True,
                capabilities=["search"],
                required_credentials=[],
            )
        )
        planner = Planner(tool_registry=registry)
        workflow = planner.plan("search for python tutorials")
        assert workflow is not None
        assert len(workflow.steps) >= 1

    def test_validate_workflow_callable(self):
        """validate_workflow() is callable and returns a bool."""
        registry = ToolRegistry()
        registry.register_tool(
            ToolMeta(
                name="web_search",
                description="Search the web",
                version="1.0",
                enabled=True,
                capabilities=["search"],
                required_credentials=[],
            )
        )
        planner = Planner(tool_registry=registry)
        workflow = planner.plan("search for news")
        result = planner.validate_workflow(workflow)
        assert isinstance(result, bool)

    def test_plan_goal_stored_as_workflow_title(self):
        """Workflow title matches the provided goal."""
        registry = ToolRegistry()
        registry.register_tool(
            ToolMeta(
                name="web_search",
                description="Search the web",
                version="1.0",
                enabled=True,
                capabilities=["search"],
                required_credentials=[],
            )
        )
        planner = Planner(tool_registry=registry)
        goal = "search for weather data"
        workflow = planner.plan(goal)
        assert workflow.title == goal
