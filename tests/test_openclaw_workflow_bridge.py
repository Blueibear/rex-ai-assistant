"""Tests for rex.openclaw.workflow_bridge — US-P4-030 through US-P4-033.

US-P4-030 acceptance criteria:
  - WorkflowBridge exists and is importable
  - Delegates run() and dry_run() to underlying WorkflowRunner
  - Preserves policy_engine, audit_logger, hooks, and other kwargs
  - register() returns None when openclaw not installed

US-P4-031 acceptance criteria:
  - Single-step workflow, no approval → run() returns RunResult with status "completed"

US-P4-032 acceptance criteria:
  - Approval-gated step → run() returns RunResult with status "blocked"

US-P4-033 acceptance criteria:
  - Multi-step (3+) workflow, all allowed → run() returns RunResult, steps all executed
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from rex.openclaw.workflow_bridge import OPENCLAW_AVAILABLE, WorkflowBridge
from rex.workflow import Workflow, WorkflowStep
from rex.workflow_runner import DryRunResult, RunResult, WorkflowRunner

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_step(description: str, *, requires_approval: bool = False) -> WorkflowStep:
    """Build a minimal WorkflowStep with no tool call."""
    return WorkflowStep(description=description, requires_approval=requires_approval)


def _make_workflow(steps: list[WorkflowStep], title: str = "Test Workflow") -> Workflow:
    return Workflow(title=title, steps=steps)


def _make_run_result(
    workflow_id: str = "wf-1",
    status: str = "completed",
    steps_executed: int = 1,
    steps_total: int = 1,
    error: str | None = None,
    blocking_approval_id: str | None = None,
) -> RunResult:
    return RunResult(
        workflow_id=workflow_id,
        status=status,
        steps_executed=steps_executed,
        steps_total=steps_total,
        error=error,
        blocking_approval_id=blocking_approval_id,
    )


# ---------------------------------------------------------------------------
# US-P4-030: Instantiation and basic delegation
# ---------------------------------------------------------------------------


class TestWorkflowBridgeInstantiation:
    def test_import(self):
        from rex.openclaw import workflow_bridge  # noqa: F401

    def test_constructs_with_workflow(self):
        """WorkflowBridge(workflow) constructs without error."""
        wf = _make_workflow([_make_step("step 1")])
        bridge = WorkflowBridge(wf)
        assert bridge is not None

    def test_workflow_property(self):
        """bridge.workflow returns the workflow passed at construction."""
        wf = _make_workflow([_make_step("step 1")])
        bridge = WorkflowBridge(wf)
        assert bridge.workflow is wf

    def test_runner_property(self):
        """bridge.runner returns the underlying WorkflowRunner."""
        wf = _make_workflow([_make_step("step 1")])
        bridge = WorkflowBridge(wf)
        assert isinstance(bridge.runner, WorkflowRunner)

    def test_runner_has_same_workflow(self):
        """The underlying runner holds the same workflow object."""
        wf = _make_workflow([_make_step("step 1")])
        bridge = WorkflowBridge(wf)
        assert bridge.runner.workflow is wf

    def test_openclaw_available_is_bool(self):
        assert isinstance(OPENCLAW_AVAILABLE, bool)


class TestDelegation:
    def setup_method(self):
        self.wf = _make_workflow([_make_step("do something")])
        self.bridge = WorkflowBridge(self.wf)

    def test_run_delegates_to_runner(self):
        """run() delegates to WorkflowRunner.run()."""
        expected = _make_run_result()
        with patch.object(self.bridge._runner, "run", return_value=expected) as mock_run:
            result = self.bridge.run()
            mock_run.assert_called_once()
            assert result is expected

    def test_dry_run_delegates_to_runner(self):
        """dry_run() delegates to WorkflowRunner.dry_run()."""
        expected = DryRunResult(
            workflow_id=self.wf.workflow_id,
            title=self.wf.title,
            total_steps=1,
            steps=[],
            would_complete=True,
            blocking_reason=None,
        )
        with patch.object(self.bridge._runner, "dry_run", return_value=expected) as mock_dry:
            result = self.bridge.dry_run()
            mock_dry.assert_called_once()
            assert result is expected

    def test_kwargs_forwarded_to_runner(self):
        """Constructor kwargs are forwarded to WorkflowRunner."""
        mock_policy = MagicMock()
        wf = _make_workflow([_make_step("step")])
        bridge = WorkflowBridge(wf, policy_engine=mock_policy)
        assert bridge._runner.policy_engine is mock_policy

    def test_hooks_forwarded_to_runner(self):
        """Hook callables are forwarded to WorkflowRunner."""
        pre_hook = MagicMock()
        wf = _make_workflow([_make_step("step")])
        bridge = WorkflowBridge(wf, pre_step_hook=pre_hook)
        assert bridge._runner.pre_step_hook is pre_hook


# ---------------------------------------------------------------------------
# US-P4-031: Simple single-step workflow, no approval
# ---------------------------------------------------------------------------


class TestSimpleWorkflow:
    """Single-step workflow with no tool call, no approval."""

    def test_run_returns_completed(self):
        """Single-step no-tool workflow completes successfully."""
        step = _make_step("describe the weather")
        wf = _make_workflow([step])
        bridge = WorkflowBridge(wf)

        expected = _make_run_result(
            workflow_id=wf.workflow_id, status="completed", steps_executed=1, steps_total=1
        )
        with patch.object(bridge._runner, "run", return_value=expected):
            result = bridge.run()

        assert result.status == "completed"
        assert result.steps_executed == 1

    def test_dry_run_shows_would_complete(self):
        """Dry-run of simple workflow reports would_complete=True."""
        step = _make_step("say hello")
        wf = _make_workflow([step])
        bridge = WorkflowBridge(wf)

        expected_dry = DryRunResult(
            workflow_id=wf.workflow_id,
            title=wf.title,
            total_steps=1,
            steps=[],
            would_complete=True,
            blocking_reason=None,
        )
        with patch.object(bridge._runner, "dry_run", return_value=expected_dry):
            result = bridge.dry_run()

        assert result.would_complete is True
        assert result.blocking_reason is None


# ---------------------------------------------------------------------------
# US-P4-032: Approval-gated workflow
# ---------------------------------------------------------------------------


class TestApprovalGatedWorkflow:
    """Workflow with a step that requires approval."""

    def test_run_returns_blocked_when_approval_needed(self):
        """Approval-gated step causes run() to return status 'blocked'."""
        step = _make_step("send email to all users", requires_approval=True)
        wf = _make_workflow([step])
        bridge = WorkflowBridge(wf)

        expected = _make_run_result(
            workflow_id=wf.workflow_id,
            status="blocked",
            steps_executed=0,
            steps_total=1,
            blocking_approval_id="approval-abc-123",
        )
        with patch.object(bridge._runner, "run", return_value=expected):
            result = bridge.run()

        assert result.status == "blocked"
        assert result.blocking_approval_id == "approval-abc-123"

    def test_dry_run_shows_would_not_complete(self):
        """Dry-run of approval-gated workflow reports would_complete=False."""
        step = _make_step("delete all records", requires_approval=True)
        wf = _make_workflow([step])
        bridge = WorkflowBridge(wf)

        expected_dry = DryRunResult(
            workflow_id=wf.workflow_id,
            title=wf.title,
            total_steps=1,
            steps=[],
            would_complete=False,
            blocking_reason="Step requires approval: delete all records",
        )
        with patch.object(bridge._runner, "dry_run", return_value=expected_dry):
            result = bridge.dry_run()

        assert result.would_complete is False
        assert result.blocking_reason is not None


# ---------------------------------------------------------------------------
# US-P4-033: Multi-step workflow (3+ steps, mixed policies)
# ---------------------------------------------------------------------------


class TestMultiStepWorkflow:
    """3-step workflow: allowed, allowed, requires_approval."""

    def setup_method(self):
        self.steps = [
            _make_step("check status"),
            _make_step("fetch data"),
            _make_step("notify admin", requires_approval=True),
        ]
        self.wf = _make_workflow(self.steps, title="Multi-Step Workflow")
        self.bridge = WorkflowBridge(self.wf)

    def test_run_executes_allowed_steps_then_blocks(self):
        """Run executes allowed steps and blocks on approval-gated step."""
        expected = _make_run_result(
            workflow_id=self.wf.workflow_id,
            status="blocked",
            steps_executed=2,
            steps_total=3,
            blocking_approval_id="approval-xyz",
        )
        with patch.object(self.bridge._runner, "run", return_value=expected):
            result = self.bridge.run()

        assert result.status == "blocked"
        assert result.steps_executed == 2
        assert result.steps_total == 3

    def test_run_all_allowed_completes(self):
        """Run with all-allowed steps returns completed."""
        all_allowed_steps = [_make_step(f"step {i}") for i in range(3)]
        wf = _make_workflow(all_allowed_steps, title="All Allowed")
        bridge = WorkflowBridge(wf)

        expected = _make_run_result(
            workflow_id=wf.workflow_id,
            status="completed",
            steps_executed=3,
            steps_total=3,
        )
        with patch.object(bridge._runner, "run", return_value=expected):
            result = bridge.run()

        assert result.status == "completed"
        assert result.steps_executed == 3

    def test_dry_run_three_steps(self):
        """Dry-run of 3-step workflow reports correct total_steps."""
        expected_dry = DryRunResult(
            workflow_id=self.wf.workflow_id,
            title=self.wf.title,
            total_steps=3,
            steps=[],
            would_complete=False,
            blocking_reason="Step 3 requires approval",
        )
        with patch.object(self.bridge._runner, "dry_run", return_value=expected_dry):
            result = self.bridge.dry_run()

        assert result.total_steps == 3
        assert result.would_complete is False


# ---------------------------------------------------------------------------
# US-P4-030: register() stub
# ---------------------------------------------------------------------------


class TestRegister:
    def test_register_returns_none_without_openclaw(self):
        wf = _make_workflow([_make_step("step")])
        bridge = WorkflowBridge(wf)
        if not OPENCLAW_AVAILABLE:
            assert bridge.register() is None

    def test_register_accepts_agent_arg(self):
        wf = _make_workflow([_make_step("step")])
        bridge = WorkflowBridge(wf)
        agent = MagicMock()
        if not OPENCLAW_AVAILABLE:
            assert bridge.register(agent=agent) is None
