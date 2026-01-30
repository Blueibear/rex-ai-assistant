"""Tests for the workflow engine.

This module tests the workflow primitives and runner:
- WorkflowStep and Workflow models
- Precondition and postcondition logic
- Policy engine integration (denial, approval, auto-execution)
- Idempotency (skipping repeated steps)
- Persistence and resumption via JSON files
"""

import tempfile

import pytest

from rex.audit import AuditLogger
from rex.contracts import RiskLevel, ToolCall
from rex.policy import ActionPolicy
from rex.policy_engine import PolicyEngine
from rex.workflow import (
    StepResult,
    Workflow,
    WorkflowApproval,
    WorkflowStep,
    clear_condition_registry,
    generate_approval_id,
    generate_step_id,
    generate_workflow_id,
    get_condition,
    register_condition,
)
from rex.workflow_runner import (
    WorkflowRunner,
    approve_workflow,
    deny_workflow,
    list_pending_approvals,
)


class TestWorkflowStep:
    """Tests for the WorkflowStep model."""

    def test_create_step_with_minimal_fields(self):
        """Test creating a step with minimal required fields."""
        step = WorkflowStep(description="Test step")
        assert step.description == "Test step"
        assert step.step_id.startswith("step_")
        assert step.tool_call is None
        assert step.precondition is None
        assert step.postcondition is None
        assert step.result is None

    def test_create_step_with_tool_call(self):
        """Test creating a step with a tool call."""
        tool_call = ToolCall(tool="time_now", args={"location": "Dallas, TX"})
        step = WorkflowStep(
            step_id="step_001",
            description="Get current time",
            tool_call=tool_call,
        )
        assert step.step_id == "step_001"
        assert step.tool_call.tool == "time_now"
        assert step.tool_call.args == {"location": "Dallas, TX"}

    def test_create_step_with_conditions(self):
        """Test creating a step with precondition and postcondition."""
        step = WorkflowStep(
            step_id="step_002",
            description="Conditional step",
            precondition="check_time_available",
            postcondition="verify_time_received",
        )
        assert step.precondition == "check_time_available"
        assert step.postcondition == "verify_time_received"

    def test_create_step_with_idempotency_key(self):
        """Test creating a step with idempotency key."""
        step = WorkflowStep(
            step_id="step_003",
            description="Idempotent step",
            idempotency_key="get_time_dallas_2024",
        )
        assert step.idempotency_key == "get_time_dallas_2024"

    def test_step_json_roundtrip(self):
        """Test serialization and deserialization of a step."""
        tool_call = ToolCall(tool="time_now", args={"location": "Dallas, TX"})
        step = WorkflowStep(
            step_id="step_001",
            description="Get time",
            tool_call=tool_call,
            precondition="always_true",
            idempotency_key="get_time_1",
        )

        json_str = step.model_dump_json()
        restored = WorkflowStep.model_validate_json(json_str)

        assert restored.step_id == step.step_id
        assert restored.description == step.description
        assert restored.tool_call.tool == step.tool_call.tool
        assert restored.precondition == step.precondition
        assert restored.idempotency_key == step.idempotency_key


class TestWorkflow:
    """Tests for the Workflow model."""

    def test_create_workflow_with_minimal_fields(self):
        """Test creating a workflow with minimal required fields."""
        wf = Workflow(title="Test Workflow")
        assert wf.title == "Test Workflow"
        assert wf.workflow_id.startswith("wf_")
        assert wf.status == "queued"
        assert wf.steps == []
        assert wf.current_step_index == 0
        assert wf.state == {}

    def test_create_workflow_with_steps(self):
        """Test creating a workflow with steps."""
        steps = [
            WorkflowStep(step_id="s1", description="Step 1"),
            WorkflowStep(step_id="s2", description="Step 2"),
        ]
        wf = Workflow(
            workflow_id="wf_test",
            title="Multi-step workflow",
            steps=steps,
        )
        assert len(wf.steps) == 2
        assert wf.steps[0].step_id == "s1"
        assert wf.steps[1].step_id == "s2"

    def test_workflow_is_finished(self):
        """Test the is_finished method."""
        wf = Workflow(title="Test")
        assert not wf.is_finished()

        wf.status = "completed"
        assert wf.is_finished()

        wf.status = "failed"
        assert wf.is_finished()

        wf.status = "canceled"
        assert wf.is_finished()

        wf.status = "running"
        assert not wf.is_finished()

    def test_workflow_is_blocked(self):
        """Test the is_blocked method."""
        wf = Workflow(title="Test")
        assert not wf.is_blocked()

        wf.status = "blocked"
        assert wf.is_blocked()

    def test_workflow_current_step(self):
        """Test getting the current step."""
        steps = [
            WorkflowStep(step_id="s1", description="Step 1"),
            WorkflowStep(step_id="s2", description="Step 2"),
        ]
        wf = Workflow(title="Test", steps=steps)

        assert wf.current_step().step_id == "s1"

        wf.current_step_index = 1
        assert wf.current_step().step_id == "s2"

        wf.current_step_index = 2
        assert wf.current_step() is None

    def test_workflow_advance(self):
        """Test advancing to the next step."""
        steps = [
            WorkflowStep(step_id="s1", description="Step 1"),
            WorkflowStep(step_id="s2", description="Step 2"),
        ]
        wf = Workflow(title="Test", steps=steps)

        assert wf.current_step_index == 0
        assert wf.advance() is True
        assert wf.current_step_index == 1

        assert wf.advance() is False
        assert wf.status == "completed"

    def test_workflow_mark_methods(self):
        """Test status marking methods."""
        wf = Workflow(title="Test")

        wf.mark_running()
        assert wf.status == "running"

        wf.mark_blocked("apr_123")
        assert wf.status == "blocked"
        assert wf.blocking_approval_id == "apr_123"

        wf.mark_failed("Something went wrong")
        assert wf.status == "failed"
        assert wf.error == "Something went wrong"
        assert wf.completed_at is not None

    def test_workflow_get_executed_idempotency_keys(self):
        """Test getting executed idempotency keys."""
        steps = [
            WorkflowStep(
                step_id="s1",
                description="Step 1",
                idempotency_key="key1",
                result=StepResult(step_id="s1", success=True),
            ),
            WorkflowStep(
                step_id="s2",
                description="Step 2",
                idempotency_key="key2",
                result=StepResult(step_id="s2", success=False),
            ),
            WorkflowStep(
                step_id="s3",
                description="Step 3",
                idempotency_key="key3",
            ),
        ]
        wf = Workflow(title="Test", steps=steps)

        keys = wf.get_executed_idempotency_keys()
        assert "key1" in keys
        assert "key2" not in keys  # Failed step
        assert "key3" not in keys  # No result yet

    def test_workflow_save_and_load(self):
        """Test saving and loading workflow from disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            steps = [
                WorkflowStep(step_id="s1", description="Step 1"),
            ]
            wf = Workflow(
                workflow_id="wf_save_test",
                title="Save Test",
                steps=steps,
                state={"key": "value"},
            )

            path = wf.save(tmpdir)
            assert path.exists()

            loaded = Workflow.load("wf_save_test", tmpdir)
            assert loaded is not None
            assert loaded.workflow_id == wf.workflow_id
            assert loaded.title == wf.title
            assert len(loaded.steps) == 1
            assert loaded.state == {"key": "value"}

    def test_workflow_load_from_file(self):
        """Test loading workflow from a specific file path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            wf = Workflow(
                workflow_id="wf_file_test",
                title="File Test",
            )
            path = wf.save(tmpdir)

            loaded = Workflow.load_from_file(path)
            assert loaded.workflow_id == wf.workflow_id
            assert loaded.title == wf.title

    def test_workflow_load_from_file_not_found(self):
        """Test loading workflow from non-existent file."""
        with pytest.raises(FileNotFoundError):
            Workflow.load_from_file("/nonexistent/path/workflow.json")

    def test_workflow_list_workflows(self):
        """Test listing workflows."""
        with tempfile.TemporaryDirectory() as tmpdir:
            wf1 = Workflow(workflow_id="wf_1", title="First", status="completed")
            wf2 = Workflow(workflow_id="wf_2", title="Second", status="queued")
            wf1.save(tmpdir)
            wf2.save(tmpdir)

            workflows = Workflow.list_workflows(tmpdir)
            assert len(workflows) == 2

            # Filter by status
            completed = Workflow.list_workflows(tmpdir, status="completed")
            assert len(completed) == 1
            assert completed[0].workflow_id == "wf_1"

    def test_workflow_json_roundtrip(self):
        """Test full JSON serialization roundtrip."""
        steps = [
            WorkflowStep(
                step_id="s1",
                description="Step 1",
                tool_call=ToolCall(tool="time_now", args={}),
            )
        ]
        wf = Workflow(
            workflow_id="wf_round",
            title="Roundtrip Test",
            steps=steps,
            state={"counter": 42},
        )

        json_str = wf.model_dump_json()
        restored = Workflow.model_validate_json(json_str)

        assert restored.workflow_id == wf.workflow_id
        assert restored.title == wf.title
        assert len(restored.steps) == 1
        assert restored.state == {"counter": 42}


class TestWorkflowApproval:
    """Tests for the WorkflowApproval model."""

    def test_create_approval(self):
        """Test creating an approval."""
        approval = WorkflowApproval(
            workflow_id="wf_001",
            step_id="step_001",
            step_description="Send important email",
        )
        assert approval.approval_id.startswith("apr_")
        assert approval.status == "pending"
        assert approval.workflow_id == "wf_001"
        assert approval.step_id == "step_001"

    def test_approval_save_and_load(self):
        """Test saving and loading approval from disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            approval = WorkflowApproval(
                approval_id="apr_test",
                workflow_id="wf_001",
                step_id="step_001",
                step_description="Test step",
            )

            path = approval.save(tmpdir)
            assert path.exists()

            loaded = WorkflowApproval.load("apr_test", tmpdir)
            assert loaded is not None
            assert loaded.approval_id == "apr_test"
            assert loaded.status == "pending"


class TestConditionRegistry:
    """Tests for the condition registry."""

    def setup_method(self):
        """Clear condition registry before each test."""
        clear_condition_registry()
        # Re-register built-in conditions
        register_condition("always_true", lambda s: True)
        register_condition("always_false", lambda s: False)

    def test_register_and_get_condition(self):
        """Test registering and retrieving conditions."""

        def my_condition(state):
            return state.get("ready", False)

        register_condition("my_condition", my_condition)
        retrieved = get_condition("my_condition")
        assert retrieved is not None
        assert retrieved({"ready": True}) is True
        assert retrieved({"ready": False}) is False

    def test_get_nonexistent_condition(self):
        """Test getting a condition that doesn't exist."""
        assert get_condition("nonexistent") is None

    def test_builtin_conditions(self):
        """Test built-in conditions."""
        always_true = get_condition("always_true")
        always_false = get_condition("always_false")

        assert always_true({}) is True
        assert always_false({}) is False


class TestWorkflowRunner:
    """Tests for the WorkflowRunner class."""

    def setup_method(self):
        """Set up test fixtures."""
        clear_condition_registry()
        register_condition("always_true", lambda s: True)
        register_condition("always_false", lambda s: False)

    def test_run_empty_workflow(self):
        """Test running a workflow with no steps."""
        with tempfile.TemporaryDirectory() as tmpdir:
            wf = Workflow(title="Empty")
            runner = WorkflowRunner(wf, workflow_dir=tmpdir, approval_dir=tmpdir)
            result = runner.run()

            assert result.status == "completed"
            assert result.steps_executed == 0
            assert result.steps_total == 0

    def test_run_workflow_with_auto_executed_step(self):
        """Test running a workflow with auto-executed (low-risk) step."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create workflow with low-risk tool
            steps = [
                WorkflowStep(
                    step_id="s1",
                    description="Get time",
                    tool_call=ToolCall(tool="time_now", args={"location": "Dallas, TX"}),
                )
            ]
            wf = Workflow(title="Auto Execute", steps=steps)

            # Use default policy engine (time_now is auto-execute)
            audit_logger = AuditLogger(log_dir=tmpdir)
            runner = WorkflowRunner(
                wf,
                workflow_dir=tmpdir,
                approval_dir=tmpdir,
                audit_logger=audit_logger,
                default_context={"timezone": "America/Chicago"},
            )
            result = runner.run()

            assert result.status == "completed"
            assert result.steps_executed == 1
            assert wf.steps[0].result is not None
            assert wf.steps[0].result.success is True

    def test_run_workflow_step_requires_approval(self):
        """Test that medium-risk steps block for approval."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create workflow with medium-risk tool
            steps = [
                WorkflowStep(
                    step_id="s1",
                    description="Send email",
                    tool_call=ToolCall(tool="send_email", args={"to": "user@example.com"}),
                )
            ]
            wf = Workflow(title="Approval Required", steps=steps)

            runner = WorkflowRunner(wf, workflow_dir=tmpdir, approval_dir=tmpdir)
            result = runner.run()

            assert result.status == "blocked"
            assert result.blocking_approval_id is not None
            assert wf.status == "blocked"

            # Check approval was created
            approval = WorkflowApproval.load(result.blocking_approval_id, tmpdir)
            assert approval is not None
            assert approval.status == "pending"

    def test_run_workflow_step_denied(self):
        """Test that denied steps fail the workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create policy that denies a specific recipient
            policy = ActionPolicy(
                tool_name="send_email",
                risk=RiskLevel.MEDIUM,
                denied_recipients=["blocked@example.com"],
            )
            engine = PolicyEngine(policies=[policy])

            steps = [
                WorkflowStep(
                    step_id="s1",
                    description="Send to blocked recipient",
                    tool_call=ToolCall(
                        tool="send_email",
                        args={"to": "blocked@example.com"},
                    ),
                )
            ]
            wf = Workflow(title="Denial Test", steps=steps)

            runner = WorkflowRunner(
                wf,
                workflow_dir=tmpdir,
                approval_dir=tmpdir,
                policy_engine=engine,
            )
            result = runner.run()

            assert result.status == "failed"
            assert wf.steps[0].result is not None
            assert wf.steps[0].result.success is False
            assert "denied" in wf.steps[0].result.error.lower()

    def test_run_workflow_precondition_skip(self):
        """Test that precondition returning False skips the step."""
        with tempfile.TemporaryDirectory() as tmpdir:
            steps = [
                WorkflowStep(
                    step_id="s1",
                    description="Skipped step",
                    precondition="always_false",
                    tool_call=ToolCall(tool="time_now", args={}),
                ),
                WorkflowStep(
                    step_id="s2",
                    description="Executed step",
                    tool_call=ToolCall(tool="time_now", args={"location": "Dallas, TX"}),
                ),
            ]
            wf = Workflow(title="Precondition Test", steps=steps)

            runner = WorkflowRunner(
                wf,
                workflow_dir=tmpdir,
                approval_dir=tmpdir,
                default_context={"timezone": "America/Chicago"},
            )
            result = runner.run()

            assert result.status == "completed"
            assert wf.steps[0].result.skipped is True
            assert wf.steps[1].result.success is True

    def test_run_workflow_postcondition_fail(self):
        """Test that postcondition returning False fails the step."""
        with tempfile.TemporaryDirectory() as tmpdir:
            steps = [
                WorkflowStep(
                    step_id="s1",
                    description="Step with failing postcondition",
                    postcondition="always_false",
                    tool_call=ToolCall(tool="time_now", args={"location": "Dallas, TX"}),
                ),
            ]
            wf = Workflow(title="Postcondition Test", steps=steps)

            runner = WorkflowRunner(
                wf,
                workflow_dir=tmpdir,
                approval_dir=tmpdir,
                default_context={"timezone": "America/Chicago"},
            )
            result = runner.run()

            assert result.status == "failed"
            assert wf.steps[0].result.success is False
            assert "postcondition" in wf.steps[0].result.error.lower()

    def test_run_workflow_idempotency_skip(self):
        """Test that steps with executed idempotency keys are skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create workflow with step that has already executed result
            steps = [
                WorkflowStep(
                    step_id="s1",
                    description="Already executed",
                    idempotency_key="unique_key_1",
                    tool_call=ToolCall(tool="time_now", args={}),
                    result=StepResult(step_id="s1", success=True),
                ),
                WorkflowStep(
                    step_id="s2",
                    description="Not executed yet",
                    idempotency_key="unique_key_1",
                    tool_call=ToolCall(tool="time_now", args={"location": "Dallas, TX"}),
                ),
            ]
            wf = Workflow(title="Idempotency Test", steps=steps, current_step_index=1)

            runner = WorkflowRunner(
                wf,
                workflow_dir=tmpdir,
                approval_dir=tmpdir,
                default_context={"timezone": "America/Chicago"},
            )
            result = runner.run()

            assert result.status == "completed"
            # Second step should be skipped due to idempotency
            assert wf.steps[1].result.skipped is True
            assert "idempotency" in wf.steps[1].result.skip_reason.lower()

    def test_resume_blocked_workflow(self):
        """Test resuming a workflow blocked on approval."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # First, create a blocked workflow
            steps = [
                WorkflowStep(
                    step_id="s1",
                    description="Needs approval",
                    tool_call=ToolCall(tool="send_email", args={"to": "user@example.com"}),
                ),
            ]
            wf = Workflow(title="Resume Test", steps=steps)

            runner = WorkflowRunner(wf, workflow_dir=tmpdir, approval_dir=tmpdir)
            result1 = runner.run()

            assert result1.status == "blocked"
            approval_id = result1.blocking_approval_id

            # Approve the request
            assert approve_workflow(approval_id, approval_dir=tmpdir)

            # Reload workflow and resume
            wf_reloaded = Workflow.load(wf.workflow_id, tmpdir)
            runner2 = WorkflowRunner(wf_reloaded, workflow_dir=tmpdir, approval_dir=tmpdir)

            # Note: The send_email tool is not implemented, so it will error
            # but the test verifies the resume logic works
            result2 = runner2.resume()

            # Workflow should have attempted to continue (may fail because tool isn't implemented)
            assert result2.status in ("failed", "completed")

    def test_resume_with_denied_approval(self):
        """Test resuming a workflow with denied approval."""
        with tempfile.TemporaryDirectory() as tmpdir:
            steps = [
                WorkflowStep(
                    step_id="s1",
                    description="Needs approval",
                    tool_call=ToolCall(tool="send_email", args={"to": "user@example.com"}),
                ),
            ]
            wf = Workflow(title="Deny Test", steps=steps)

            runner = WorkflowRunner(wf, workflow_dir=tmpdir, approval_dir=tmpdir)
            result1 = runner.run()

            assert result1.status == "blocked"

            # Deny the request
            assert deny_workflow(
                result1.blocking_approval_id,
                reason="Not authorized",
                approval_dir=tmpdir,
            )

            # Resume
            wf_reloaded = Workflow.load(wf.workflow_id, tmpdir)
            runner2 = WorkflowRunner(wf_reloaded, workflow_dir=tmpdir, approval_dir=tmpdir)
            result2 = runner2.resume()

            assert result2.status == "failed"
            assert "denied" in wf_reloaded.error.lower()

    def test_dry_run(self):
        """Test dry-run mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            steps = [
                WorkflowStep(
                    step_id="s1",
                    description="Auto-execute step",
                    tool_call=ToolCall(tool="time_now", args={}),
                ),
                WorkflowStep(
                    step_id="s2",
                    description="Approval required step",
                    tool_call=ToolCall(tool="send_email", args={"to": "user@example.com"}),
                ),
            ]
            wf = Workflow(title="Dry Run Test", steps=steps)

            runner = WorkflowRunner(wf, workflow_dir=tmpdir, approval_dir=tmpdir)
            result = runner.dry_run()

            assert result.workflow_id == wf.workflow_id
            assert result.total_steps == 2
            assert len(result.steps) == 2

            # First step would execute (auto-allowed)
            assert result.steps[0].would_execute is True
            assert result.steps[0].policy_decision == "allowed"

            # Second step would require approval
            assert result.steps[1].would_execute is True  # Would execute after approval
            assert result.steps[1].policy_decision == "requires_approval"

            # Workflow should complete (after approvals)
            assert result.would_complete is True

    def test_dry_run_with_denial(self):
        """Test dry-run mode with a step that would be denied."""
        with tempfile.TemporaryDirectory() as tmpdir:
            policy = ActionPolicy(
                tool_name="send_email",
                risk=RiskLevel.MEDIUM,
                denied_recipients=["blocked@example.com"],
            )
            engine = PolicyEngine(policies=[policy])

            steps = [
                WorkflowStep(
                    step_id="s1",
                    description="Denied step",
                    tool_call=ToolCall(tool="send_email", args={"to": "blocked@example.com"}),
                ),
            ]
            wf = Workflow(title="Dry Run Denial", steps=steps)

            runner = WorkflowRunner(
                wf,
                workflow_dir=tmpdir,
                approval_dir=tmpdir,
                policy_engine=engine,
            )
            result = runner.dry_run()

            assert result.steps[0].would_execute is False
            assert result.steps[0].policy_decision == "denied"
            assert result.would_complete is False
            assert result.blocking_reason is not None

    def test_workflow_persistence_after_each_step(self):
        """Test that workflow state is saved after each step."""
        with tempfile.TemporaryDirectory() as tmpdir:
            steps = [
                WorkflowStep(
                    step_id="s1",
                    description="Step 1",
                    tool_call=ToolCall(tool="time_now", args={"location": "Dallas, TX"}),
                ),
                WorkflowStep(
                    step_id="s2",
                    description="Step 2",
                    tool_call=ToolCall(tool="time_now", args={"location": "Dallas, TX"}),
                ),
            ]
            wf = Workflow(title="Persistence Test", steps=steps)

            runner = WorkflowRunner(
                wf,
                workflow_dir=tmpdir,
                approval_dir=tmpdir,
                default_context={"timezone": "America/Chicago"},
            )
            runner.run()

            # Load workflow from disk
            loaded = Workflow.load(wf.workflow_id, tmpdir)
            assert loaded is not None
            assert loaded.status == "completed"
            assert loaded.current_step_index == 2
            assert loaded.steps[0].result is not None
            assert loaded.steps[1].result is not None


class TestApprovalHelpers:
    """Tests for approval helper functions."""

    def test_approve_workflow(self):
        """Test approving a workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            approval = WorkflowApproval(
                approval_id="apr_test",
                workflow_id="wf_001",
                step_id="step_001",
            )
            approval.save(tmpdir)

            assert approve_workflow("apr_test", decided_by="test_user", approval_dir=tmpdir)

            loaded = WorkflowApproval.load("apr_test", tmpdir)
            assert loaded.status == "approved"
            assert loaded.decided_by == "test_user"
            assert loaded.decided_at is not None

    def test_deny_workflow(self):
        """Test denying a workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            approval = WorkflowApproval(
                approval_id="apr_test",
                workflow_id="wf_001",
                step_id="step_001",
            )
            approval.save(tmpdir)

            assert deny_workflow(
                "apr_test",
                decided_by="test_user",
                reason="Not allowed",
                approval_dir=tmpdir,
            )

            loaded = WorkflowApproval.load("apr_test", tmpdir)
            assert loaded.status == "denied"
            assert loaded.reason == "Not allowed"

    def test_approve_nonexistent_approval(self):
        """Test approving a nonexistent approval."""
        with tempfile.TemporaryDirectory() as tmpdir:
            assert not approve_workflow("nonexistent", approval_dir=tmpdir)

    def test_list_pending_approvals(self):
        """Test listing pending approvals."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create multiple approvals
            apr1 = WorkflowApproval(
                approval_id="apr_1",
                workflow_id="wf_001",
                step_id="step_001",
                status="pending",
            )
            apr2 = WorkflowApproval(
                approval_id="apr_2",
                workflow_id="wf_002",
                step_id="step_001",
                status="approved",
            )
            apr3 = WorkflowApproval(
                approval_id="apr_3",
                workflow_id="wf_003",
                step_id="step_001",
                status="pending",
            )
            apr1.save(tmpdir)
            apr2.save(tmpdir)
            apr3.save(tmpdir)

            pending = list_pending_approvals(tmpdir)
            assert len(pending) == 2
            approval_ids = {a.approval_id for a in pending}
            assert "apr_1" in approval_ids
            assert "apr_3" in approval_ids
            assert "apr_2" not in approval_ids


class TestStepResult:
    """Tests for the StepResult model."""

    def test_create_step_result(self):
        """Test creating a step result."""
        result = StepResult(
            step_id="step_001",
            success=True,
            output={"time": "12:00"},
        )
        assert result.step_id == "step_001"
        assert result.success is True
        assert result.output == {"time": "12:00"}
        assert result.executed_at is not None

    def test_create_failed_step_result(self):
        """Test creating a failed step result."""
        result = StepResult(
            step_id="step_001",
            success=False,
            error="Something went wrong",
        )
        assert result.success is False
        assert result.error == "Something went wrong"

    def test_create_skipped_step_result(self):
        """Test creating a skipped step result."""
        result = StepResult(
            step_id="step_001",
            success=True,
            skipped=True,
            skip_reason="Precondition failed",
        )
        assert result.skipped is True
        assert result.skip_reason == "Precondition failed"


class TestIDGenerators:
    """Tests for ID generator functions."""

    def test_generate_workflow_id(self):
        """Test workflow ID generation."""
        id1 = generate_workflow_id()
        id2 = generate_workflow_id()

        assert id1.startswith("wf_")
        assert id2.startswith("wf_")
        assert id1 != id2

    def test_generate_step_id(self):
        """Test step ID generation."""
        id1 = generate_step_id()
        id2 = generate_step_id()

        assert id1.startswith("step_")
        assert id2.startswith("step_")
        assert id1 != id2

    def test_generate_approval_id(self):
        """Test approval ID generation."""
        id1 = generate_approval_id()
        id2 = generate_approval_id()

        assert id1.startswith("apr_")
        assert id2.startswith("apr_")
        assert id1 != id2
