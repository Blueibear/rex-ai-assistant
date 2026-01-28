"""Workflow runner for executing Rex workflows.

This module provides the WorkflowRunner class that executes workflows step by step,
handling policy checks, approvals, idempotency, and persistence.

The runner:
- Evaluates preconditions before each step
- Consults the PolicyEngine to gate tool calls
- Creates approvals for steps that require user review
- Executes allowed tool calls via the tool router
- Evaluates postconditions after execution
- Persists state to disk after each step for recovery
- Supports dry-run mode for previewing actions
- Supports resume from blocked state after approval

Usage:
    from rex.workflow import Workflow, WorkflowStep
    from rex.workflow_runner import WorkflowRunner
    from rex.contracts import ToolCall

    workflow = Workflow(
        title="Example",
        steps=[
            WorkflowStep(
                description="Get time",
                tool_call=ToolCall(tool="time_now", args={"location": "Dallas, TX"})
            )
        ]
    )

    runner = WorkflowRunner(workflow)
    result = runner.run()
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rex.audit import LogEntry, get_audit_logger, AuditLogger
from rex.contracts import ToolCall
from rex.policy import PolicyDecision
from rex.policy_engine import PolicyEngine, get_policy_engine
from rex.tool_router import (
    execute_tool,
    PolicyDeniedError,
    ApprovalRequiredError,
)
from rex.workflow import (
    Workflow,
    WorkflowStep,
    WorkflowApproval,
    StepResult,
    get_condition,
    generate_approval_id,
    DEFAULT_WORKFLOW_DIR,
    DEFAULT_APPROVAL_DIR,
)

logger = logging.getLogger(__name__)


@dataclass
class DryRunStepResult:
    """Result of a dry-run step evaluation."""

    step_id: str
    description: str
    tool: str | None
    would_execute: bool
    policy_decision: str  # "allowed", "requires_approval", "denied"
    reason: str
    precondition_passed: bool
    precondition_name: str | None


@dataclass
class DryRunResult:
    """Summary of a dry-run workflow execution."""

    workflow_id: str
    title: str
    total_steps: int
    steps: list[DryRunStepResult]
    would_complete: bool
    blocking_reason: str | None


@dataclass
class RunResult:
    """Result of running a workflow."""

    workflow_id: str
    status: str
    steps_executed: int
    steps_total: int
    error: str | None
    blocking_approval_id: str | None


class WorkflowRunner:
    """Runner for executing workflows step by step.

    The WorkflowRunner handles the execution lifecycle of a workflow:
    1. Load or create workflow state
    2. Iterate through steps from current_step_index
    3. For each step:
       - Evaluate preconditions (skip if False)
       - Check idempotency (skip if already executed)
       - Consult policy engine for tool calls
       - Handle approvals for gated actions
       - Execute tool calls
       - Evaluate postconditions
       - Update and persist state
    4. Handle completion, failure, and blocking states

    The runner uses dependency injection for policy engine and audit logger,
    making it testable and configurable.
    """

    def __init__(
        self,
        workflow: Workflow,
        *,
        policy_engine: PolicyEngine | None = None,
        audit_logger: AuditLogger | None = None,
        workflow_dir: Path | str | None = None,
        approval_dir: Path | str | None = None,
        default_context: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the workflow runner.

        Args:
            workflow: The workflow to execute.
            policy_engine: Optional policy engine. Uses singleton if not provided.
            audit_logger: Optional audit logger. Uses singleton if not provided.
            workflow_dir: Directory for workflow persistence.
            approval_dir: Directory for approval persistence.
            default_context: Default context for tool execution.
        """
        self.workflow = workflow
        self.policy_engine = policy_engine or get_policy_engine()
        self.audit_logger = audit_logger or get_audit_logger()
        self.workflow_dir = Path(workflow_dir) if workflow_dir else DEFAULT_WORKFLOW_DIR
        self.approval_dir = Path(approval_dir) if approval_dir else DEFAULT_APPROVAL_DIR
        self.default_context = default_context or {}

    def run(self) -> RunResult:
        """Run the workflow from its current state.

        Executes steps starting from current_step_index until:
        - All steps are completed (status -> "completed")
        - A step fails (status -> "failed")
        - A step requires approval (status -> "blocked")

        State is persisted after each step to enable recovery.

        Returns:
            RunResult with execution summary.
        """
        steps_executed = 0

        # Mark as running if not already blocked
        if self.workflow.status != "blocked":
            self.workflow.mark_running()
            self._save_workflow()

        logger.info(
            "Running workflow %s from step %d of %d",
            self.workflow.workflow_id,
            self.workflow.current_step_index,
            len(self.workflow.steps),
        )

        while not self.workflow.is_finished() and not self.workflow.is_blocked():
            step = self.workflow.current_step()
            if step is None:
                # No more steps, workflow is complete
                self.workflow.status = "completed"
                self.workflow.completed_at = datetime.now(timezone.utc)
                self.workflow.updated_at = datetime.now(timezone.utc)
                self._save_workflow()
                break

            try:
                step_result = self._execute_step(step)
                steps_executed += 1

                if step_result.success or step_result.skipped:
                    # Advance to next step
                    self.workflow.advance()
                else:
                    # Step failed, mark workflow as failed
                    self.workflow.mark_failed(
                        step_result.error or f"Step {step.step_id} failed"
                    )
            except ApprovalBlockedError as e:
                # Step is blocked pending approval
                logger.info(
                    "Workflow %s blocked at step %s pending approval %s",
                    self.workflow.workflow_id,
                    step.step_id,
                    e.approval_id,
                )
                # Don't increment steps_executed for blocked step
                break

            self._save_workflow()

        result = RunResult(
            workflow_id=self.workflow.workflow_id,
            status=self.workflow.status,
            steps_executed=steps_executed,
            steps_total=len(self.workflow.steps),
            error=self.workflow.error,
            blocking_approval_id=self.workflow.blocking_approval_id,
        )

        logger.info(
            "Workflow %s finished with status=%s, executed=%d/%d steps",
            self.workflow.workflow_id,
            self.workflow.status,
            steps_executed,
            len(self.workflow.steps),
        )

        return result

    def resume(self) -> RunResult:
        """Resume a blocked workflow after approval.

        Checks the approval status and resumes execution if approved.
        If the approval is denied, marks the workflow as failed.
        If the approval is still pending, returns immediately.

        Returns:
            RunResult with execution summary.

        Raises:
            ValueError: If the workflow is not in blocked state.
        """
        if self.workflow.status != "blocked":
            raise ValueError(
                f"Cannot resume workflow in status '{self.workflow.status}', "
                "expected 'blocked'"
            )

        if not self.workflow.blocking_approval_id:
            raise ValueError("Workflow is blocked but has no blocking approval ID")

        # Load the approval
        approval = WorkflowApproval.load(
            self.workflow.blocking_approval_id,
            approval_dir=self.approval_dir,
        )

        if approval is None:
            raise ValueError(
                f"Approval {self.workflow.blocking_approval_id} not found"
            )

        logger.info(
            "Checking approval %s status=%s for workflow %s",
            approval.approval_id,
            approval.status,
            self.workflow.workflow_id,
        )

        if approval.status == "pending":
            # Still waiting for approval
            return RunResult(
                workflow_id=self.workflow.workflow_id,
                status="blocked",
                steps_executed=0,
                steps_total=len(self.workflow.steps),
                error=None,
                blocking_approval_id=self.workflow.blocking_approval_id,
            )

        if approval.status == "denied":
            self.workflow.mark_failed(
                f"Approval denied: {approval.reason or 'No reason provided'}"
            )
            self._save_workflow()
            return RunResult(
                workflow_id=self.workflow.workflow_id,
                status="failed",
                steps_executed=0,
                steps_total=len(self.workflow.steps),
                error=self.workflow.error,
                blocking_approval_id=None,
            )

        if approval.status == "expired":
            self.workflow.mark_failed("Approval expired")
            self._save_workflow()
            return RunResult(
                workflow_id=self.workflow.workflow_id,
                status="failed",
                steps_executed=0,
                steps_total=len(self.workflow.steps),
                error=self.workflow.error,
                blocking_approval_id=None,
            )

        # Approval is approved, resume execution
        logger.info(
            "Approval %s granted, resuming workflow %s",
            approval.approval_id,
            self.workflow.workflow_id,
        )

        # Update the step to mark approval received
        current_step = self.workflow.current_step()
        if current_step and current_step.step_id == approval.step_id:
            current_step.approval_id = approval.approval_id

        # Clear blocked state and run
        self.workflow.mark_running()
        self._save_workflow()

        return self.run()

    def dry_run(self) -> DryRunResult:
        """Preview workflow execution without making changes.

        Evaluates all steps, checking preconditions and policies,
        but does not execute any tool calls.

        Returns:
            DryRunResult with preview of what would happen.
        """
        steps: list[DryRunStepResult] = []
        would_complete = True
        blocking_reason: str | None = None

        # Temporarily save current state
        original_status = self.workflow.status
        original_step_index = self.workflow.current_step_index

        for i, step in enumerate(self.workflow.steps):
            # Skip already executed steps in dry run preview
            if i < original_step_index:
                continue

            # Evaluate precondition
            precondition_passed = True
            precondition_name = step.precondition
            if step.precondition:
                condition_func = get_condition(step.precondition)
                if condition_func:
                    try:
                        precondition_passed = condition_func(self.workflow.state)
                    except Exception as e:
                        logger.warning("Precondition %s failed: %s", step.precondition, e)
                        precondition_passed = False

            # Check policy for tool calls
            policy_decision = "allowed"
            reason = "No tool call"
            would_execute = precondition_passed

            if step.tool_call and precondition_passed:
                decision = self._evaluate_policy(step.tool_call)
                if decision.denied:
                    policy_decision = "denied"
                    reason = decision.reason
                    would_execute = False
                    would_complete = False
                    blocking_reason = f"Step {step.step_id} would be denied: {reason}"
                elif decision.requires_approval:
                    policy_decision = "requires_approval"
                    reason = decision.reason
                    # Would still execute after approval
                else:
                    policy_decision = "allowed"
                    reason = decision.reason

            if not precondition_passed:
                reason = f"Precondition '{step.precondition}' would fail"
                would_execute = False

            steps.append(DryRunStepResult(
                step_id=step.step_id,
                description=step.description,
                tool=step.tool_call.tool if step.tool_call else None,
                would_execute=would_execute,
                policy_decision=policy_decision,
                reason=reason,
                precondition_passed=precondition_passed,
                precondition_name=precondition_name,
            ))

        return DryRunResult(
            workflow_id=self.workflow.workflow_id,
            title=self.workflow.title,
            total_steps=len(self.workflow.steps),
            steps=steps,
            would_complete=would_complete,
            blocking_reason=blocking_reason,
        )

    def _execute_step(self, step: WorkflowStep) -> StepResult:
        """Execute a single workflow step.

        Args:
            step: The step to execute.

        Returns:
            StepResult with execution outcome.

        Raises:
            ApprovalBlockedError: If the step requires approval.
        """
        logger.debug("Executing step %s: %s", step.step_id, step.description)

        # Check idempotency
        if step.idempotency_key:
            executed_keys = self.workflow.get_executed_idempotency_keys()
            if step.idempotency_key in executed_keys:
                logger.info(
                    "Skipping step %s: idempotency key %s already executed",
                    step.step_id,
                    step.idempotency_key,
                )
                result = StepResult(
                    step_id=step.step_id,
                    success=True,
                    skipped=True,
                    skip_reason=f"Idempotency key '{step.idempotency_key}' already executed",
                )
                step.result = result
                return result

        # Evaluate precondition
        if step.precondition:
            condition_func = get_condition(step.precondition)
            if condition_func:
                try:
                    passed = condition_func(self.workflow.state)
                except Exception as e:
                    logger.warning("Precondition %s raised exception: %s", step.precondition, e)
                    passed = False

                if not passed:
                    logger.info(
                        "Skipping step %s: precondition '%s' returned False",
                        step.step_id,
                        step.precondition,
                    )
                    result = StepResult(
                        step_id=step.step_id,
                        success=True,
                        skipped=True,
                        skip_reason=f"Precondition '{step.precondition}' returned False",
                    )
                    step.result = result
                    return result
            else:
                logger.warning(
                    "Precondition '%s' not found in registry, skipping check",
                    step.precondition,
                )

        # If no tool call, step is just a checkpoint
        if step.tool_call is None:
            logger.debug("Step %s has no tool call, marking success", step.step_id)
            result = StepResult(
                step_id=step.step_id,
                success=True,
                output=None,
            )
            step.result = result
            return result

        # Check policy
        decision = self._evaluate_policy(step.tool_call)

        if decision.denied:
            logger.warning(
                "Step %s denied by policy: %s",
                step.step_id,
                decision.reason,
            )
            self._log_step_to_audit(step, "denied", None, decision.reason)
            result = StepResult(
                step_id=step.step_id,
                success=False,
                error=f"Denied by policy: {decision.reason}",
            )
            step.result = result
            return result

        if decision.requires_approval:
            # Check if step already has an approved approval
            if step.approval_id:
                approval = WorkflowApproval.load(step.approval_id, self.approval_dir)
                if approval and approval.status == "approved":
                    logger.debug(
                        "Step %s already approved via %s",
                        step.step_id,
                        step.approval_id,
                    )
                    # Continue to execution
                else:
                    # Re-block if approval not found or not approved
                    raise self._create_approval_and_block(step, decision.reason)
            else:
                # Create new approval and block
                raise self._create_approval_and_block(step, decision.reason)

        # Execute the tool call
        try:
            tool_result = execute_tool(
                {"tool": step.tool_call.tool, "args": step.tool_call.args},
                self.default_context,
                policy_engine=self.policy_engine,
                skip_policy_check=True,  # Already checked above
                task_id=self.workflow.workflow_id,
                requested_by=f"workflow:{self.workflow.workflow_id}",
            )
        except Exception as e:
            logger.error("Step %s tool execution failed: %s", step.step_id, e)
            result = StepResult(
                step_id=step.step_id,
                success=False,
                error=str(e),
            )
            step.result = result
            return result

        # Check for error in result
        if "error" in tool_result:
            error_info = tool_result["error"]
            error_msg = error_info.get("message", str(error_info)) if isinstance(error_info, dict) else str(error_info)
            result = StepResult(
                step_id=step.step_id,
                success=False,
                output=tool_result,
                error=error_msg,
            )
            step.result = result
            return result

        # Store output in workflow state
        self.workflow.state[f"step_{step.step_id}_output"] = tool_result

        # Evaluate postcondition
        if step.postcondition:
            condition_func = get_condition(step.postcondition)
            if condition_func:
                try:
                    passed = condition_func(self.workflow.state)
                except Exception as e:
                    logger.warning("Postcondition %s raised exception: %s", step.postcondition, e)
                    passed = False

                if not passed:
                    logger.warning(
                        "Step %s postcondition '%s' failed",
                        step.step_id,
                        step.postcondition,
                    )
                    result = StepResult(
                        step_id=step.step_id,
                        success=False,
                        output=tool_result,
                        error=f"Postcondition '{step.postcondition}' failed",
                    )
                    step.result = result
                    return result

        # Success!
        result = StepResult(
            step_id=step.step_id,
            success=True,
            output=tool_result,
        )
        step.result = result

        self._log_step_to_audit(step, "allowed", tool_result, None)

        return result

    def _evaluate_policy(self, tool_call: ToolCall) -> PolicyDecision:
        """Evaluate policy for a tool call.

        Args:
            tool_call: The tool call to evaluate.

        Returns:
            PolicyDecision from the policy engine.
        """
        metadata = self._extract_policy_metadata(tool_call.args)
        return self.policy_engine.decide(tool_call, metadata)

    def _extract_policy_metadata(self, args: dict[str, Any]) -> dict[str, Any]:
        """Extract metadata from tool arguments for policy evaluation."""
        metadata: dict[str, Any] = {}

        for key in ("to", "recipient", "email", "address"):
            if key in args and isinstance(args[key], str):
                metadata["recipient"] = args[key]
                break

        for key in ("domain", "host"):
            if key in args and isinstance(args[key], str):
                metadata["domain"] = args[key]
                break

        if "url" in args and isinstance(args["url"], str):
            url = args["url"]
            if "://" in url:
                domain_part = url.split("://", 1)[1].split("/", 1)[0]
                domain_part = domain_part.split(":")[0]
                metadata["domain"] = domain_part

        return metadata

    def _create_approval_and_block(
        self,
        step: WorkflowStep,
        reason: str,
    ) -> "ApprovalBlockedError":
        """Create an approval request and block the workflow.

        Args:
            step: The step requiring approval.
            reason: Reason approval is required.

        Returns:
            ApprovalBlockedError to be raised.
        """
        tool_summary = None
        if step.tool_call:
            tool_summary = f"{step.tool_call.tool}({step.tool_call.args})"

        approval = WorkflowApproval(
            approval_id=generate_approval_id(),
            workflow_id=self.workflow.workflow_id,
            step_id=step.step_id,
            status="pending",
            reason=reason,
            requested_by="workflow_runner",
            step_description=step.description,
            tool_call_summary=tool_summary,
        )
        approval.save(self.approval_dir)

        step.requires_approval = True
        step.approval_id = approval.approval_id
        self.workflow.mark_blocked(approval.approval_id)
        self._save_workflow()

        self._log_step_to_audit(step, "requires_approval", None, reason)

        logger.info(
            "Created approval %s for step %s, workflow blocked",
            approval.approval_id,
            step.step_id,
        )

        return ApprovalBlockedError(approval.approval_id, step.step_id)

    def _log_step_to_audit(
        self,
        step: WorkflowStep,
        decision: str,
        result: dict[str, Any] | None,
        error: str | None,
    ) -> None:
        """Log a step execution to the audit log.

        Args:
            step: The executed step.
            decision: Policy decision ("allowed", "denied", "requires_approval").
            result: Tool result if any.
            error: Error message if any.
        """
        if step.tool_call is None:
            return

        try:
            entry = LogEntry(
                action_id=f"wf_{self.workflow.workflow_id}_{step.step_id}",
                task_id=self.workflow.workflow_id,
                tool=step.tool_call.tool,
                tool_call_args=step.tool_call.args,
                policy_decision=decision,  # type: ignore
                tool_result=result,
                error=error,
                requested_by=f"workflow:{self.workflow.workflow_id}",
            )
            self.audit_logger.log(entry)
        except Exception as e:
            logger.warning("Failed to log step to audit: %s", e)

    def _save_workflow(self) -> None:
        """Save the workflow to disk."""
        try:
            self.workflow.save(self.workflow_dir)
        except Exception as e:
            logger.error("Failed to save workflow: %s", e)


class ApprovalBlockedError(Exception):
    """Raised when a step is blocked pending approval."""

    def __init__(self, approval_id: str, step_id: str) -> None:
        self.approval_id = approval_id
        self.step_id = step_id
        super().__init__(
            f"Step '{step_id}' blocked pending approval '{approval_id}'"
        )


def approve_workflow(
    approval_id: str,
    *,
    decided_by: str = "user",
    reason: str | None = None,
    approval_dir: Path | str | None = None,
) -> bool:
    """Approve a pending workflow approval.

    This is a helper function for programmatically approving workflow steps.

    Args:
        approval_id: ID of the approval to approve.
        decided_by: Who approved (default "user").
        reason: Optional approval reason.
        approval_dir: Directory containing approvals.

    Returns:
        True if approval was updated, False if not found.
    """
    if approval_dir is None:
        approval_dir = DEFAULT_APPROVAL_DIR

    approval = WorkflowApproval.load(approval_id, approval_dir)
    if approval is None:
        return False

    approval.status = "approved"
    approval.decided_by = decided_by
    approval.reason = reason
    approval.decided_at = datetime.now(timezone.utc)
    approval.save(approval_dir)

    logger.info("Approved workflow approval %s by %s", approval_id, decided_by)
    return True


def deny_workflow(
    approval_id: str,
    *,
    decided_by: str = "user",
    reason: str | None = None,
    approval_dir: Path | str | None = None,
) -> bool:
    """Deny a pending workflow approval.

    Args:
        approval_id: ID of the approval to deny.
        decided_by: Who denied (default "user").
        reason: Optional denial reason.
        approval_dir: Directory containing approvals.

    Returns:
        True if approval was updated, False if not found.
    """
    if approval_dir is None:
        approval_dir = DEFAULT_APPROVAL_DIR

    approval = WorkflowApproval.load(approval_id, approval_dir)
    if approval is None:
        return False

    approval.status = "denied"
    approval.decided_by = decided_by
    approval.reason = reason
    approval.decided_at = datetime.now(timezone.utc)
    approval.save(approval_dir)

    logger.info("Denied workflow approval %s by %s", approval_id, decided_by)
    return True


def list_pending_approvals(
    approval_dir: Path | str | None = None,
) -> list[WorkflowApproval]:
    """List all pending workflow approvals.

    Args:
        approval_dir: Directory containing approvals.

    Returns:
        List of pending WorkflowApproval objects.
    """
    if approval_dir is None:
        approval_dir = DEFAULT_APPROVAL_DIR
    approval_dir = Path(approval_dir)

    if not approval_dir.exists():
        return []

    pending = []
    for file_path in approval_dir.glob("*.json"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                approval = WorkflowApproval.model_validate_json(f.read())
                if approval.status == "pending":
                    pending.append(approval)
        except Exception as e:
            logger.warning("Failed to load approval from %s: %s", file_path, e)

    return sorted(pending, key=lambda a: a.requested_at)


__all__ = [
    "WorkflowRunner",
    "RunResult",
    "DryRunResult",
    "DryRunStepResult",
    "ApprovalBlockedError",
    "approve_workflow",
    "deny_workflow",
    "list_pending_approvals",
]
