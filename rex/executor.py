"""Executor for Rex - runs workflows with budget constraints and evidence collection.

This module implements the Executor class that wraps the WorkflowRunner to add:
1. Budget enforcement (max actions, messages, time)
2. Evidence collection (screenshots, tool results)
3. Autonomy mode checking
4. Enhanced audit logging

The executor provides safe, bounded execution of autonomous workflows with
comprehensive tracking and safety limits.

Usage:
    from rex.executor import Executor
    from rex.workflow import Workflow

    workflow = Workflow(...)
    budget = {
        "max_actions": 10,
        "max_messages": 5,
        "max_time_seconds": 300
    }

    executor = Executor(workflow, budget)
    result = executor.run()
    print(f"Actions taken: {result.actions_taken}")
    print(f"Evidence collected: {len(result.evidence)}")
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from rex.audit import get_audit_logger, AuditLogger
from rex.contracts import EvidenceRef
from rex.policy_engine import PolicyEngine, get_policy_engine
from rex.workflow import Workflow, WorkflowStep, StepResult
from rex.workflow_runner import WorkflowRunner, RunResult

logger = logging.getLogger(__name__)


class BudgetExceededError(Exception):
    """Raised when a budget limit is exceeded during execution."""

    def __init__(self, budget_type: str, limit: int, current: int):
        self.budget_type = budget_type
        self.limit = limit
        self.current = current
        super().__init__(
            f"Budget exceeded: {budget_type} limit={limit}, current={current}"
        )


@dataclass
class ExecutionBudget:
    """Budget constraints for workflow execution.

    Attributes:
        max_actions: Maximum number of tool calls to execute (0 = unlimited)
        max_messages: Maximum number of messages to send (0 = unlimited)
        max_time_seconds: Maximum execution time in seconds (0 = unlimited)
    """

    max_actions: int = 0
    max_messages: int = 0
    max_time_seconds: int = 0

    def is_unlimited(self) -> bool:
        """Check if all budget limits are unlimited."""
        return (
            self.max_actions == 0
            and self.max_messages == 0
            and self.max_time_seconds == 0
        )

    def __repr__(self) -> str:
        parts = []
        if self.max_actions > 0:
            parts.append(f"actions={self.max_actions}")
        if self.max_messages > 0:
            parts.append(f"messages={self.max_messages}")
        if self.max_time_seconds > 0:
            parts.append(f"time={self.max_time_seconds}s")

        if not parts:
            return "ExecutionBudget(unlimited)"
        return f"ExecutionBudget({', '.join(parts)})"


@dataclass
class ExecutionResult:
    """Result of executing a workflow with the executor.

    Attributes:
        workflow_id: ID of the executed workflow
        status: Final workflow status
        actions_taken: Number of actions executed
        messages_sent: Number of messages sent
        elapsed_seconds: Total execution time
        budget: Budget limits that were in effect
        remaining_budget: Remaining budget after execution
        evidence: Evidence collected during execution
        error: Error message if execution failed
        blocking_approval_id: Approval ID if workflow is blocked
        summary: Human-readable summary of execution
    """

    workflow_id: str
    status: str
    actions_taken: int
    messages_sent: int
    elapsed_seconds: float
    budget: ExecutionBudget
    remaining_budget: ExecutionBudget
    evidence: list[EvidenceRef] = field(default_factory=list)
    error: str | None = None
    blocking_approval_id: str | None = None
    summary: str = ""

    def __str__(self) -> str:
        """Format result for display."""
        lines = [
            f"Workflow: {self.workflow_id}",
            f"Status: {self.status}",
            f"Actions taken: {self.actions_taken}",
            f"Messages sent: {self.messages_sent}",
            f"Elapsed time: {self.elapsed_seconds:.2f}s",
            f"Evidence: {len(self.evidence)} items",
        ]

        if self.error:
            lines.append(f"Error: {self.error}")

        if self.blocking_approval_id:
            lines.append(f"Blocked on approval: {self.blocking_approval_id}")

        if self.summary:
            lines.append(f"\n{self.summary}")

        return "\n".join(lines)


class Executor:
    """Executor that runs workflows with budget constraints and safety checks.

    The executor wraps the WorkflowRunner to provide:
    - Budget enforcement (actions, messages, time)
    - Evidence collection (tool results, logs)
    - Enhanced audit logging
    - Summary generation

    Attributes:
        workflow: Workflow to execute
        budget: Budget constraints
        policy_engine: Policy engine for validation
        audit_logger: Audit logger for tracking
    """

    def __init__(
        self,
        workflow: Workflow,
        budget: dict[str, int] | ExecutionBudget | None = None,
        *,
        policy_engine: PolicyEngine | None = None,
        audit_logger: AuditLogger | None = None,
        workflow_dir: Path | str | None = None,
        approval_dir: Path | str | None = None,
    ):
        """Initialize the executor.

        Args:
            workflow: Workflow to execute
            budget: Budget constraints (dict or ExecutionBudget)
            policy_engine: Optional policy engine
            audit_logger: Optional audit logger
            workflow_dir: Directory for workflow persistence
            approval_dir: Directory for approval persistence
        """
        self.workflow = workflow

        # Parse budget
        if budget is None:
            self.budget = ExecutionBudget()
        elif isinstance(budget, ExecutionBudget):
            self.budget = budget
        else:
            self.budget = ExecutionBudget(
                max_actions=budget.get("max_actions", 0),
                max_messages=budget.get("max_messages", 0),
                max_time_seconds=budget.get("max_time_seconds", 0),
            )

        self.policy_engine = policy_engine or get_policy_engine()
        self.audit_logger = audit_logger or get_audit_logger()
        self.workflow_dir = workflow_dir
        self.approval_dir = approval_dir

        # Tracking
        self.actions_taken = 0
        self.messages_sent = 0
        self.start_time: float | None = None
        self.evidence: list[EvidenceRef] = []

    def run(self) -> ExecutionResult:
        """Execute the workflow with budget enforcement.

        Returns:
            ExecutionResult with execution summary and evidence

        Raises:
            BudgetExceededError: If a budget limit is exceeded (should not happen,
                as we check before each step)
        """
        if self.workflow.is_finished():
            logger.info(
                "Workflow %s already finished with status %s; skipping execution.",
                self.workflow.workflow_id,
                self.workflow.status,
            )
            return ExecutionResult(
                workflow_id=self.workflow.workflow_id,
                status=self.workflow.status,
                actions_taken=0,
                messages_sent=0,
                elapsed_seconds=0.0,
                budget=self.budget,
                remaining_budget=self.budget,
                evidence=[],
                error=self.workflow.error,
                blocking_approval_id=self.workflow.blocking_approval_id,
                summary="Workflow already finished; skipped execution for idempotency.",
            )

        logger.info(
            "Starting execution of workflow %s with budget %s",
            self.workflow.workflow_id,
            self.budget,
        )

        self.start_time = time.time()

        # Create workflow runner with budget hooks
        runner = WorkflowRunner(
            self.workflow,
            policy_engine=self.policy_engine,
            audit_logger=self.audit_logger,
            workflow_dir=self.workflow_dir,
            approval_dir=self.approval_dir,
            pre_step_hook=self._check_time_budget,
            before_tool_call_hook=self._check_action_budget,
            after_tool_call_hook=self._record_step,
        )

        # Execute workflow with budget checks
        try:
            # Run workflow
            run_result = runner.run()

            # Generate summary
            summary = self._generate_summary(run_result)

            elapsed = time.time() - self.start_time

            return ExecutionResult(
                workflow_id=self.workflow.workflow_id,
                status=run_result.status,
                actions_taken=self.actions_taken,
                messages_sent=self.messages_sent,
                elapsed_seconds=elapsed,
                budget=self.budget,
                remaining_budget=self._calculate_remaining_budget(elapsed),
                evidence=self.evidence,
                error=run_result.error,
                blocking_approval_id=run_result.blocking_approval_id,
                summary=summary,
            )

        except BudgetExceededError as e:
            # Budget exceeded, mark workflow as failed
            self.workflow.mark_failed(str(e))
            self.workflow.save(self.workflow_dir)

            elapsed = time.time() - self.start_time

            logger.error("Execution stopped: %s", e)

            return ExecutionResult(
                workflow_id=self.workflow.workflow_id,
                status="failed",
                actions_taken=self.actions_taken,
                messages_sent=self.messages_sent,
                elapsed_seconds=elapsed,
                budget=self.budget,
                remaining_budget=self._calculate_remaining_budget(elapsed),
                evidence=self.evidence,
                error=str(e),
                summary=f"Execution failed: {e}",
            )

    def _check_time_budget(self, step: WorkflowStep) -> None:
        """Check if the time budget has been exceeded."""
        if self.budget.max_time_seconds > 0 and self.start_time is not None:
            elapsed = time.time() - self.start_time
            if elapsed >= self.budget.max_time_seconds:
                raise BudgetExceededError(
                    "time_seconds",
                    self.budget.max_time_seconds,
                    int(elapsed),
                )

    def _check_action_budget(self, step: WorkflowStep) -> None:
        """Check action and message budgets before executing a tool call."""
        if step.tool_call is None:
            return

        if self.budget.max_actions > 0 and self.actions_taken >= self.budget.max_actions:
            raise BudgetExceededError(
                "actions",
                self.budget.max_actions,
                self.actions_taken,
            )

        if self.budget.max_messages > 0:
            if step.tool_call.tool in ("send_email", "send_sms", "send_notification"):
                if self.messages_sent >= self.budget.max_messages:
                    raise BudgetExceededError(
                        "messages",
                        self.budget.max_messages,
                        self.messages_sent,
                    )

    def _record_step(self, step: WorkflowStep, result: StepResult) -> None:
        """Record counts and evidence after a tool call executes."""
        if step.tool_call is None:
            return

        self.actions_taken += 1

        if step.tool_call.tool in ("send_email", "send_sms", "send_notification"):
            self.messages_sent += 1

        if result.success and result.output:
            evidence = EvidenceRef(
                evidence_id=f"ev_{self.workflow.workflow_id}_{step.step_id}",
                kind="log",
                uri=None,
                created_at=result.executed_at,
            )
            self.evidence.append(evidence)

    def _collect_evidence(self) -> None:
        """Collect evidence from executed workflow steps."""
        for step in self.workflow.steps:
            if step.result is None:
                continue

            # Count actions
            if step.tool_call is not None:
                self.actions_taken += 1

                # Count messages sent
                if step.tool_call.tool in ("send_email", "send_sms", "send_notification"):
                    self.messages_sent += 1

            # Collect evidence from tool results
            if step.result.success and step.result.output:
                # Create evidence reference for this step
                evidence = EvidenceRef(
                    evidence_id=f"ev_{self.workflow.workflow_id}_{step.step_id}",
                    kind="log",
                    uri=None,  # Could be enhanced to store results in files
                    created_at=step.result.executed_at,
                )
                self.evidence.append(evidence)

    def _calculate_remaining_budget(self, elapsed: float) -> ExecutionBudget:
        """Calculate remaining budget after execution."""
        return ExecutionBudget(
            max_actions=max(0, self.budget.max_actions - self.actions_taken)
                       if self.budget.max_actions > 0 else 0,
            max_messages=max(0, self.budget.max_messages - self.messages_sent)
                        if self.budget.max_messages > 0 else 0,
            max_time_seconds=max(0, int(self.budget.max_time_seconds - elapsed))
                            if self.budget.max_time_seconds > 0 else 0,
        )

    def _generate_summary(self, run_result: RunResult) -> str:
        """Generate a human-readable summary of execution."""
        lines = []

        if run_result.status == "completed":
            lines.append(
                f"Workflow completed successfully. "
                f"Executed {run_result.steps_executed} of {run_result.steps_total} steps."
            )
        elif run_result.status == "blocked":
            lines.append(
                f"Workflow blocked pending approval. "
                f"Executed {run_result.steps_executed} of {run_result.steps_total} steps."
            )
        elif run_result.status == "failed":
            lines.append(
                f"Workflow failed: {run_result.error or 'Unknown error'}. "
                f"Executed {run_result.steps_executed} of {run_result.steps_total} steps."
            )
        else:
            lines.append(
                f"Workflow status: {run_result.status}. "
                f"Executed {run_result.steps_executed} of {run_result.steps_total} steps."
            )

        # Add budget info if not unlimited
        if not self.budget.is_unlimited():
            budget_parts = []
            if self.budget.max_actions > 0:
                budget_parts.append(
                    f"{self.actions_taken}/{self.budget.max_actions} actions"
                )
            if self.budget.max_messages > 0:
                budget_parts.append(
                    f"{self.messages_sent}/{self.budget.max_messages} messages"
                )
            if budget_parts:
                lines.append(f"Budget used: {', '.join(budget_parts)}")

        return " ".join(lines)


__all__ = [
    "Executor",
    "ExecutionBudget",
    "ExecutionResult",
    "BudgetExceededError",
]
