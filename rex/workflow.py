"""Workflow primitives for Rex.

This module defines the core data models for Rex's workflow engine, providing
structured, persistent workflows that can be paused, resumed, and recovered
after restarts.

A workflow consists of ordered steps, each representing an action to be taken.
Steps can have preconditions that must pass before execution and postconditions
that validate the result.

Workflow State:
- Workflows are saved to disk as JSON files in data/workflows/{workflow_id}.json
- State is persisted after each step to enable recovery
- Approvals are tracked via the approval system in data/approvals/

Usage:
    from rex.workflow import Workflow, WorkflowStep
    from rex.contracts import ToolCall

    step = WorkflowStep(
        step_id="step_1",
        description="Get current time",
        tool_call=ToolCall(tool="time_now", args={"location": "Dallas, TX"}),
    )
    workflow = Workflow(
        workflow_id="wf_001",
        title="Example workflow",
        steps=[step],
    )
    workflow.save()
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal

from pydantic import BaseModel, Field

from rex.contracts import ToolCall

logger = logging.getLogger(__name__)

# Default directories for workflow data
DEFAULT_WORKFLOW_DIR = Path("data/workflows")
DEFAULT_APPROVAL_DIR = Path("data/approvals")


def _utc_now() -> datetime:
    """Return the current UTC datetime."""
    return datetime.now(timezone.utc)


def generate_workflow_id() -> str:
    """Generate a unique workflow ID."""
    return f"wf_{uuid.uuid4().hex[:12]}"


def generate_step_id() -> str:
    """Generate a unique step ID."""
    return f"step_{uuid.uuid4().hex[:8]}"


def generate_approval_id() -> str:
    """Generate a unique approval ID."""
    return f"apr_{uuid.uuid4().hex[:12]}"


class StepResult(BaseModel):
    """Result of executing a workflow step.

    Captures the outcome of a step execution including success/failure,
    output data, and any error information.
    """

    step_id: str = Field(
        ...,
        description="ID of the step that was executed",
    )
    success: bool = Field(
        ...,
        description="Whether the step executed successfully",
    )
    output: dict[str, Any] | None = Field(
        default=None,
        description="Output data from the step execution",
    )
    error: str | None = Field(
        default=None,
        description="Error message if the step failed",
    )
    executed_at: datetime = Field(
        default_factory=_utc_now,
        description="When the step was executed",
    )
    skipped: bool = Field(
        default=False,
        description="Whether the step was skipped (e.g., idempotency check)",
    )
    skip_reason: str | None = Field(
        default=None,
        description="Reason the step was skipped",
    )


class WorkflowStep(BaseModel):
    """A single step in a workflow.

    Each step represents an action to be taken, optionally with a tool call.
    Steps can have preconditions (evaluated before execution) and postconditions
    (evaluated after execution) to control flow and validate results.

    Preconditions and postconditions are callable names that can be resolved
    at runtime. They receive the workflow state and return True/False.
    """

    step_id: str = Field(
        default_factory=generate_step_id,
        description="Unique identifier for this step",
    )
    description: str = Field(
        ...,
        description="Human-readable description of what this step does",
    )
    tool_call: ToolCall | None = Field(
        default=None,
        description="Tool call to execute for this step",
    )
    precondition: str | None = Field(
        default=None,
        description="Name of precondition function to evaluate before execution. "
        "If the precondition returns False, the step is skipped.",
    )
    postcondition: str | None = Field(
        default=None,
        description="Name of postcondition function to evaluate after execution. "
        "If the postcondition returns False, the step is marked as failed.",
    )
    idempotency_key: str | None = Field(
        default=None,
        description="Key for idempotency checking. If a step with the same key "
        "was already executed successfully, it will be skipped.",
    )
    result: StepResult | None = Field(
        default=None,
        description="Result of step execution, populated after the step runs",
    )
    requires_approval: bool = Field(
        default=False,
        description="Whether this step requires explicit approval before execution",
    )
    approval_id: str | None = Field(
        default=None,
        description="ID of the approval request if approval was required",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "step_id": "step_001",
                    "description": "Get current time in Dallas",
                    "tool_call": {
                        "tool": "time_now",
                        "args": {"location": "Dallas, TX"},
                    },
                    "idempotency_key": "get_time_dallas",
                }
            ]
        }
    }


class WorkflowApproval(BaseModel):
    """An approval request for a workflow step.

    Approvals gate execution of risky steps. They are persisted to disk
    and can be updated externally (e.g., via UI or manual JSON edit).
    """

    approval_id: str = Field(
        default_factory=generate_approval_id,
        description="Unique identifier for this approval",
    )
    workflow_id: str = Field(
        ...,
        description="ID of the workflow this approval belongs to",
    )
    step_id: str = Field(
        ...,
        description="ID of the step requiring approval",
    )
    status: Literal["pending", "approved", "denied", "expired"] = Field(
        default="pending",
        description="Current status of the approval",
    )
    reason: str | None = Field(
        default=None,
        description="Reason for the approval decision",
    )
    requested_by: str | None = Field(
        default=None,
        description="Who/what requested this approval (e.g., 'workflow_runner')",
    )
    decided_by: str | None = Field(
        default=None,
        description="Who made the approval decision",
    )
    requested_at: datetime = Field(
        default_factory=_utc_now,
        description="When the approval was requested",
    )
    decided_at: datetime | None = Field(
        default=None,
        description="When the approval decision was made",
    )
    step_description: str | None = Field(
        default=None,
        description="Description of the step for context",
    )
    tool_call_summary: str | None = Field(
        default=None,
        description="Summary of the tool call for context",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "approval_id": "apr_001",
                    "workflow_id": "wf_001",
                    "step_id": "step_001",
                    "status": "pending",
                    "requested_by": "workflow_runner",
                    "step_description": "Send email to user",
                }
            ]
        }
    }

    def save(self, approval_dir: Path | str | None = None) -> Path:
        """Save the approval to disk.

        Args:
            approval_dir: Directory to save approvals. Defaults to data/approvals.

        Returns:
            Path to the saved approval file.
        """
        if approval_dir is None:
            approval_dir = DEFAULT_APPROVAL_DIR
        approval_dir = Path(approval_dir)
        approval_dir.mkdir(parents=True, exist_ok=True)

        file_path = approval_dir / f"{self.approval_id}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(self.model_dump_json(indent=2))
        logger.debug("Saved approval %s to %s", self.approval_id, file_path)
        return file_path

    @classmethod
    def load(
        cls,
        approval_id: str,
        approval_dir: Path | str | None = None,
    ) -> WorkflowApproval | None:
        """Load an approval from disk.

        Args:
            approval_id: ID of the approval to load.
            approval_dir: Directory containing approvals. Defaults to data/approvals.

        Returns:
            The loaded WorkflowApproval or None if not found.
        """
        if approval_dir is None:
            approval_dir = DEFAULT_APPROVAL_DIR
        approval_dir = Path(approval_dir)
        file_path = approval_dir / f"{approval_id}.json"

        if not file_path.exists():
            return None

        try:
            with open(file_path, encoding="utf-8") as f:
                return cls.model_validate_json(f.read())
        except Exception as e:
            logger.warning("Failed to load approval %s: %s", approval_id, e)
            return None


class Workflow(BaseModel):
    """A workflow representing a sequence of steps to be executed.

    Workflows are the primary unit of execution in the workflow engine.
    They maintain state including:
    - Current step index (for pause/resume)
    - Execution status
    - Intermediate values in the state dict
    - History of executed steps

    Workflows can be saved to and loaded from JSON files, enabling
    persistence across restarts and recovery from failures.
    """

    workflow_id: str = Field(
        default_factory=generate_workflow_id,
        description="Unique identifier for this workflow",
    )
    title: str = Field(
        ...,
        description="Human-readable title describing the workflow",
    )
    created_at: datetime = Field(
        default_factory=_utc_now,
        description="When the workflow was created",
    )
    updated_at: datetime = Field(
        default_factory=_utc_now,
        description="When the workflow was last updated",
    )
    status: Literal["queued", "running", "blocked", "completed", "failed", "canceled"] = Field(
        default="queued",
        description="Current status of the workflow",
    )
    steps: list[WorkflowStep] = Field(
        default_factory=list,
        description="Ordered list of steps in the workflow",
    )
    current_step_index: int = Field(
        default=0,
        description="Index of the next step to execute (0-based)",
    )
    idempotency_key: str | None = Field(
        default=None,
        description="Global idempotency key for the entire workflow",
    )
    state: dict[str, Any] = Field(
        default_factory=dict,
        description="Dictionary for storing intermediate values between steps",
    )
    blocking_approval_id: str | None = Field(
        default=None,
        description="ID of the approval that is blocking this workflow",
    )
    error: str | None = Field(
        default=None,
        description="Error message if the workflow failed",
    )
    requested_by: str | None = Field(
        default=None,
        description="Who/what created this workflow",
    )
    completed_at: datetime | None = Field(
        default=None,
        description="When the workflow completed",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "workflow_id": "wf_001",
                    "title": "Morning routine workflow",
                    "status": "queued",
                    "steps": [
                        {
                            "step_id": "step_001",
                            "description": "Get current time",
                            "tool_call": {"tool": "time_now", "args": {}},
                        }
                    ],
                }
            ]
        }
    }

    def is_finished(self) -> bool:
        """Check if the workflow has finished (completed, failed, or canceled)."""
        return self.status in ("completed", "failed", "canceled")

    def is_blocked(self) -> bool:
        """Check if the workflow is blocked waiting for approval."""
        return self.status == "blocked"

    def current_step(self) -> WorkflowStep | None:
        """Get the current step to execute, or None if finished."""
        if self.current_step_index >= len(self.steps):
            return None
        return self.steps[self.current_step_index]

    def advance(self) -> bool:
        """Advance to the next step.

        Returns:
            True if there is a next step, False if workflow is complete.
        """
        self.current_step_index += 1
        self.updated_at = _utc_now()

        if self.current_step_index >= len(self.steps):
            self.status = "completed"
            self.completed_at = _utc_now()
            return False
        return True

    def mark_blocked(self, approval_id: str) -> None:
        """Mark the workflow as blocked pending approval."""
        self.status = "blocked"
        self.blocking_approval_id = approval_id
        self.updated_at = _utc_now()

    def mark_failed(self, error: str) -> None:
        """Mark the workflow as failed with an error message."""
        self.status = "failed"
        self.error = error
        self.completed_at = _utc_now()
        self.updated_at = _utc_now()

    def mark_running(self) -> None:
        """Mark the workflow as running."""
        self.status = "running"
        self.blocking_approval_id = None
        self.updated_at = _utc_now()

    def mark_canceled(self) -> None:
        """Mark the workflow as canceled."""
        self.status = "canceled"
        self.completed_at = _utc_now()
        self.updated_at = _utc_now()

    def get_executed_idempotency_keys(self) -> set[str]:
        """Get the set of idempotency keys for successfully executed steps."""
        keys: set[str] = set()
        for step in self.steps:
            if (
                step.idempotency_key
                and step.result is not None
                and step.result.success
            ):
                keys.add(step.idempotency_key)
        return keys

    def save(self, workflow_dir: Path | str | None = None) -> Path:
        """Save the workflow to disk.

        Args:
            workflow_dir: Directory to save workflows. Defaults to data/workflows.

        Returns:
            Path to the saved workflow file.
        """
        if workflow_dir is None:
            workflow_dir = DEFAULT_WORKFLOW_DIR
        workflow_dir = Path(workflow_dir)
        workflow_dir.mkdir(parents=True, exist_ok=True)

        file_path = workflow_dir / f"{self.workflow_id}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(self.model_dump_json(indent=2))
        logger.debug("Saved workflow %s to %s", self.workflow_id, file_path)
        return file_path

    @classmethod
    def load(
        cls,
        workflow_id: str,
        workflow_dir: Path | str | None = None,
    ) -> Workflow | None:
        """Load a workflow from disk by ID.

        Args:
            workflow_id: ID of the workflow to load.
            workflow_dir: Directory containing workflows. Defaults to data/workflows.

        Returns:
            The loaded Workflow or None if not found.
        """
        if workflow_dir is None:
            workflow_dir = DEFAULT_WORKFLOW_DIR
        workflow_dir = Path(workflow_dir)
        file_path = workflow_dir / f"{workflow_id}.json"

        if not file_path.exists():
            return None

        try:
            with open(file_path, encoding="utf-8") as f:
                return cls.model_validate_json(f.read())
        except Exception as e:
            logger.warning("Failed to load workflow %s: %s", workflow_id, e)
            return None

    @classmethod
    def load_from_file(cls, file_path: Path | str) -> Workflow:
        """Load a workflow from a specific file path.

        Args:
            file_path: Path to the workflow JSON file.

        Returns:
            The loaded Workflow.

        Raises:
            FileNotFoundError: If the file doesn't exist.
            ValueError: If the file contains invalid JSON or workflow data.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Workflow file not found: {file_path}")

        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()
        except OSError as e:
            raise ValueError(f"Failed to read workflow file: {e}") from e

        try:
            return cls.model_validate_json(content)
        except Exception as e:
            raise ValueError(f"Invalid workflow JSON: {e}") from e

    @classmethod
    def list_workflows(
        cls,
        workflow_dir: Path | str | None = None,
        status: str | None = None,
    ) -> list[Workflow]:
        """List all workflows in the workflow directory.

        Args:
            workflow_dir: Directory containing workflows. Defaults to data/workflows.
            status: Optional status filter.

        Returns:
            List of workflows.
        """
        if workflow_dir is None:
            workflow_dir = DEFAULT_WORKFLOW_DIR
        workflow_dir = Path(workflow_dir)

        if not workflow_dir.exists():
            return []

        workflows = []
        for file_path in workflow_dir.glob("*.json"):
            try:
                with open(file_path, encoding="utf-8") as f:
                    wf = cls.model_validate_json(f.read())
                    if status is None or wf.status == status:
                        workflows.append(wf)
            except Exception as e:
                logger.warning("Failed to load workflow from %s: %s", file_path, e)

        return sorted(workflows, key=lambda w: w.created_at, reverse=True)


# Type alias for condition functions
ConditionFunc = Callable[[dict[str, Any]], bool]


# Registry of condition functions
_condition_registry: dict[str, ConditionFunc] = {}


def register_condition(name: str, func: ConditionFunc) -> None:
    """Register a condition function by name.

    Condition functions are used for preconditions and postconditions
    in workflow steps. They receive the workflow state dict and return
    True if the condition passes, False otherwise.

    Args:
        name: Name to register the function under.
        func: The condition function.
    """
    _condition_registry[name] = func
    logger.debug("Registered condition function: %s", name)


def get_condition(name: str) -> ConditionFunc | None:
    """Get a registered condition function by name.

    Args:
        name: Name of the condition function.

    Returns:
        The condition function or None if not found.
    """
    return _condition_registry.get(name)


def clear_condition_registry() -> None:
    """Clear all registered condition functions."""
    _condition_registry.clear()


# Built-in condition functions
def _always_true(state: dict[str, Any]) -> bool:
    """Condition that always returns True."""
    return True


def _always_false(state: dict[str, Any]) -> bool:
    """Condition that always returns False."""
    return False


def _state_has_key(key: str) -> ConditionFunc:
    """Create a condition that checks if a key exists in state."""
    def check(state: dict[str, Any]) -> bool:
        return key in state
    return check


# Register built-in conditions
register_condition("always_true", _always_true)
register_condition("always_false", _always_false)


__all__ = [
    "WorkflowStep",
    "Workflow",
    "WorkflowApproval",
    "StepResult",
    "generate_workflow_id",
    "generate_step_id",
    "generate_approval_id",
    "register_condition",
    "get_condition",
    "clear_condition_registry",
    "ConditionFunc",
    "DEFAULT_WORKFLOW_DIR",
    "DEFAULT_APPROVAL_DIR",
]
