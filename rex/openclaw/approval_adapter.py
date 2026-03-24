"""OpenClaw approval adapter — US-P3-018.

Wraps Rex's file-based approval system (:class:`~rex.workflow.WorkflowApproval`,
:func:`~rex.workflow_runner.approve_workflow`, :func:`~rex.workflow_runner.deny_workflow`,
:func:`~rex.workflow_runner.list_pending_approvals`) as a single object that can
be wired into an OpenClaw agent's approval flow.

When the ``openclaw`` package is not installed, :func:`register` logs a warning
and returns ``None``.  The adapter methods work without OpenClaw installed.

Typical usage::

    from rex.openclaw.approval_adapter import ApprovalAdapter
    from pathlib import Path

    adapter = ApprovalAdapter(approval_dir=Path("data/approvals"))

    # Create and persist an approval request
    approval = adapter.create(
        workflow_id="wf_001",
        step_id="step_001",
        step_description="Send email to customer",
        requested_by="workflow_runner",
    )

    # List what is pending
    pending = adapter.list_pending()

    # Approve or deny
    adapter.approve(approval.approval_id, decided_by="user")
    adapter.deny(approval.approval_id, decided_by="user", reason="Not now")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from rex.workflow import DEFAULT_APPROVAL_DIR, WorkflowApproval, generate_approval_id
from rex.workflow_runner import approve_workflow, deny_workflow, list_pending_approvals

logger = logging.getLogger(__name__)


class ApprovalAdapter:
    """Adapter that presents Rex's file-based approval system to OpenClaw.

    Bridges :class:`~rex.workflow.WorkflowApproval` save/load and the
    ``approve_workflow`` / ``deny_workflow`` / ``list_pending_approvals``
    helpers from :mod:`rex.workflow_runner` under a single injectable object.

    When ``openclaw`` is installed, :meth:`register` registers the adapter
    so that OpenClaw can route approval decisions through Rex's approval
    system (stub — filled in once the OpenClaw approval API is confirmed,
    see PRD §8.3).

    Args:
        approval_dir: Directory used for persisting approval records.
            Defaults to :data:`~rex.workflow.DEFAULT_APPROVAL_DIR`
            (``data/approvals``).  Pass a :class:`~pathlib.Path` pointing
            to a temporary directory in tests for isolation.
    """

    def __init__(self, approval_dir: Path | str | None = None) -> None:
        self._approval_dir: Path = (
            Path(approval_dir) if approval_dir is not None else DEFAULT_APPROVAL_DIR
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def approval_dir(self) -> Path:
        """The directory where approval JSON files are stored."""
        return self._approval_dir

    # ------------------------------------------------------------------
    # Core approval methods
    # ------------------------------------------------------------------

    def create(
        self,
        workflow_id: str,
        step_id: str,
        *,
        step_description: str | None = None,
        tool_call_summary: str | None = None,
        requested_by: str | None = "workflow_runner",
    ) -> WorkflowApproval:
        """Create a new *pending* approval and persist it to disk.

        Args:
            workflow_id: ID of the workflow this approval belongs to.
            step_id: ID of the step requiring approval.
            step_description: Human-readable description of the step.
            tool_call_summary: Summary of the tool call for context.
            requested_by: Who/what requested the approval.

        Returns:
            The saved :class:`~rex.workflow.WorkflowApproval` object.
        """
        approval = WorkflowApproval(
            approval_id=generate_approval_id(),
            workflow_id=workflow_id,
            step_id=step_id,
            status="pending",
            step_description=step_description,
            tool_call_summary=tool_call_summary,
            requested_by=requested_by,
        )
        approval.save(self._approval_dir)
        logger.debug(
            "Created approval %s for workflow %s step %s",
            approval.approval_id,
            workflow_id,
            step_id,
        )
        return approval

    def load(self, approval_id: str) -> WorkflowApproval | None:
        """Load an approval record by ID.

        Args:
            approval_id: The approval ID to load.

        Returns:
            The :class:`~rex.workflow.WorkflowApproval` if found, else
            ``None``.
        """
        return WorkflowApproval.load(approval_id, self._approval_dir)

    def approve(
        self,
        approval_id: str,
        *,
        decided_by: str = "user",
        reason: str | None = None,
    ) -> bool:
        """Approve a pending approval.

        Delegates to :func:`~rex.workflow_runner.approve_workflow`.

        Args:
            approval_id: ID of the approval to approve.
            decided_by: Who approved (default ``"user"``).
            reason: Optional approval reason.

        Returns:
            ``True`` if the approval was found and updated, ``False``
            otherwise.
        """
        return approve_workflow(
            approval_id,
            decided_by=decided_by,
            reason=reason,
            approval_dir=self._approval_dir,
        )

    def deny(
        self,
        approval_id: str,
        *,
        decided_by: str = "user",
        reason: str | None = None,
    ) -> bool:
        """Deny a pending approval.

        Delegates to :func:`~rex.workflow_runner.deny_workflow`.

        Args:
            approval_id: ID of the approval to deny.
            decided_by: Who denied (default ``"user"``).
            reason: Optional denial reason.

        Returns:
            ``True`` if the approval was found and updated, ``False``
            otherwise.
        """
        return deny_workflow(
            approval_id,
            decided_by=decided_by,
            reason=reason,
            approval_dir=self._approval_dir,
        )

    def list_pending(self) -> list[WorkflowApproval]:
        """Return all pending approvals sorted by requested time.

        Delegates to :func:`~rex.workflow_runner.list_pending_approvals`.

        Returns:
            List of :class:`~rex.workflow.WorkflowApproval` objects with
            ``status == "pending"``, sorted oldest-first.
        """
        return list_pending_approvals(self._approval_dir)

    # ------------------------------------------------------------------
    # OpenClaw registration
    # ------------------------------------------------------------------

    def register(self, agent: Any = None) -> Any:
        """Register this adapter with an OpenClaw agent.

        When ``openclaw`` is installed, this method wires the adapter into
        OpenClaw's approval lifecycle so that approval decisions flow through
        Rex's file-based system.  When OpenClaw is absent, logs a warning
        and returns ``None``.

        .. note::
            The exact OpenClaw approval registration call is a stub (see PRD
            §8.3 — *"Confirm OpenClaw's hook/middleware registration API"*).
            Replace the ``# TODO`` below once the API is confirmed.

        Args:
            agent: Optional OpenClaw agent handle.

        Returns:
            The registration handle from OpenClaw, or ``None``.
        """
        from rex.config import load_config as _load_config
        from rex.openclaw.http_client import get_openclaw_client

        if get_openclaw_client(_load_config()) is None:
            logger.warning(
                "OpenClaw gateway not configured — ApprovalAdapter not registered",
            )
            return None

        # TODO: replace with real OpenClaw approval registration once API is confirmed.
        # Expected shape (to be verified):
        #   handle = _openclaw.register_approval_handler(
        #       approve=self.approve,
        #       deny=self.deny,
        #       list_pending=self.list_pending,
        #       agent=agent,
        #   )
        #   return handle
        logger.warning(
            "OpenClaw approval registration stub — update once API is confirmed (PRD §8.3)"
        )
        return None


__all__ = ["ApprovalAdapter"]
