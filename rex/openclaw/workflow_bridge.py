"""OpenClaw workflow bridge — US-P4-030.

Wraps Rex's :class:`~rex.workflow_runner.WorkflowRunner` to route workflow
execution through the OpenClaw bridge layer while preserving all Rex-specific
semantics:

- Policy gating via :class:`~rex.policy_engine.PolicyEngine`
- Approval gates via :class:`~rex.workflow_runner.WorkflowApproval`
- Precondition / postcondition evaluation
- Idempotency checking
- Disk persistence (``data/workflows/{id}.json``)
- Dry-run preview mode
- Pre/post-step hooks

The bridge presents the same constructor and method surface as
:class:`~rex.workflow_runner.WorkflowRunner` so that callers can substitute
it without behavioural change.

When the ``openclaw`` package is not installed, :meth:`register` logs a
warning and returns ``None``.  All other methods work without OpenClaw
installed because they delegate to the existing Rex workflow runner.

.. note::
    A formal ``WorkflowRunnerProtocol`` in ``rex.contracts`` has not been
    defined yet.  The bridge satisfies the interface structurally (duck
    typing).  A contract should be added in a future iteration.

Typical usage::

    from rex.openclaw.workflow_bridge import WorkflowBridge
    from rex.workflow import Workflow

    bridge = WorkflowBridge(workflow)
    result = bridge.run()

    # Dry-run preview
    preview = bridge.dry_run()
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from rex.workflow import Workflow, WorkflowStep
from rex.workflow_runner import (
    DryRunResult,
    RunResult,
    WorkflowRunner,
)

logger = logging.getLogger(__name__)


class WorkflowBridge:
    """Thin wrapper around :class:`~rex.workflow_runner.WorkflowRunner`.

    Presents the same constructor and ``run`` / ``dry_run`` interface as
    :class:`~rex.workflow_runner.WorkflowRunner` so that callers route through
    the bridge without behavioural change.

    All Rex-specific execution semantics are preserved — policy checks,
    approval gates, preconditions, postconditions, idempotency, persistence,
    and dry-run mode all delegate intact to the underlying runner.

    Args:
        workflow: The :class:`~rex.workflow.Workflow` to execute.
        policy_engine: Optional policy engine.  Uses Rex singleton if absent.
        audit_logger: Optional audit logger.  Uses Rex singleton if absent.
        workflow_dir: Directory for workflow persistence.
        approval_dir: Directory for approval persistence.
        default_context: Default context forwarded to each tool call.
        pre_step_hook: Called before each step begins.
        before_tool_call_hook: Called just before a tool call is executed.
        after_tool_call_hook: Called after a tool call completes.
    """

    def __init__(
        self,
        workflow: Workflow,
        *,
        policy_engine: Any = None,
        audit_logger: Any = None,
        workflow_dir: Path | str | None = None,
        approval_dir: Path | str | None = None,
        default_context: dict[str, Any] | None = None,
        pre_step_hook: Callable[[WorkflowStep], None] | None = None,
        before_tool_call_hook: Callable[[WorkflowStep], None] | None = None,
        after_tool_call_hook: Callable[[WorkflowStep, Any], None] | None = None,
    ) -> None:
        self._runner = WorkflowRunner(
            workflow,
            policy_engine=policy_engine,
            audit_logger=audit_logger,
            workflow_dir=workflow_dir,
            approval_dir=approval_dir,
            default_context=default_context,
            pre_step_hook=pre_step_hook,
            before_tool_call_hook=before_tool_call_hook,
            after_tool_call_hook=after_tool_call_hook,
        )

    # ------------------------------------------------------------------
    # WorkflowRunner delegation
    # ------------------------------------------------------------------

    def run(self) -> RunResult:
        """Run the workflow from its current state.

        Delegates to :meth:`~rex.workflow_runner.WorkflowRunner.run`.

        Returns:
            :class:`~rex.workflow_runner.RunResult` with execution summary.
        """
        return self._runner.run()

    def dry_run(self) -> DryRunResult:
        """Preview workflow execution without making changes.

        Delegates to :meth:`~rex.workflow_runner.WorkflowRunner.dry_run`.

        Returns:
            :class:`~rex.workflow_runner.DryRunResult` with preview details.
        """
        return self._runner.dry_run()

    @property
    def workflow(self) -> Workflow:
        """The underlying :class:`~rex.workflow.Workflow` instance."""
        return self._runner.workflow

    @property
    def runner(self) -> WorkflowRunner:
        """The underlying :class:`~rex.workflow_runner.WorkflowRunner` instance."""
        return self._runner

    # ------------------------------------------------------------------
    # OpenClaw registration
    # ------------------------------------------------------------------

    def register(self, agent: Any = None) -> Any:
        """Register this bridge as the OpenClaw workflow executor.

        When ``use_openclaw_tools`` is False (the default), registration is a
        no-op and this method returns ``None``.

        When ``use_openclaw_tools`` is True, the workflow executor registration
        API is not yet available, so this method raises
        :class:`~rex.openclaw.errors.OpenClawConfigError` so callers fail
        closed rather than silently bypassing OpenClaw.

        Args:
            agent: Optional OpenClaw agent handle (reserved for future use).

        Returns:
            ``None`` when OpenClaw tools are disabled.

        Raises:
            :class:`~rex.openclaw.errors.OpenClawConfigError`: When
                ``use_openclaw_tools`` is True and the workflow executor API is
                not available.
        """
        from rex.config import load_config as _load_config
        from rex.openclaw.errors import OpenClawConfigError

        cfg = _load_config()
        if not cfg.use_openclaw_tools:
            logger.info("OpenClaw tools disabled — workflow executor registration skipped")
            return None

        raise OpenClawConfigError("workflow executor not available")
