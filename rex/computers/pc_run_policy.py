"""Policy and approval integration for ``rex pc run``.

This module gates remote command execution through the Rex policy engine and
approval system before any network call is made.

Flow
----
1. ``check_pc_run_policy()`` is called by the CLI with the target computer,
   command, and arguments.
2. The policy engine is consulted for the ``"pc_run"`` tool:
   - ``denied``           → refuse immediately, no approval created.
   - ``requires_approval``→ look for an existing approved/pending approval
     record in ``data/approvals/``; if none, create a new pending record and
     tell the user how to approve it.
   - auto-execute (custom low-risk policy only) → allow to proceed.
3. The caller then applies the ``--yes`` guard as a second layer.
4. After the user runs ``rex approvals --approve <id>`` and re-runs the
   command, the existing approved approval is found and execution proceeds.

Approval payload includes (no secrets stored):
- ``computer_id``          — target computer ID
- ``command``              — base command name
- ``args``                 — argument list
- ``allowlist_decision``   — "allowed" or "denied" (client-side check outcome)
- ``allowlist_rule_matched`` — which allowlist entry matched (if allowed)
- ``initiated_by``         — user identity from session/config, or "unknown"

The approval record reuses ``WorkflowApproval`` from ``rex.workflow`` with:
- ``workflow_id = "pc_run"``  (constant sentinel for pc-run approvals)
- ``step_id``  = deterministic SHA-256 prefix over (computer_id, command, args)
  so re-runs find the same pending/approved record without an index file.

Auth tokens are **never** stored in the approval payload.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from rex.contracts import ToolCall
from rex.policy import PolicyDecision
from rex.workflow import DEFAULT_APPROVAL_DIR, WorkflowApproval, generate_approval_id

if TYPE_CHECKING:
    from rex.policy_engine import PolicyEngine

logger = logging.getLogger(__name__)

# Tool name registered in the policy engine for pc run actions.
PC_RUN_TOOL_NAME = "pc_run"

# Sentinel workflow_id written into every pc-run approval record.
PC_RUN_WORKFLOW_ID = "pc_run"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _command_step_id(computer_id: str, command: str, args: list[str]) -> str:
    """Return a deterministic step_id for a (computer_id, command, args) triple.

    Uses a 16-hex-char SHA-256 prefix so the same command on the same computer
    always maps to the same step_id, allowing re-runs to find existing approvals
    without a separate index.

    Args:
        computer_id: Target computer ID from config.
        command: Base command name (no arguments).
        args: Argument list.

    Returns:
        A string like ``"pc_desktop_a1b2c3d4e5f6g7h8"``.
    """
    key = json.dumps(
        {"cid": computer_id, "cmd": command, "args": args},
        sort_keys=True,
    )
    h = hashlib.sha256(key.encode()).hexdigest()[:16]
    return f"pc_{computer_id}_{h}"


def _build_tool_call_summary(
    computer_id: str,
    command: str,
    args: list[str],
    allowlist_matched: bool,
    allowlist_rule: str | None,
    initiated_by: str | None,
) -> str:
    """Build a redacted JSON summary for the approval payload.

    Auth tokens are intentionally excluded.  The summary is stored verbatim
    in the approval file on disk and shown in ``rex approvals --show <id>``.

    Args:
        computer_id: Target computer ID.
        command: Base command name.
        args: Argument list.
        allowlist_matched: Whether the command passed the client-side allowlist.
        allowlist_rule: Which allowlist entry matched (the command name itself),
            or ``None`` if not matched.
        initiated_by: Resolved user identity, or ``None``.

    Returns:
        A JSON-formatted string.
    """
    return json.dumps(
        {
            "computer_id": computer_id,
            "command": command,
            "args": args,
            "allowlist_decision": "allowed" if allowlist_matched else "denied",
            "allowlist_rule_matched": allowlist_rule,
            "initiated_by": initiated_by or "unknown",
        },
        indent=2,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def find_pending_or_approved_approval(
    computer_id: str,
    command: str,
    args: list[str],
    approval_dir: Path | str | None = None,
) -> WorkflowApproval | None:
    """Search ``approval_dir`` for an existing pending or approved pc-run approval.

    Scans files in modification-time descending order so the most recent
    matching approval is returned first.

    Args:
        computer_id: Target computer ID.
        command: Base command name.
        args: Argument list.
        approval_dir: Directory containing approval JSON files.
            Defaults to ``data/approvals``.

    Returns:
        A :class:`~rex.workflow.WorkflowApproval` with status ``"pending"`` or
        ``"approved"``, or ``None`` if no matching record is found.
    """
    if approval_dir is None:
        approval_dir = DEFAULT_APPROVAL_DIR
    approval_dir = Path(approval_dir)
    if not approval_dir.exists():
        return None

    step_id = _command_step_id(computer_id, command, args)

    # Sort by mtime descending so the newest matching record wins.
    try:
        files = sorted(
            approval_dir.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
    except OSError:
        return None

    for file_path in files:
        try:
            with open(file_path, encoding="utf-8") as f:
                approval = WorkflowApproval.model_validate_json(f.read())
            if (
                approval.workflow_id == PC_RUN_WORKFLOW_ID
                and approval.step_id == step_id
                and approval.status in ("pending", "approved")
            ):
                return approval
        except Exception:  # noqa: BLE001
            continue

    return None


def check_pc_run_policy(
    computer_id: str,
    command: str,
    args: list[str],
    *,
    allowlist_matched: bool = True,
    allowlist_rule: str | None = None,
    initiated_by: str | None = None,
    policy_engine: PolicyEngine | None = None,
    approval_dir: Path | str | None = None,
) -> tuple[PolicyDecision, WorkflowApproval | None]:
    """Evaluate policy for a ``rex pc run`` action.

    This is the main entry-point for policy + approval gating.  The CLI calls
    this **before** the ``--yes`` guard and **before** any network call.

    Args:
        computer_id: Target computer ID.
        command: Base command name (no arguments).
        args: Argument list.
        allowlist_matched: Whether the command passed the client-side allowlist
            check (used only for the approval payload; denial is enforced
            separately by :class:`~rex.computers.service.ComputerService`).
        allowlist_rule: The allowlist entry that matched (typically equal to
            *command* when matched), or ``None``.
        initiated_by: Resolved user identity for the approval record.
        policy_engine: Override the shared :class:`~rex.policy_engine.PolicyEngine`
            singleton.  Primarily for testing.
        approval_dir: Override the approval storage directory.  Defaults to
            ``data/approvals``.

    Returns:
        A ``(decision, approval)`` tuple:

        - ``(denied_decision, None)``   — policy explicitly denies; stop.
        - ``(auto_decision, None)``     — policy allows auto-execute; proceed.
        - ``(approval_decision, approval)`` where ``approval.status == "approved"``
          — an existing approved approval was found; proceed (apply ``--yes``).
        - ``(approval_decision, approval)`` where ``approval.status == "pending"``
          — a new (or existing unactioned) pending approval was saved; show the
          approval ID and wait.
    """
    if policy_engine is None:
        from rex.policy_engine import get_policy_engine

        policy_engine = get_policy_engine()

    tool_call = ToolCall(
        tool=PC_RUN_TOOL_NAME,
        args={"computer_id": computer_id, "command": command, "args": args},
        requested_by=initiated_by,
    )
    decision = policy_engine.decide(tool_call, metadata={})

    if decision.denied:
        logger.info(
            "pc_run denied by policy for computer=%s command=%r: %s",
            computer_id,
            command,
            decision.reason,
        )
        return decision, None

    if not decision.requires_approval:
        # Custom policy configured auto-execute (e.g. low-risk override).
        logger.debug(
            "pc_run auto-execute allowed for computer=%s command=%r",
            computer_id,
            command,
        )
        return decision, None

    # Requires approval — look for an existing pending or approved record.
    existing = find_pending_or_approved_approval(
        computer_id=computer_id,
        command=command,
        args=args,
        approval_dir=approval_dir,
    )
    if existing is not None:
        logger.debug(
            "Found existing pc_run approval %s (status=%s) for computer=%s command=%r",
            existing.approval_id,
            existing.status,
            computer_id,
            command,
        )
        return decision, existing

    # No existing record — create a new pending approval.
    step_id = _command_step_id(computer_id, command, args)
    summary = _build_tool_call_summary(
        computer_id=computer_id,
        command=command,
        args=args,
        allowlist_matched=allowlist_matched,
        allowlist_rule=allowlist_rule,
        initiated_by=initiated_by,
    )
    approval = WorkflowApproval(
        approval_id=generate_approval_id(),
        workflow_id=PC_RUN_WORKFLOW_ID,
        step_id=step_id,
        status="pending",
        requested_by=initiated_by or "cli",
        step_description=f"Remote command execution on {computer_id!r}: {command}",
        tool_call_summary=summary,
    )
    approval.save(approval_dir)

    logger.info(
        "Created pc_run approval %s for computer=%s command=%r",
        approval.approval_id,
        computer_id,
        command,
    )
    return decision, approval


__all__ = [
    "PC_RUN_TOOL_NAME",
    "PC_RUN_WORKFLOW_ID",
    "check_pc_run_policy",
    "find_pending_or_approved_approval",
]
