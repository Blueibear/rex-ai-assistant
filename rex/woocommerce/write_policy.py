"""Policy and approval integration for WooCommerce write actions (Cycle 6.3).

This module gates destructive WooCommerce API calls through the Rex policy
engine and approval system before any network call is made.

Supported write actions
-----------------------
- ``wc_order_set_status``  — PUT /wc/v3/orders/<id>  (HIGH risk)
- ``wc_coupon_create``     — POST /wc/v3/coupons      (HIGH risk)
- ``wc_coupon_disable``    — PUT /wc/v3/coupons/<id>  (HIGH risk)

Flow
----
1. ``check_wc_write_policy()`` is called by the CLI with the action name,
   site ID, and action-specific parameters.
2. The policy engine is consulted for the relevant tool name:
   - ``denied``            → refuse immediately, no approval created.
   - ``requires_approval`` → look for an existing approved/pending approval
     in ``data/approvals/``; if none, create a new pending record and tell
     the user how to approve it.
   - auto-execute (custom low-risk policy only) → allow to proceed.
3. The caller then applies the ``--yes`` guard as a second layer.
4. After the user runs ``rex approvals --approve <id>`` and re-runs the
   command, the existing approved approval is found and execution proceeds.

Approval payload includes (no secrets stored):
- ``action``      — tool name (e.g. ``"wc_order_set_status"``)
- ``site_id``     — WooCommerce site identifier
- action-specific fields (order_id, status, coupon_code, etc.)
- ``initiated_by`` — user identity from session/config, or ``"unknown"``

The approval record reuses ``WorkflowApproval`` from ``rex.workflow`` with:
- ``workflow_id = WC_WRITE_WORKFLOW_ID``   (constant sentinel)
- ``step_id``  = deterministic SHA-256 prefix over the action + site_id +
  relevant identifiers so re-runs find the same pending/approved record.

Consumer keys/secrets are **never** stored in the approval payload.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rex.contracts import ToolCall
from rex.policy import PolicyDecision
from rex.workflow import DEFAULT_APPROVAL_DIR, WorkflowApproval, generate_approval_id

if TYPE_CHECKING:
    from rex.policy_engine import PolicyEngine

logger = logging.getLogger(__name__)

# Sentinel workflow_id written into every WC write approval record.
WC_WRITE_WORKFLOW_ID = "wc_write"

# Tool names for the three write actions.
WC_ORDER_SET_STATUS_TOOL = "wc_order_set_status"
WC_COUPON_CREATE_TOOL = "wc_coupon_create"
WC_COUPON_DISABLE_TOOL = "wc_coupon_disable"

# All known WC write tool names (used for filtering when scanning approvals).
WC_WRITE_TOOL_NAMES = frozenset(
    {WC_ORDER_SET_STATUS_TOOL, WC_COUPON_CREATE_TOOL, WC_COUPON_DISABLE_TOOL}
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _action_step_id(action: str, site_id: str, identifiers: dict[str, Any]) -> str:
    """Return a deterministic step_id for a (action, site_id, identifiers) triple.

    Uses a 16-hex-char SHA-256 prefix so the same write action on the same
    site with the same parameters always maps to the same step_id, allowing
    re-runs to find existing approvals without a separate index.

    Args:
        action: Tool name (e.g. ``"wc_order_set_status"``).
        site_id: WooCommerce site identifier.
        identifiers: Action-specific stable identifiers (e.g. order_id + status,
            coupon code + amount + type, or coupon_id). Must be JSON-serialisable.

    Returns:
        A string like ``"wc_myshop_a1b2c3d4e5f6g7h8"``.
    """
    key = json.dumps(
        {"action": action, "site": site_id, "ids": identifiers},
        sort_keys=True,
    )
    h = hashlib.sha256(key.encode()).hexdigest()[:16]
    return f"wc_{site_id}_{h}"


def _build_approval_summary(
    action: str,
    site_id: str,
    params: dict[str, Any],
    initiated_by: str | None,
) -> str:
    """Build a redacted JSON summary for the approval payload.

    Consumer keys/secrets are intentionally excluded.  The summary is stored
    verbatim in the approval file on disk and shown in ``rex approvals --show <id>``.

    Args:
        action: Tool name.
        site_id: WooCommerce site identifier.
        params: Action-specific parameters (no credentials).
        initiated_by: Resolved user identity, or ``None``.

    Returns:
        A JSON-formatted string.
    """
    return json.dumps(
        {
            "action": action,
            "site_id": site_id,
            **params,
            "initiated_by": initiated_by or "unknown",
        },
        indent=2,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def find_pending_or_approved_wc_approval(
    action: str,
    site_id: str,
    identifiers: dict[str, Any],
    approval_dir: Path | str | None = None,
) -> WorkflowApproval | None:
    """Search ``approval_dir`` for an existing pending or approved WC write approval.

    Scans files in modification-time descending order so the most recent
    matching approval is returned first.

    Args:
        action: Tool name (e.g. ``"wc_order_set_status"``).
        site_id: WooCommerce site identifier.
        identifiers: Stable identifiers used to compute the step_id.
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

    step_id = _action_step_id(action, site_id, identifiers)

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
                approval.workflow_id == WC_WRITE_WORKFLOW_ID
                and approval.step_id == step_id
                and approval.status in ("pending", "approved")
            ):
                return approval
        except Exception:  # noqa: BLE001
            continue

    return None


def check_wc_write_policy(
    action: str,
    site_id: str,
    identifiers: dict[str, Any],
    params: dict[str, Any],
    *,
    step_description: str,
    initiated_by: str | None = None,
    policy_engine: PolicyEngine | None = None,
    approval_dir: Path | str | None = None,
) -> tuple[PolicyDecision, WorkflowApproval | None]:
    """Evaluate policy for a WooCommerce write action.

    This is the main entry-point for policy + approval gating.  The CLI calls
    this **before** the ``--yes`` guard and **before** any network call.

    Args:
        action: Tool name (e.g. ``"wc_order_set_status"``).
        site_id: WooCommerce site identifier.
        identifiers: Stable identifiers used for deterministic step_id
            computation (e.g. ``{"order_id": 101, "status": "completed"}``).
        params: Full action parameters for the approval payload (no credentials).
        step_description: Human-readable description for the approval record.
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
    # Resolve approval_dir early so that monkeypatching
    # rex.woocommerce.write_policy.DEFAULT_APPROVAL_DIR in tests is effective
    # for ALL downstream calls (find + save).
    if approval_dir is None:
        approval_dir = DEFAULT_APPROVAL_DIR
    approval_dir = Path(approval_dir)

    if policy_engine is None:
        from rex.policy_engine import get_policy_engine

        policy_engine = get_policy_engine()

    tool_call = ToolCall(
        tool=action,
        args={"site_id": site_id, **params},
        requested_by=initiated_by,
    )
    decision = policy_engine.decide(tool_call, metadata={})

    if decision.denied:
        logger.info(
            "WC write action %s denied by policy for site=%s: %s",
            action,
            site_id,
            decision.reason,
        )
        return decision, None

    if not decision.requires_approval:
        logger.debug(
            "WC write action %s auto-execute allowed for site=%s",
            action,
            site_id,
        )
        return decision, None

    # Requires approval — look for an existing pending or approved record.
    existing = find_pending_or_approved_wc_approval(
        action=action,
        site_id=site_id,
        identifiers=identifiers,
        approval_dir=approval_dir,
    )
    if existing is not None:
        logger.debug(
            "Found existing WC write approval %s (status=%s) for action=%s site=%s",
            existing.approval_id,
            existing.status,
            action,
            site_id,
        )
        return decision, existing

    # No existing record — create a new pending approval.
    step_id = _action_step_id(action, site_id, identifiers)
    summary = _build_approval_summary(
        action=action,
        site_id=site_id,
        params=params,
        initiated_by=initiated_by,
    )
    approval = WorkflowApproval(
        approval_id=generate_approval_id(),
        workflow_id=WC_WRITE_WORKFLOW_ID,
        step_id=step_id,
        status="pending",
        requested_by=initiated_by or "cli",
        step_description=step_description,
        tool_call_summary=summary,
    )
    approval.save(approval_dir)

    logger.info(
        "Created WC write approval %s for action=%s site=%s",
        approval.approval_id,
        action,
        site_id,
    )
    return decision, approval


__all__ = [
    "WC_COUPON_CREATE_TOOL",
    "WC_COUPON_DISABLE_TOOL",
    "WC_ORDER_SET_STATUS_TOOL",
    "WC_WRITE_TOOL_NAMES",
    "WC_WRITE_WORKFLOW_ID",
    "check_wc_write_policy",
    "find_pending_or_approved_wc_approval",
]
