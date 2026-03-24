# OPENCLAW-WRAP: This module will be wrapped around OpenClaw. Preserve public API.

"""Policy models for the Rex permissions and policy engine.

This module defines Pydantic models for managing tool and action safety policies,
including risk classification, allow/deny lists, and policy decisions.

The policy system enforces least-privilege principles by:
- Classifying actions by risk level (low, medium, high)
- Maintaining allow/deny lists for recipients and domains
- Requiring approval for higher-risk actions
- Auto-executing low-risk actions when permitted

Usage:
    from rex.policy import ActionPolicy, PolicyDecision
    from rex.contracts import RiskLevel

    policy = ActionPolicy(
        tool_name="send_email",
        risk=RiskLevel.MEDIUM,
        allow_auto=False,
    )
"""

from __future__ import annotations

from pydantic import BaseModel, Field

# Re-export RiskLevel from contracts for convenience
from rex.contracts import RiskLevel


class ActionPolicy(BaseModel):
    """Policy configuration for a specific tool or action.

    Defines the risk level and access control rules for a tool,
    including whether it can auto-execute and which recipients/domains
    are allowed or denied.
    """

    tool_name: str = Field(
        ...,
        description="Name of the tool this policy applies to",
    )
    risk: RiskLevel = Field(
        default=RiskLevel.LOW,
        description="Risk level classification for this tool",
    )
    allow_auto: bool = Field(
        default=False,
        description="Whether this tool can auto-execute without user approval",
    )
    allowed_recipients: list[str] | None = Field(
        default=None,
        description="List of allowed recipient identifiers (e.g., email addresses). "
        "None means all recipients are allowed unless in denied list.",
    )
    denied_recipients: list[str] | None = Field(
        default=None,
        description="List of denied recipient identifiers. "
        "If a recipient is in this list, the action is denied.",
    )
    allowed_domains: list[str] | None = Field(
        default=None,
        description="List of allowed domains (e.g., 'company.com'). "
        "None means all domains are allowed unless in denied list.",
    )
    denied_domains: list[str] | None = Field(
        default=None,
        description="List of denied domains. " "If a domain is in this list, the action is denied.",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "tool_name": "send_email",
                    "risk": "medium",
                    "allow_auto": False,
                    "allowed_recipients": None,
                    "denied_recipients": ["spam@example.com"],
                    "allowed_domains": None,
                    "denied_domains": ["malicious.com"],
                },
                {
                    "tool_name": "time_now",
                    "risk": "low",
                    "allow_auto": True,
                    "allowed_recipients": None,
                    "denied_recipients": None,
                    "allowed_domains": None,
                    "denied_domains": None,
                },
            ]
        }
    }


class PolicyDecision(BaseModel):
    """Result of a policy evaluation for a tool call.

    Represents the decision made by the policy engine, indicating
    whether the action is allowed, denied, or requires user approval.

    Decision outcomes:
    - allowed=True, denied=False, requires_approval=False: Auto-execute
    - allowed=True, denied=False, requires_approval=True: Needs user approval
    - allowed=False, denied=True, requires_approval=False: Blocked
    """

    allowed: bool = Field(
        ...,
        description="Whether the action is permitted to proceed (after any required approval)",
    )
    reason: str = Field(
        ...,
        description="Human-readable explanation of the policy decision",
    )
    requires_approval: bool = Field(
        default=False,
        description="Whether user approval is required before execution",
    )
    denied: bool = Field(
        default=False,
        description="Whether the action is explicitly denied and should not proceed",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "allowed": True,
                    "reason": "Low-risk tool with auto-execute enabled",
                    "requires_approval": False,
                    "denied": False,
                },
                {
                    "allowed": True,
                    "reason": "Medium-risk action requires user approval",
                    "requires_approval": True,
                    "denied": False,
                },
                {
                    "allowed": False,
                    "reason": "Recipient is on the deny list",
                    "requires_approval": False,
                    "denied": True,
                },
            ]
        }
    }


__all__ = [
    "RiskLevel",
    "ActionPolicy",
    "PolicyDecision",
]
