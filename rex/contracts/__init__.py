"""Rex core contracts and schemas.

This module provides versioned Pydantic models that define the internal
contracts used across all Rex components: voice node, web dashboard,
gateway API, tool adapters, audit log, and scheduler.

Usage:
    from rex.contracts import (
        CONTRACT_VERSION,
        Task, Action, ToolCall, ToolResult,
        Approval, Notification, EvidenceRef,
        RiskLevel,  # Used by policy engine
    )

See Also:
    rex.policy: Policy models (ActionPolicy, PolicyDecision)
    rex.policy_engine: Policy engine for tool call evaluation
"""

from rex.contracts.version import CONTRACT_VERSION, get_version_info
from rex.contracts.core import (
    EvidenceRef,
    ToolCall,
    ToolResult,
    Approval,
    Action,
    Task,
    Notification,
    EvidenceKind,
    ApprovalStatus,
    RiskLevel,
    TaskStatus,
    NotificationChannel,
    NotificationPriority,
    redact_sensitive_keys,
)

__all__ = [
    # Version
    "CONTRACT_VERSION",
    "get_version_info",
    # Enums
    "EvidenceKind",
    "ApprovalStatus",
    "RiskLevel",
    "TaskStatus",
    "NotificationChannel",
    "NotificationPriority",
    # Models
    "EvidenceRef",
    "ToolCall",
    "ToolResult",
    "Approval",
    "Action",
    "Task",
    "Notification",
    # Utilities
    "redact_sensitive_keys",
]
