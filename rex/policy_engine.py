"""Policy engine for Rex permissions and action safety.

This module provides the PolicyEngine class that evaluates tool calls against
configured policies to determine whether actions should auto-execute, require
approval, or be denied.

The engine supports:
- Default policies for common tools
- Custom policy extension via initialization
- Recipient and domain-based access control
- Risk-based approval requirements

Future Enhancement:
    Policies could be loaded from a JSON or YAML config file at
    `config/policy.json` or `config/policy.yaml`. The format would be:

    {
        "policies": [
            {
                "tool_name": "send_email",
                "risk": "medium",
                "allow_auto": false,
                "denied_domains": ["malicious.com"]
            }
        ]
    }

    This dynamic loading can be implemented in a later chunk by adding
    a `load_from_file()` class method.

Usage:
    from rex.policy_engine import PolicyEngine
    from rex.contracts import ToolCall

    engine = PolicyEngine()
    tool_call = ToolCall(tool="send_email", args={"to": "user@example.com"})
    decision = engine.decide(tool_call, metadata={"recipient": "user@example.com"})

    if decision.denied:
        raise PermissionError(decision.reason)
    elif decision.requires_approval:
        # Queue for approval
        pass
    else:
        # Auto-execute
        pass
"""

from __future__ import annotations

import logging
from typing import Any

from rex.contracts import RiskLevel, ToolCall
from rex.policy import ActionPolicy, PolicyDecision

logger = logging.getLogger(__name__)


# Default policies for built-in tools
DEFAULT_POLICIES: list[ActionPolicy] = [
    # Low-risk, auto-execute tools
    ActionPolicy(
        tool_name="time_now",
        risk=RiskLevel.LOW,
        allow_auto=True,
    ),
    ActionPolicy(
        tool_name="weather_now",
        risk=RiskLevel.LOW,
        allow_auto=True,
    ),
    ActionPolicy(
        tool_name="web_search",
        risk=RiskLevel.LOW,
        allow_auto=True,
    ),
    # Medium-risk tools (require approval by default)
    ActionPolicy(
        tool_name="send_email",
        risk=RiskLevel.MEDIUM,
        allow_auto=False,
    ),
    ActionPolicy(
        tool_name="calendar_create_event",
        risk=RiskLevel.MEDIUM,
        allow_auto=False,
    ),
    ActionPolicy(
        tool_name="calendar_delete_event",
        risk=RiskLevel.MEDIUM,
        allow_auto=False,
    ),
    # High-risk tools
    ActionPolicy(
        tool_name="execute_command",
        risk=RiskLevel.HIGH,
        allow_auto=False,
    ),
    ActionPolicy(
        tool_name="file_write",
        risk=RiskLevel.HIGH,
        allow_auto=False,
    ),
    ActionPolicy(
        tool_name="file_delete",
        risk=RiskLevel.HIGH,
        allow_auto=False,
    ),
    ActionPolicy(
        tool_name="home_assistant_call_service",
        risk=RiskLevel.MEDIUM,
        allow_auto=False,
    ),
]


class PolicyEngine:
    """Engine for evaluating tool calls against configured policies.

    The PolicyEngine maintains a registry of tool policies and provides
    the `decide()` method to evaluate whether a tool call should be
    auto-executed, require approval, or be denied.

    Attributes:
        policies: Dictionary mapping tool names to their ActionPolicy.
        default_policy: Policy applied to tools without an explicit policy.

    Example:
        >>> engine = PolicyEngine()
        >>> call = ToolCall(tool="time_now", args={})
        >>> decision = engine.decide(call, {})
        >>> decision.allowed
        True
        >>> decision.requires_approval
        False
    """

    def __init__(
        self,
        policies: list[ActionPolicy] | None = None,
        default_policy: ActionPolicy | None = None,
    ) -> None:
        """Initialize the PolicyEngine with policies.

        Args:
            policies: Additional policies to add to the defaults.
                These will override default policies for the same tool name.
            default_policy: Policy to apply for tools without an explicit policy.
                Defaults to a medium-risk policy requiring approval.
        """
        # Start with default policies
        self._policies: dict[str, ActionPolicy] = {
            policy.tool_name: policy for policy in DEFAULT_POLICIES
        }

        # Override/extend with custom policies
        if policies:
            for policy in policies:
                self._policies[policy.tool_name] = policy

        # Default policy for unknown tools
        self._default_policy = default_policy or ActionPolicy(
            tool_name="__default__",
            risk=RiskLevel.MEDIUM,
            allow_auto=False,
        )

    @property
    def policies(self) -> dict[str, ActionPolicy]:
        """Get the current policy registry."""
        return self._policies.copy()

    def get_policy(self, tool_name: str) -> ActionPolicy:
        """Get the policy for a specific tool.

        Args:
            tool_name: Name of the tool to look up.

        Returns:
            The ActionPolicy for the tool, or the default policy if not found.
        """
        return self._policies.get(tool_name, self._default_policy)

    def decide(
        self,
        tool_call: ToolCall,
        metadata: dict[str, Any] | None = None,
    ) -> PolicyDecision:
        """Evaluate a tool call against policies and return a decision.

        Args:
            tool_call: The tool call to evaluate.
            metadata: Additional context for the decision, such as:
                - recipient: Email address or identifier of the recipient
                - domain: Target domain for the action
                - target: Generic target identifier

        Returns:
            A PolicyDecision indicating whether to allow, deny, or require approval.

        Examples:
            >>> engine = PolicyEngine()
            >>> call = ToolCall(tool="send_email", args={})
            >>> decision = engine.decide(call, {"recipient": "user@blocked.com"})
        """
        if metadata is None:
            metadata = {}

        policy = self.get_policy(tool_call.tool)
        logger.debug(
            "Evaluating policy for tool=%s, risk=%s, allow_auto=%s",
            tool_call.tool,
            policy.risk.value,
            policy.allow_auto,
        )

        # Extract target information from metadata
        recipient = metadata.get("recipient", "")
        domain = metadata.get("domain", "")

        # Extract domain from recipient if not explicitly provided
        if not domain and recipient and "@" in recipient:
            domain = recipient.split("@", 1)[1].lower()

        # Check denied recipients first
        if policy.denied_recipients and recipient:
            if self._matches_list(recipient, policy.denied_recipients):
                logger.info(
                    "Denying tool=%s: recipient %s is on deny list", tool_call.tool, recipient
                )
                return PolicyDecision(
                    allowed=False,
                    reason=f"Recipient '{recipient}' is on the deny list",
                    requires_approval=False,
                    denied=True,
                )

        # Check denied domains
        if policy.denied_domains and domain:
            if self._matches_list(domain, policy.denied_domains):
                logger.info("Denying tool=%s: domain %s is on deny list", tool_call.tool, domain)
                return PolicyDecision(
                    allowed=False,
                    reason=f"Domain '{domain}' is on the deny list",
                    requires_approval=False,
                    denied=True,
                )

        # Check allowed recipients (if specified, recipient must be present and in list)
        if policy.allowed_recipients is not None:
            if not recipient:
                logger.info(
                    "Denying tool=%s: recipient required for allowed list",
                    tool_call.tool,
                )
                return PolicyDecision(
                    allowed=False,
                    reason="Recipient is required but missing for allowed list",
                    requires_approval=False,
                    denied=True,
                )
            if not self._matches_list(recipient, policy.allowed_recipients):
                logger.info(
                    "Denying tool=%s: recipient %s not in allowed list",
                    tool_call.tool,
                    recipient,
                )
                return PolicyDecision(
                    allowed=False,
                    reason=f"Recipient '{recipient}' is not in the allowed list",
                    requires_approval=False,
                    denied=True,
                )

        # Check allowed domains (if specified, domain must be present and in list)
        if policy.allowed_domains is not None:
            if not domain:
                logger.info(
                    "Denying tool=%s: domain required for allowed list",
                    tool_call.tool,
                )
                return PolicyDecision(
                    allowed=False,
                    reason="Domain is required but missing for allowed list",
                    requires_approval=False,
                    denied=True,
                )
            if not self._matches_list(domain, policy.allowed_domains):
                logger.info(
                    "Denying tool=%s: domain %s not in allowed list",
                    tool_call.tool,
                    domain,
                )
                return PolicyDecision(
                    allowed=False,
                    reason=f"Domain '{domain}' is not in the allowed list",
                    requires_approval=False,
                    denied=True,
                )

        # Determine if auto-execution is allowed
        if policy.allow_auto and policy.risk == RiskLevel.LOW:
            logger.debug("Auto-executing tool=%s (low risk, auto enabled)", tool_call.tool)
            return PolicyDecision(
                allowed=True,
                reason="Low-risk tool with auto-execute enabled",
                requires_approval=False,
                denied=False,
            )

        # For medium/high risk or when allow_auto is False, require approval
        risk_label = policy.risk.value
        logger.debug(
            "Requiring approval for tool=%s (risk=%s, allow_auto=%s)",
            tool_call.tool,
            risk_label,
            policy.allow_auto,
        )
        return PolicyDecision(
            allowed=True,
            reason=f"{risk_label.capitalize()}-risk action requires user approval",
            requires_approval=True,
            denied=False,
        )

    def _matches_list(self, value: str, patterns: list[str]) -> bool:
        """Check if a value matches any pattern in the list.

        Currently performs case-insensitive exact matching.
        Future enhancement: support glob patterns or regex.

        Args:
            value: The value to check.
            patterns: List of patterns to match against.

        Returns:
            True if the value matches any pattern.
        """
        value_lower = value.lower()
        return any(pattern.lower() == value_lower for pattern in patterns)

    def add_policy(self, policy: ActionPolicy) -> None:
        """Add or update a policy for a tool.

        Args:
            policy: The policy to add or update.
        """
        self._policies[policy.tool_name] = policy
        logger.info("Added/updated policy for tool=%s", policy.tool_name)

    def remove_policy(self, tool_name: str) -> bool:
        """Remove a policy for a tool.

        Args:
            tool_name: Name of the tool to remove the policy for.

        Returns:
            True if a policy was removed, False if no policy existed.
        """
        if tool_name in self._policies:
            del self._policies[tool_name]
            logger.info("Removed policy for tool=%s", tool_name)
            return True
        return False


# Module-level singleton for convenience
_default_engine: PolicyEngine | None = None


def get_policy_engine() -> PolicyEngine:
    """Get the default PolicyEngine singleton.

    Returns:
        The shared PolicyEngine instance.
    """
    global _default_engine
    if _default_engine is None:
        _default_engine = PolicyEngine()
    return _default_engine


def reset_policy_engine() -> None:
    """Reset the default PolicyEngine singleton.

    Useful for testing or when policies need to be reloaded.
    """
    global _default_engine
    _default_engine = None


__all__ = [
    "PolicyEngine",
    "DEFAULT_POLICIES",
    "get_policy_engine",
    "reset_policy_engine",
]
