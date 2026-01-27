"""Tool routing for single line tool requests.

All tool calls are evaluated by the policy engine before execution to determine
whether they should auto-execute, require approval, or be denied.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable
from zoneinfo import ZoneInfo

from rex.contracts import ToolCall
from rex.policy_engine import PolicyEngine, get_policy_engine

logger = logging.getLogger(__name__)

TOOL_REQUEST_PREFIX = "TOOL_REQUEST:"
TOOL_RESULT_PREFIX = "TOOL_RESULT:"


@dataclass(frozen=True)
class ToolError:
    message: str


class PolicyDeniedError(Exception):
    """Raised when a tool call is denied by the policy engine."""

    def __init__(self, tool: str, reason: str) -> None:
        self.tool = tool
        self.reason = reason
        super().__init__(f"Tool '{tool}' denied by policy: {reason}")


class ApprovalRequiredError(Exception):
    """Raised when a tool call requires user approval.

    This exception is raised when the policy engine determines that a tool
    call requires approval before execution. The actual approval flow should
    be handled by the caller.
    """

    def __init__(self, tool: str, reason: str) -> None:
        self.tool = tool
        self.reason = reason
        super().__init__(f"Tool '{tool}' requires approval: {reason}")


def parse_tool_request(text: str) -> dict[str, Any] | None:
    """Return parsed tool request data or None if not a valid request."""
    if not isinstance(text, str):
        return None
    if "\n" in text or "\r" in text:
        return None
    line = text.strip()
    if not line.startswith(TOOL_REQUEST_PREFIX):
        return None

    json_payload = line[len(TOOL_REQUEST_PREFIX):].strip()
    if not json_payload:
        return None

    try:
        data = json.loads(json_payload)
    except json.JSONDecodeError:
        return None

    if not isinstance(data, dict):
        return None

    tool = data.get("tool")
    args = data.get("args")

    if not isinstance(tool, str) or not tool:
        return None

    if args is None:
        args = {}
    if not isinstance(args, dict):
        return None

    return {"tool": tool, "args": args}


def execute_tool(
    request: dict[str, Any],
    default_context: dict[str, Any],
    *,
    policy_engine: PolicyEngine | None = None,
    skip_policy_check: bool = False,
) -> dict[str, Any]:
    """Execute a tool request and return a result dictionary.

    Before executing a tool, the policy engine is consulted to determine
    whether the action should proceed. If the policy denies the action,
    a PolicyDeniedError is raised. If approval is required, an
    ApprovalRequiredError is raised.

    Args:
        request: Tool request dict with 'tool' and 'args' keys.
        default_context: Default context for tool execution.
        policy_engine: Optional policy engine instance. If not provided,
            uses the default singleton.
        skip_policy_check: If True, skip the policy check. Use with caution.

    Returns:
        A result dictionary with the tool output or error.

    Raises:
        PolicyDeniedError: If the policy engine denies the action.
        ApprovalRequiredError: If the action requires user approval.
    """
    if not isinstance(request, dict):
        return _error_result("Invalid tool request payload")

    tool = request.get("tool")
    args = request.get("args")
    if not isinstance(tool, str) or not tool:
        return _error_result("Invalid tool name")
    if args is None:
        args = {}
    if not isinstance(args, dict):
        return _error_result("Invalid tool arguments", tool=tool, args={})

    # Check policy before execution
    if not skip_policy_check:
        engine = policy_engine or get_policy_engine()
        tool_call = ToolCall(tool=tool, args=args)
        metadata = _extract_policy_metadata(args)
        decision = engine.decide(tool_call, metadata)

        if decision.denied:
            logger.warning("Policy denied tool=%s: %s", tool, decision.reason)
            raise PolicyDeniedError(tool, decision.reason)

        if decision.requires_approval:
            logger.info("Policy requires approval for tool=%s: %s", tool, decision.reason)
            raise ApprovalRequiredError(tool, decision.reason)

        logger.debug("Policy allowed tool=%s: %s", tool, decision.reason)

    if tool == "time_now":
        return _execute_time_now(args, default_context)

    if tool in {"weather_now", "web_search"}:
        return _error_result(f"Tool {tool} is not implemented", tool=tool, args=args)

    return _error_result(f"Unknown tool {tool}", tool=tool, args=args)


def _extract_policy_metadata(args: dict[str, Any]) -> dict[str, Any]:
    """Extract metadata from tool arguments for policy evaluation.

    Looks for common argument patterns that indicate recipients or domains:
    - to, recipient, email: Email address
    - url, domain: Domain name
    - target: Generic target identifier

    Args:
        args: Tool arguments dictionary.

    Returns:
        Metadata dictionary for policy evaluation.
    """
    metadata: dict[str, Any] = {}

    # Check for recipient-like arguments
    for key in ("to", "recipient", "email", "address"):
        if key in args and isinstance(args[key], str):
            metadata["recipient"] = args[key]
            break

    # Check for domain-like arguments
    for key in ("domain", "host"):
        if key in args and isinstance(args[key], str):
            metadata["domain"] = args[key]
            break

    # Extract domain from URL if present
    if "url" in args and isinstance(args["url"], str):
        url = args["url"]
        # Simple domain extraction from URL
        if "://" in url:
            domain_part = url.split("://", 1)[1].split("/", 1)[0]
            # Remove port if present
            domain_part = domain_part.split(":")[0]
            metadata["domain"] = domain_part

    # Check for generic target
    if "target" in args and isinstance(args["target"], str):
        metadata["target"] = args["target"]

    return metadata


def format_tool_result(tool: str, args: dict[str, Any], result: dict[str, Any]) -> str:
    """Format the tool result as a single line message."""
    payload = {
        "tool": tool,
        "args": args,
        "result": result,
    }
    json_payload = json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    return f"{TOOL_RESULT_PREFIX} {json_payload}"


def route_if_tool_request(
    llm_text: str,
    default_context: dict[str, Any],
    model_call_fn: Callable[[dict[str, str]], str],
    *,
    policy_engine: PolicyEngine | None = None,
    skip_policy_check: bool = False,
) -> str:
    """Route a tool request if present and return the final model output.

    Args:
        llm_text: The LLM output text that may contain a tool request.
        default_context: Default context for tool execution.
        model_call_fn: Function to call the model with tool results.
        policy_engine: Optional policy engine instance for policy checks.
        skip_policy_check: If True, skip policy checks on tool execution.

    Returns:
        The final model output or an error message.
    """
    request = parse_tool_request(llm_text)
    if request is None:
        return llm_text

    tool = request.get("tool", "unknown")
    args = request.get("args", {})

    try:
        result = execute_tool(
            request,
            default_context,
            policy_engine=policy_engine,
            skip_policy_check=skip_policy_check,
        )
    except PolicyDeniedError as e:
        logger.warning("Tool request denied: %s", e)
        return f"I cannot execute that action: {e.reason}"
    except ApprovalRequiredError as e:
        logger.info("Tool request requires approval: %s", e)
        return f"This action requires your approval: {e.reason}"

    tool_result_line = format_tool_result(tool, args, result)
    tool_message = {"role": "tool", "content": tool_result_line}

    try:
        return model_call_fn(tool_message)
    except Exception:
        return "Sorry, I could not complete that tool request."


def _execute_time_now(args: dict[str, Any], default_context: dict[str, Any]) -> dict[str, Any]:
    location = args.get("location")
    if not isinstance(location, str) or not location.strip():
        location = default_context.get("location")

    if not isinstance(location, str) or not location.strip():
        return _error_result("Missing required location for time_now", tool="time_now", args=args)

    location = location.strip()
    timezone = _resolve_timezone(location, default_context)
    if isinstance(timezone, ToolError):
        return _error_result(timezone.message, tool="time_now", args=args)

    try:
        zone = ZoneInfo(timezone)
    except Exception:
        return _error_result("Invalid timezone for time_now", tool="time_now", args=args)

    now = datetime.now(tz=zone)
    return {
        "local_time": now.strftime("%Y-%m-%d %H:%M"),
        "timezone": timezone,
    }


def _resolve_timezone(location: str, default_context: dict[str, Any]) -> str | ToolError:
    normalized = location.strip().lower()
    if normalized == "dallas, tx":
        return "America/Chicago"

    fallback = default_context.get("timezone")
    if isinstance(fallback, str) and fallback.strip():
        return fallback.strip()

    return "UTC"


def _error_result(
    message: str,
    tool: str | None = None,
    args: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {"message": message}
    if tool:
        payload["tool"] = tool
    if args is not None:
        payload["args"] = args
    return {"error": payload}


__all__ = [
    "parse_tool_request",
    "execute_tool",
    "format_tool_result",
    "route_if_tool_request",
    "PolicyDeniedError",
    "ApprovalRequiredError",
]
