"""Tool routing for single line tool requests."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, Optional
from zoneinfo import ZoneInfo

TOOL_REQUEST_PREFIX = "TOOL_REQUEST:"
TOOL_RESULT_PREFIX = "TOOL_RESULT:"


@dataclass(frozen=True)
class ToolError:
    message: str


def parse_tool_request(text: str) -> Optional[Dict[str, Any]]:
    """Return parsed tool request data or None if not a valid request."""
    if not isinstance(text, str):
        return None
    line = text.strip()
    if not line.startswith(TOOL_REQUEST_PREFIX):
        return None
    if len(line.splitlines()) != 1:
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


def execute_tool(request: Dict[str, Any], default_context: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a tool request and return a result dictionary."""
    if not isinstance(request, dict):
        return _error_result("Invalid tool request payload")

    tool = request.get("tool")
    args = request.get("args")
    if not isinstance(tool, str) or not tool:
        return _error_result("Invalid tool name")
    if args is None:
        args = {}
    if not isinstance(args, dict):
        return _error_result("Invalid tool arguments", tool=tool)

    if tool == "time_now":
        return _execute_time_now(args, default_context)

    if tool in {"weather_now", "web_search"}:
        return _error_result(f"Tool {tool} is not implemented", tool=tool, args=args)

    return _error_result(f"Unknown tool {tool}", tool=tool, args=args)


def format_tool_result(tool: str, args: Dict[str, Any], result: Dict[str, Any]) -> str:
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
    default_context: Dict[str, Any],
    model_call_fn: Callable[[Dict[str, str]], str],
) -> str:
    """Route a tool request if present and return the final model output."""
    request = parse_tool_request(llm_text)
    if request is None:
        return llm_text

    tool = request.get("tool", "unknown")
    args = request.get("args", {})
    result = execute_tool(request, default_context)
    tool_result_line = format_tool_result(tool, args, result)
    tool_message = {"role": "tool", "content": tool_result_line}

    try:
        return model_call_fn(tool_message)
    except Exception:
        return "Sorry, I could not complete that tool request."


def _execute_time_now(args: Dict[str, Any], default_context: Dict[str, Any]) -> Dict[str, Any]:
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


def _resolve_timezone(location: str, default_context: Dict[str, Any]) -> str | ToolError:
    normalized = location.strip().lower()
    if normalized == "dallas, tx":
        return "America/Chicago"

    fallback = default_context.get("timezone")
    if isinstance(fallback, str) and fallback.strip():
        return fallback.strip()

    return "UTC"


def _error_result(message: str, tool: Optional[str] = None, args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
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
]
