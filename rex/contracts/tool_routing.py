"""Protocol defining the tool routing interface for Rex.

This contract captures the public API of ``rex.tool_router`` so that an
OpenClaw-backed adapter can be substituted transparently.  Any class that
implements these methods satisfies the protocol (structural subtyping via
``typing.Protocol``).
"""

from __future__ import annotations

from typing import Any, Callable, Protocol, runtime_checkable


@runtime_checkable
class ToolRoutingProtocol(Protocol):
    """Structural protocol for Rex tool routing.

    Implementors must provide the three core operations:

    - ``parse_tool_request`` — detect and decode a tool-call line from LLM output
    - ``execute_tool`` — run a decoded tool request through policy and credential checks
    - ``route_if_tool_request`` — end-to-end: detect, execute, re-call model, return text

    The ``format_tool_result`` helper is intentionally excluded because it is a
    presentation concern that does not need to vary across adapters.
    """

    def parse_tool_request(self, text: str) -> dict[str, Any] | None:
        """Return a parsed tool-request dict or *None* if ``text`` is not a tool call.

        Args:
            text: A single line of LLM output to inspect.

        Returns:
            A dict with keys ``"tool"`` (str) and ``"args"`` (dict), or
            ``None`` when ``text`` does not contain a valid tool request.
        """
        ...

    def execute_tool(
        self,
        request: dict[str, Any],
        default_context: dict[str, Any],
        *,
        skip_policy_check: bool = False,
        skip_credential_check: bool = False,
        task_id: str | None = None,
        requested_by: str | None = None,
        skip_audit_log: bool = False,
    ) -> dict[str, Any]:
        """Execute a decoded tool request and return a result dictionary.

        Args:
            request: Dict with ``"tool"`` and ``"args"`` keys as returned by
                :meth:`parse_tool_request`.
            default_context: Ambient context (timezone, location, user, …).
            skip_policy_check: When *True*, bypass policy gating.
            skip_credential_check: When *True*, bypass credential validation.
            task_id: Optional correlation ID for audit logging.
            requested_by: Optional identifier of the requesting entity.
            skip_audit_log: When *True*, do not write an audit log entry.

        Returns:
            A dict containing at minimum a ``"status"`` key (``"ok"`` or
            ``"error"``) and a ``"result"`` key with the tool output.
        """
        ...

    def route_if_tool_request(
        self,
        llm_text: str,
        default_context: dict[str, Any],
        model_call_fn: Callable[[dict[str, str]], str],
        *,
        skip_policy_check: bool = False,
    ) -> str:
        """Detect a tool call in *llm_text*, execute it, and return the final reply.

        If *llm_text* does not contain a tool request it is returned unchanged.

        Args:
            llm_text: Raw LLM output that may contain a tool-call line.
            default_context: Ambient context forwarded to :meth:`execute_tool`.
            model_call_fn: Callable that sends a follow-up message to the model
                and returns the model's response string.
            skip_policy_check: When *True*, bypass policy gating.

        Returns:
            The final text response (either *llm_text* verbatim, or the model's
            response after the tool result was injected).
        """
        ...
