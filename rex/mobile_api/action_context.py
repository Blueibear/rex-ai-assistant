"""Per-request mobile capability context for action/tool dispatch (S6).

Desktop, CLI, and local voice calls run without this context and retain their
existing policy model.  Mobile chat/voice requests install a server-derived
scope set plus a live session revalidation callback.  Every lower action
layer can then fail closed without trusting prompt text, JWT claims, or tool
arguments.
"""

from __future__ import annotations

import asyncio
import contextvars
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any


class MobileActionDeniedError(PermissionError):
    """A mobile device grant does not authorize an action."""


@dataclass(frozen=True)
class MobileActionContext:
    scopes: frozenset[str]
    permissions: frozenset[str]
    revalidate: Callable[[], None] | None = None


_CONTEXT: contextvars.ContextVar[MobileActionContext | None] = contextvars.ContextVar(
    "askrex_mobile_action_context", default=None
)


@contextmanager
def mobile_action_context(
    scopes: frozenset[str] | set[str] | tuple[str, ...] | list[str],
    *,
    permissions: frozenset[str] | set[str] | tuple[str, ...] | list[str] = frozenset(),
    revalidate: Callable[[], None] | None = None,
) -> Iterator[None]:
    token = _CONTEXT.set(MobileActionContext(frozenset(scopes), frozenset(permissions), revalidate))
    try:
        yield
    finally:
        _CONTEXT.reset(token)


def current_mobile_action_context() -> MobileActionContext | None:
    return _CONTEXT.get()


def mobile_action_context_active() -> bool:
    return _CONTEXT.get() is not None


def mobile_scope_granted(scope: str) -> bool:
    """Return whether the current mobile grant has *scope*.

    Non-mobile callers return True.  A live revalidation callback still runs
    before the answer so revocation cannot be hidden by a stale context.
    """
    context = _CONTEXT.get()
    if context is None:
        return True
    if context.revalidate is not None:
        context.revalidate()
    from rex.mobile_api.authorization import require_scope  # noqa: PLC0415

    try:
        require_scope(context.scopes, scope, permissions=context.permissions)
    except ValueError:
        return False
    return True


def authorize_mobile_action(required_scope: str | None, action_name: str) -> None:
    """Authorize one lower-layer action when called from a mobile request.

    ``None`` means the action is intentionally unmapped and therefore denied
    for mobile callers.  Non-mobile callers have no context and are unchanged.
    """
    context = _CONTEXT.get()
    if context is None:
        return
    if context.revalidate is not None:
        context.revalidate()
    if required_scope is None:
        raise MobileActionDeniedError(
            f"The paired device is not authorized for action {action_name!r}."
        )
    from rex.mobile_api.authorization import require_scope  # noqa: PLC0415

    try:
        require_scope(
            context.scopes,
            required_scope,
            permissions=context.permissions,
        )
    except ValueError as exc:
        raise MobileActionDeniedError(
            f"The user and paired device are not authorized for action {action_name!r}."
        ) from exc


def required_scope_for_tool(
    tool_name: str,
    *,
    capability_tags: tuple[str, ...] | list[str] | None = None,
    operation: str | None = None,
) -> str | None:
    """Map a known canonical tool to a mobile grant scope.

    Authorization is intentionally name-based and fail-closed.  Capability
    tags are descriptive metadata, not an authority source: a future plugin
    cannot gain mobile access merely by choosing a trusted-looking tag or
    name fragment.  New tools require an explicit entry and tests here.
    """
    del capability_tags  # Metadata must never widen authority.
    name = tool_name.strip().lower()
    normalized_operation = operation.strip().lower() if isinstance(operation, str) else None

    read_only_chat = {"time_now", "weather_now", "web_search"}
    if name in read_only_chat:
        return "chat.send" if normalized_operation in {None, "read", "query"} else None

    home_mutations = {
        "home_assistant_call_service",
        "music_play",
        "music_pause",
        "music_resume",
        "music_skip",
        "music_volume",
    }
    if name in home_mutations:
        return "home.control" if normalized_operation in {None, "mutation", "write"} else None

    # Email, calendar, SMS, shell, filesystem, diagnostics, dynamic OpenClaw
    # tools, and every future tool are unavailable to mobile until a deliberate
    # scope mapping and enforcement test are added.
    return None


def authorize_mobile_tool(
    tool_name: str,
    *,
    capability_tags: tuple[str, ...] | list[str] | None = None,
    operation: str | None = None,
) -> None:
    authorize_mobile_action(
        required_scope_for_tool(
            tool_name,
            capability_tags=capability_tags,
            operation=operation,
        ),
        f"tool:{tool_name}",
    )


async def run_in_executor_with_mobile_context(
    loop: asyncio.AbstractEventLoop,
    func: Callable[..., Any],
    *args: Any,
) -> Any:
    """Run a blocking call while preserving the current mobile context."""
    copied = contextvars.copy_context()
    return await loop.run_in_executor(None, copied.run, func, *args)


__all__ = [
    "MobileActionContext",
    "MobileActionDeniedError",
    "authorize_mobile_action",
    "authorize_mobile_tool",
    "current_mobile_action_context",
    "mobile_action_context",
    "mobile_action_context_active",
    "mobile_scope_granted",
    "required_scope_for_tool",
    "run_in_executor_with_mobile_context",
]
