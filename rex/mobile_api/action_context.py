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


class MobileStrongAuthRequiredError(MobileActionDeniedError):
    """A privileged mobile action lacks a valid one-time S8 approval."""

    def __init__(
        self,
        message: str,
        *,
        challenge: Any | None = None,
        action_name: str | None = None,
        action: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.challenge = challenge
        self.action_name = action_name
        self.action = action


@dataclass(frozen=True)
class MobileActionContext:
    scopes: frozenset[str]
    permissions: frozenset[str]
    revalidate: Callable[[], None] | None = None
    strong_auth_authority: Any | None = None
    strong_auth_principal: Any | None = None
    strong_auth_approval_id: str | None = None


_CONTEXT: contextvars.ContextVar[MobileActionContext | None] = contextvars.ContextVar(
    "askrex_mobile_action_context", default=None
)
_AUTHORIZED_ACTIONS: contextvars.ContextVar[tuple[tuple[str, str], ...]] = contextvars.ContextVar(
    "askrex_mobile_authorized_actions", default=()
)


@contextmanager
def mobile_action_context(
    scopes: frozenset[str] | set[str] | tuple[str, ...] | list[str],
    *,
    permissions: frozenset[str] | set[str] | tuple[str, ...] | list[str] = frozenset(),
    revalidate: Callable[[], None] | None = None,
    strong_auth_authority: Any | None = None,
    strong_auth_principal: Any | None = None,
    strong_auth_approval_id: str | None = None,
) -> Iterator[None]:
    token = _CONTEXT.set(
        MobileActionContext(
            scopes=frozenset(scopes),
            permissions=frozenset(permissions),
            revalidate=revalidate,
            strong_auth_authority=strong_auth_authority,
            strong_auth_principal=strong_auth_principal,
            strong_auth_approval_id=strong_auth_approval_id,
        )
    )
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


def _required_error(
    context: MobileActionContext,
    *,
    action_name: str,
    action_arguments: dict[str, Any],
    message: str,
) -> MobileStrongAuthRequiredError:
    challenge = None
    authority = context.strong_auth_authority
    principal = context.strong_auth_principal
    if authority is not None and principal is not None:
        from rex.mobile_api.strong_auth import StrongAuthError  # noqa: PLC0415

        try:
            challenge = authority.create_challenge(
                principal,
                action_name=action_name,
                payload=action_arguments,
            )
        except StrongAuthError as exc:
            if exc.reason in {
                "paired_session_required",
                "scope_denied",
                "binding_revoked",
                "binding_changed",
                "binding_expired",
                "binding_invalid",
            }:
                raise MobileActionDeniedError(str(exc)) from exc
    public_action = {
        key: value
        for key, value in action_arguments.items()
        if key not in {"_user_id", "_request_id", "context", "strong_auth_approval_id"}
    }
    return MobileStrongAuthRequiredError(
        message,
        challenge=challenge,
        action_name=action_name,
        action=public_action,
    )


@contextmanager
def authorized_mobile_tool(
    tool_name: str,
    *,
    capability_tags: tuple[str, ...] | list[str] | None = None,
    operation: str | None = None,
    arguments: dict[str, Any] | None = None,
) -> Iterator[None]:
    authorize_mobile_action(
        required_scope_for_tool(
            tool_name,
            capability_tags=capability_tags,
            operation=operation,
        ),
        f"tool:{tool_name}",
    )
    context = _CONTEXT.get()
    if context is None:
        yield
        return

    from rex.mobile_api.strong_auth import (  # noqa: PLC0415
        StrongAuthError,
        canonical_action,
        policy_for_action,
    )

    normalized_name = tool_name.strip().lower()
    action_arguments = dict(arguments or {})
    policy = policy_for_action(normalized_name, action_arguments)
    if normalized_name == "home_assistant_call_service" and policy is None:
        raise MobileActionDeniedError(
            "This Home Assistant action is not allowed from a mobile device."
        )
    if policy is None or not policy.requires_strong_auth:
        yield
        return

    _, _, action_hash = canonical_action(normalized_name, action_arguments)
    binding = (normalized_name, action_hash)
    authorized_actions = _AUTHORIZED_ACTIONS.get()
    if authorized_actions and authorized_actions[-1] == binding:
        yield
        return

    if (
        context.strong_auth_authority is None
        or context.strong_auth_principal is None
        or context.strong_auth_approval_id is None
    ):
        raise _required_error(
            context,
            action_name=normalized_name,
            action_arguments=action_arguments,
            message="Strong authentication is required for this mobile action.",
        )
    try:
        context.strong_auth_authority.consume_approval(
            context.strong_auth_principal,
            approval_id=context.strong_auth_approval_id,
            action_name=normalized_name,
            payload=action_arguments,
        )
    except StrongAuthError as exc:
        raise _required_error(
            context,
            action_name=normalized_name,
            action_arguments=action_arguments,
            message="The strong-authentication approval is invalid, expired, or already used.",
        ) from exc

    token = _AUTHORIZED_ACTIONS.set((*authorized_actions, binding))
    try:
        yield
    finally:
        _AUTHORIZED_ACTIONS.reset(token)


def authorize_mobile_tool(
    tool_name: str,
    *,
    capability_tags: tuple[str, ...] | list[str] | None = None,
    operation: str | None = None,
    arguments: dict[str, Any] | None = None,
) -> None:
    """Compatibility check; execution paths should use authorized_mobile_tool."""
    with authorized_mobile_tool(
        tool_name,
        capability_tags=capability_tags,
        operation=operation,
        arguments=arguments,
    ):
        return


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
    "MobileStrongAuthRequiredError",
    "authorized_mobile_tool",
    "authorize_mobile_action",
    "authorize_mobile_tool",
    "current_mobile_action_context",
    "mobile_action_context",
    "mobile_action_context_active",
    "mobile_scope_granted",
    "required_scope_for_tool",
    "run_in_executor_with_mobile_context",
]
