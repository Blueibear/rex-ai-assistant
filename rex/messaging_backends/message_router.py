"""Multi-channel message router.

Routes outbound messages to the correct delivery backend (dashboard, email, or
SMS) based on a channel identifier.  The active channel and backend instances
are configurable without code changes via :class:`RouterConfig`.

Usage example::

    from rex.messaging_backends.message_router import MessageRouter, MessagePayload, RouterConfig
    from rex.messaging_backends.sms_sender_stub import SmsSenderStub
    from rex.email_backends.stub import StubEmailBackend
    from rex.dashboard.sse import get_broadcaster

    router = MessageRouter(
        config=RouterConfig(active_channel="sms"),
        sms_backend=SmsSenderStub(),
        email_backend=StubEmailBackend(),
        dashboard_broadcaster=get_broadcaster(),
    )
    result = router.route(MessagePayload(body="Hello!"), channel="sms")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Known channel identifiers
# ---------------------------------------------------------------------------

CHANNEL_DASHBOARD = "dashboard"
CHANNEL_EMAIL = "email"
CHANNEL_SMS = "sms"

KNOWN_CHANNELS: frozenset[str] = frozenset({CHANNEL_DASHBOARD, CHANNEL_EMAIL, CHANNEL_SMS})


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class UnknownChannelError(ValueError):
    """Raised when an unsupported or unconfigured channel is requested."""

    def __init__(self, channel: str) -> None:
        self.channel = channel
        super().__init__(
            f"Unknown or unconfigured channel: {channel!r}. "
            f"Supported channels: {sorted(KNOWN_CHANNELS)}"
        )


class ChannelNotConfiguredError(RuntimeError):
    """Raised when the requested channel has no registered backend."""

    def __init__(self, channel: str) -> None:
        self.channel = channel
        super().__init__(
            f"Channel {channel!r} is known but has no backend configured in this router instance."
        )


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class MessagePayload:
    """Portable message payload accepted by all channels.

    Attributes:
        body: Main message text (required).
        subject: Subject or title — used by email and dashboard.
        to: Recipient address or phone number — used by email and SMS.
        metadata: Extra key-value pairs forwarded to the backend.
    """

    body: str
    subject: str = ""
    to: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RouteResult:
    """Outcome of a :meth:`MessageRouter.route` call.

    Attributes:
        ok: ``True`` when the message was accepted by the backend.
        channel: Channel that was used.
        detail: Backend-specific response or description.
        error: Human-readable error string when ``ok`` is ``False``.
    """

    ok: bool
    channel: str
    detail: Any = None
    error: str | None = None


@dataclass
class RouterConfig:
    """Runtime configuration for :class:`MessageRouter`.

    Attributes:
        active_channel: Default channel used by :meth:`MessageRouter.send`.
            Must be one of ``"dashboard"``, ``"email"``, or ``"sms"``.
    """

    active_channel: str = CHANNEL_DASHBOARD


# ---------------------------------------------------------------------------
# Protocol-style type aliases (avoid hard imports of optional backends)
# ---------------------------------------------------------------------------

# We import these lazily via TYPE_CHECKING only so that missing optional
# dependencies do not break the import of this module.
if TYPE_CHECKING:
    from rex.dashboard.sse import NotificationBroadcaster
    from rex.email_backends.base import EmailBackend
    from rex.messaging_backends.twilio_adapter import TwilioAdapter


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


class MessageRouter:
    """Routes messages to dashboard, email, or SMS backends.

    Backends are injected at construction time; any can be ``None`` if that
    channel is not in use.  Requesting a ``None`` backend raises
    :class:`ChannelNotConfiguredError`.

    Args:
        config: Runtime configuration (active channel, etc.).
        sms_backend: An object implementing :class:`TwilioAdapter` (i.e. has
            ``send_sms(to, body)``).
        email_backend: An :class:`~rex.email_backends.base.EmailBackend` instance.
        dashboard_broadcaster: A :class:`~rex.dashboard.sse.NotificationBroadcaster`.
    """

    def __init__(
        self,
        config: RouterConfig | None = None,
        *,
        sms_backend: TwilioAdapter | None = None,
        email_backend: EmailBackend | None = None,
        dashboard_broadcaster: NotificationBroadcaster | None = None,
    ) -> None:
        self._config = config or RouterConfig()
        self._sms = sms_backend
        self._email = email_backend
        self._dashboard = dashboard_broadcaster

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def active_channel(self) -> str:
        """The currently configured default delivery channel."""
        return self._config.active_channel

    @active_channel.setter
    def active_channel(self, value: str) -> None:
        """Update the active channel at runtime without code changes."""
        if value not in KNOWN_CHANNELS:
            raise UnknownChannelError(value)
        self._config.active_channel = value

    def configure(self, config: RouterConfig) -> None:
        """Replace the router configuration at runtime."""
        self._config = config

    def route(self, payload: MessagePayload, channel: str) -> RouteResult:
        """Deliver *payload* to the specified *channel*.

        Args:
            payload: The message to deliver.
            channel: One of ``"dashboard"``, ``"email"``, or ``"sms"``.

        Returns:
            A :class:`RouteResult` describing the outcome.

        Raises:
            :class:`UnknownChannelError`: When *channel* is not a known channel
                identifier.  This is a handled error — callers should catch it
                rather than letting it propagate to the user.
            :class:`ChannelNotConfiguredError`: When *channel* is known but the
                corresponding backend was not provided to this router instance.
        """
        if channel not in KNOWN_CHANNELS:
            err = UnknownChannelError(channel)
            logger.warning("[MessageRouter] %s", err)
            raise err

        try:
            if channel == CHANNEL_SMS:
                return self._route_sms(payload)
            if channel == CHANNEL_EMAIL:
                return self._route_email(payload)
            if channel == CHANNEL_DASHBOARD:
                return self._route_dashboard(payload)
        except (UnknownChannelError, ChannelNotConfiguredError):
            raise
        except Exception as exc:  # noqa: BLE001
            logger.error("[MessageRouter] Delivery to %r failed: %s", channel, exc)
            return RouteResult(ok=False, channel=channel, error=str(exc))

        # Should never reach here given KNOWN_CHANNELS guard above.
        raise UnknownChannelError(channel)  # pragma: no cover

    def send(self, payload: MessagePayload) -> RouteResult:
        """Deliver *payload* to the active channel (from :attr:`active_channel`).

        This is a convenience wrapper around :meth:`route` that uses the
        configured default channel so callers don't need to repeat it.
        """
        return self.route(payload, self._config.active_channel)

    # ------------------------------------------------------------------
    # Channel-specific routing helpers
    # ------------------------------------------------------------------

    def _route_sms(self, payload: MessagePayload) -> RouteResult:
        if self._sms is None:
            raise ChannelNotConfiguredError(CHANNEL_SMS)
        result = self._sms.send_sms(to=payload.to, body=payload.body)
        ok: bool = bool(result.get("ok", False))
        logger.info("[MessageRouter] SMS to %s: ok=%s", payload.to, ok)
        return RouteResult(ok=ok, channel=CHANNEL_SMS, detail=result)

    def _route_email(self, payload: MessagePayload) -> RouteResult:
        if self._email is None:
            raise ChannelNotConfiguredError(CHANNEL_EMAIL)
        from_addr = payload.metadata.get("from_addr", "rex@localhost")
        to_addrs: list[str] = payload.metadata.get("to_addrs") or (
            [payload.to] if payload.to else []
        )
        result = self._email.send(
            from_addr=from_addr,
            to_addrs=to_addrs,
            subject=payload.subject or "(no subject)",
            body=payload.body,
        )
        logger.info("[MessageRouter] Email to %s: ok=%s", to_addrs, result.ok)
        return RouteResult(ok=result.ok, channel=CHANNEL_EMAIL, detail=result)

    def _route_dashboard(self, payload: MessagePayload) -> RouteResult:
        if self._dashboard is None:
            raise ChannelNotConfiguredError(CHANNEL_DASHBOARD)
        event: dict[str, Any] = {
            "type": "message",
            "body": payload.body,
            "subject": payload.subject,
            **payload.metadata,
        }
        self._dashboard.publish(event)
        logger.info("[MessageRouter] Published to dashboard: %s", payload.body[:80])
        return RouteResult(ok=True, channel=CHANNEL_DASHBOARD, detail=event)


__all__ = [
    "CHANNEL_DASHBOARD",
    "CHANNEL_EMAIL",
    "CHANNEL_SMS",
    "KNOWN_CHANNELS",
    "ChannelNotConfiguredError",
    "MessagePayload",
    "MessageRouter",
    "RouteResult",
    "RouterConfig",
    "UnknownChannelError",
]
