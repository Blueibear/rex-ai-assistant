"""Protocol definitions for the Rex dashboard subsystem.

This contract captures the public APIs of ``rex.dashboard`` so that an
OpenClaw-backed adapter can be substituted transparently.

Three protocols are defined:

- ``SessionManagerProtocol`` — token-based session auth (create, validate, invalidate)
- ``NotificationBroadcasterProtocol`` — SSE event broadcasting (subscribe, publish, stream)
- ``DashboardProtocol`` — the top-level service that wires auth, SSE, and route registration
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Optional, Protocol, runtime_checkable


# ---------------------------------------------------------------------------
# Session authentication
# ---------------------------------------------------------------------------


@runtime_checkable
class SessionManagerProtocol(Protocol):
    """Structural protocol for dashboard session management.

    Covers the public API of ``rex.dashboard.auth.SessionManager``:

    - ``create_session`` — issue a new authenticated session token
    - ``validate_session`` — verify a token and return the live session
    - ``invalidate_session`` — revoke a token (logout)
    - ``get_active_session_count`` — introspection helper

    Implementations must be thread-safe.
    """

    def create_session(
        self,
        user_key: str = "dashboard",
        metadata: Optional[dict] = None,
    ) -> Any:
        """Issue a new session and return the session object.

        Args:
            user_key: Logical identifier for the authenticated user.
            metadata: Optional arbitrary key/value data attached to the session.

        Returns:
            A session object with at least ``token`` (str) and ``expires_at``
            attributes, and an ``is_expired()`` method.
        """
        ...

    def validate_session(self, token: str) -> Optional[Any]:
        """Validate *token* and return the associated session, or ``None``.

        Args:
            token: The raw session token as issued by :meth:`create_session`.

        Returns:
            The live session object when valid and unexpired, otherwise ``None``.
        """
        ...

    def invalidate_session(self, token: str) -> bool:
        """Revoke *token* (logout).

        Args:
            token: The session token to revoke.

        Returns:
            ``True`` if the token existed and was removed, ``False`` otherwise.
        """
        ...

    def get_active_session_count(self) -> int:
        """Return the number of currently active (non-expired) sessions."""
        ...


# ---------------------------------------------------------------------------
# SSE notification broadcasting
# ---------------------------------------------------------------------------


@runtime_checkable
class NotificationBroadcasterProtocol(Protocol):
    """Structural protocol for dashboard SSE notification broadcasting.

    Covers the public API of ``rex.dashboard.sse.NotificationBroadcaster``:

    - ``subscribe`` — register a new SSE subscriber
    - ``unsubscribe`` — deregister a subscriber
    - ``publish`` — push an event to all active subscribers
    - ``stream`` — yield SSE-formatted chunks for one subscriber
    - ``shutdown`` — close all subscribers (teardown)
    - ``subscriber_count`` — live count of active subscribers

    Implementations must be thread-safe and non-blocking in ``publish``.
    """

    def subscribe(self, *, max_events: int = 100) -> Any:
        """Register a new subscriber with a bounded event queue.

        Args:
            max_events: Maximum number of queued events before the oldest is
                dropped.

        Returns:
            An opaque subscriber handle to pass to :meth:`stream` and
            :meth:`unsubscribe`.
        """
        ...

    def unsubscribe(self, subscriber: Any) -> None:
        """Deregister *subscriber* and mark it as closed.

        Args:
            subscriber: The handle returned by :meth:`subscribe`.
        """
        ...

    def publish(self, event: Any) -> None:
        """Broadcast *event* to all active subscribers without blocking.

        Args:
            event: Either a ``NotificationEvent`` dataclass instance or a
                JSON-serialisable ``dict``.  Closed or full subscriber queues
                handle overflow (oldest event dropped or subscriber removed).
        """
        ...

    def stream(
        self,
        subscriber: Any,
        *,
        timeout: float = 15.0,
        keepalive_interval: float = 15.0,
    ) -> Iterator[str]:
        """Yield SSE-formatted strings for *subscriber* until it is closed.

        Yields keep-alive comments (``": keep-alive\\n\\n"``) when no event
        arrives within *keepalive_interval* seconds.

        Args:
            subscriber: The handle returned by :meth:`subscribe`.
            timeout: Per-poll wait time in seconds.
            keepalive_interval: Interval between keep-alive yields in seconds.

        Yields:
            SSE-formatted strings (``"event: ...\\ndata: ...\\n\\n"``).
        """
        ...

    def shutdown(self) -> None:
        """Close and remove all active subscribers (for clean teardown)."""
        ...

    @property
    def subscriber_count(self) -> int:
        """Live count of active (non-closed) subscribers."""
        ...


# ---------------------------------------------------------------------------
# Top-level dashboard service
# ---------------------------------------------------------------------------


@runtime_checkable
class DashboardProtocol(Protocol):
    """Structural protocol for the Rex dashboard service.

    Implementations provide the three pillars of the dashboard:

    - ``session_manager`` — auth for dashboard API endpoints
    - ``broadcaster`` — SSE notification delivery
    - ``register_routes`` — mount the dashboard's HTTP handlers on a web app

    Any class that exposes these attributes and method satisfies this protocol
    via structural subtyping (``isinstance`` check supported at runtime).
    """

    @property
    def session_manager(self) -> SessionManagerProtocol:
        """The session manager used for authenticating dashboard requests."""
        ...

    @property
    def broadcaster(self) -> NotificationBroadcasterProtocol:
        """The SSE broadcaster used to push notifications to connected clients."""
        ...

    def register_routes(self, app: Any) -> None:
        """Mount dashboard HTTP routes on *app*.

        Args:
            app: The web application object (e.g., a Flask ``Flask`` instance or
                an OpenClaw app equivalent).  Implementations must register all
                dashboard endpoints on this object.
        """
        ...
