"""OpenClaw identity adapter — US-P3-013.

Wraps Rex's user identity resolution (from :mod:`rex.identity`) as an
OpenClaw-compatible identity service.  Provides a single ``IdentityAdapter``
class that exposes session management and user resolution through a clean
interface, and builds the session context dict expected by the OpenClaw
agent.

Profile CRUD operations (``create_user_profile``, ``get_user_profile``,
``update_user_preferences``, ``list_known_users``) are delegated directly to
Rex's file-based implementation.  They are not wrapped, because they operate
on Rex's ``Memory/`` directory structure which is Rex-specific and will not
be replaced by OpenClaw.

When the ``openclaw`` package is not installed, :meth:`~IdentityAdapter.register`
logs a warning and returns ``None``.  All other methods work without OpenClaw.

Typical usage::

    from rex.openclaw.identity_adapter import IdentityAdapter

    adapter = IdentityAdapter()

    # Set the active user for this session
    adapter.set_user("james")

    # Resolve the active user (explicit override > session > config)
    user_id = adapter.resolve_user()

    # Build an OpenClaw-compatible session dict
    session = adapter.build_session()
"""

from __future__ import annotations

import logging
from typing import Any

from rex.identity import (
    clear_session_user,
    get_session_user,
    resolve_active_user,
    set_session_user,
)
from rex.openclaw.session import build_session_context

logger = logging.getLogger(__name__)


class IdentityAdapter:
    """Adapter that presents Rex's identity system to OpenClaw.

    Bridges Rex's four-level user resolution chain (explicit → session →
    config → None) to whatever user/session primitives OpenClaw exposes.
    Until OpenClaw's session schema is confirmed (PRD §8.3), the adapter
    returns plain dicts that match :func:`~rex.openclaw.session.build_session_context`.

    Args:
        config: Optional Rex config dict (raw ``dict`` form of ``AppConfig``).
            Passed to the identity resolution chain as a config fallback.
    """

    def __init__(self, config: dict | None = None) -> None:
        self._config = config or {}

    # ------------------------------------------------------------------
    # Session state
    # ------------------------------------------------------------------

    def get_user(self) -> str | None:
        """Return the active user from session state, or ``None``.

        Delegates to :func:`~rex.identity.get_session_user`.
        """
        return get_session_user()

    def set_user(self, user_id: str) -> None:
        """Set the active user in the Rex session state.

        Delegates to :func:`~rex.identity.set_session_user`.

        Args:
            user_id: The user ID to activate for the current session.
        """
        set_session_user(user_id)
        logger.info("IdentityAdapter: session user set to %r", user_id)

    def clear_user(self) -> None:
        """Clear the active user from session state.

        Delegates to :func:`~rex.identity.clear_session_user`.
        """
        clear_session_user()
        logger.info("IdentityAdapter: session user cleared")

    # ------------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------------

    def resolve_user(self, explicit_user: str | None = None) -> str | None:
        """Resolve the active user through Rex's priority chain.

        Priority: explicit_user → session state → config ``runtime.active_user``
        → config ``runtime.user_id``.

        Args:
            explicit_user: An optional override (e.g., from a ``--user`` flag).

        Returns:
            Resolved user ID, or ``None`` if no user can be determined.
        """
        return resolve_active_user(explicit_user, config=self._config)

    # ------------------------------------------------------------------
    # OpenClaw session context
    # ------------------------------------------------------------------

    def build_session(
        self,
        explicit_user: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build an OpenClaw-compatible session context dict.

        Delegates to :func:`~rex.openclaw.session.build_session_context`.
        The returned dict is suitable for passing to an OpenClaw agent as
        session context once the OpenClaw session schema is confirmed
        (PRD §8.3).

        Args:
            explicit_user: Optional user ID override.
            metadata: Optional extra key/value pairs to include in the
                session context (e.g., ``{"channel": "voice"}``).

        Returns:
            A plain dict with guaranteed keys ``user_id``,
            ``session_started_at``, ``rex_known_users``, and
            ``rex_user_profile``.
        """
        return build_session_context(
            explicit_user,
            config=self._config if self._config else None,
            metadata=metadata,
        )

