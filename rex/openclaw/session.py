"""OpenClaw session bridge — maps Rex user identity to OpenClaw session model.

Translates Rex's user identity (from ``rex.identity``) into the session
context dict that OpenClaw's agent API expects.  Because OpenClaw's exact
session schema is an open dependency (see PRD Section 8.3), the output is
a plain ``dict`` that will be refined once the API surface is confirmed.

Typical usage::

    from rex.openclaw.session import build_session_context

    # Resolve the active user and build the session dict:
    session = build_session_context()

    # Or pass an explicit user ID (e.g., from a --user flag):
    session = build_session_context(explicit_user="james")
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from rex.identity import get_user_profile, list_known_users, resolve_active_user


def build_session_context(
    explicit_user: str | None = None,
    *,
    config: dict | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build an OpenClaw-compatible session context from Rex user identity.

    Resolves the active Rex user through the standard priority chain
    (explicit arg → session state → config), then constructs a session dict
    with identity, profile, and timing fields.

    .. note::
        The exact keys expected by OpenClaw are not yet confirmed (PRD §8.3).
        Keys prefixed with ``rex_`` are Rex-specific and will be remapped once
        the OpenClaw session schema is known.

    Args:
        explicit_user: User ID from a ``--user`` flag or caller-supplied
            context.  Takes priority over all other sources.
        config: Rex config dict (the raw ``dict`` form of ``AppConfig``).
            Used as a fallback in the user resolution chain.
        metadata: Optional arbitrary key/value pairs to include in the
            session context (e.g., channel, request ID).

    Returns:
        A plain ``dict`` representing the session context.  Guaranteed keys:

        - ``user_id`` (``str | None``) — resolved Rex user ID
        - ``session_started_at`` (``str``) — ISO 8601 UTC timestamp
        - ``rex_known_users`` (``list[dict]``) — users with memory profiles
        - ``rex_user_profile`` (``dict | None``) — profile for the active user
    """
    user_id = resolve_active_user(explicit_user, config=config)

    # Load the user's Rex profile (from Memory/<user_id>/core.json)
    user_profile: dict[str, Any] | None = None
    if user_id:
        try:
            user_profile = get_user_profile(user_id)
        except Exception:
            user_profile = None

    session: dict[str, Any] = {
        # Primary identity fields (remapped once OpenClaw schema is known)
        "user_id": user_id,
        "session_started_at": datetime.now(tz=UTC).isoformat(),
        # Rex-specific extras
        "rex_known_users": list_known_users(),
        "rex_user_profile": user_profile,
    }

    if metadata:
        session.update(metadata)

    return session
