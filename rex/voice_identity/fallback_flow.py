"""Fallback identity flow for uncertain or unknown speakers.

When voice identity returns a ``review`` or ``unknown`` decision, this
module bridges into the existing session-scoped identity system
(:mod:`rex.identity`) so the assistant can ask the user to confirm who
they are or continue with the current session user.

Design notes
------------
* No new identity system is created; this delegates to
  :func:`rex.identity.resolve_active_user` and
  :func:`rex.identity.set_session_user`.
* Audio-based confirmation (voice PIN, re-enrollment) is deferred to a
  future PR.  The current implementation is a callable hook that the
  voice loop can invoke.
"""

from __future__ import annotations

import logging

from rex.identity import resolve_active_user, set_session_user
from rex.voice_identity.types import RecognitionDecision, RecognitionResult

logger = logging.getLogger(__name__)


def resolve_speaker_identity(
    result: RecognitionResult,
    *,
    explicit_user: str | None = None,
    config: dict | None = None,
) -> str | None:
    """Determine the active user from a recognition result.

    Decision logic:

    * **recognized** -- accept ``result.best_user_id`` and update the
      session so downstream commands see the correct user.
    * **review** -- if the existing session/config already resolves to
      the same user as the best match, accept it silently.  Otherwise
      fall back to the existing identity resolution chain (which may
      prompt interactively in a future PR).
    * **unknown** -- fall through to the existing identity chain without
      setting a new session user.

    Returns:
        The resolved user ID, or ``None`` if no user could be determined.
    """
    if result.decision == RecognitionDecision.RECOGNIZED and result.best_user_id:
        logger.info(
            "Speaker recognized as %s (score=%.3f)",
            result.best_user_id,
            result.score,
        )
        set_session_user(result.best_user_id)
        return result.best_user_id

    if result.decision == RecognitionDecision.REVIEW and result.best_user_id:
        # Check whether the existing identity chain already agrees
        current = resolve_active_user(explicit_user, config=config)
        if current == result.best_user_id:
            logger.info(
                "Speaker review: existing session user matches best guess %s "
                "(score=%.3f); accepting.",
                result.best_user_id,
                result.score,
            )
            return current
        logger.info(
            "Speaker review: score=%.3f for %s but session user is %s; "
            "falling back to identity resolution.",
            result.score,
            result.best_user_id,
            current,
        )
        # Fall through to existing identity resolution
        return resolve_active_user(explicit_user, config=config)

    # Unknown speaker -- use existing identity chain
    logger.info(
        "Speaker unknown (score=%.3f); using existing identity chain.",
        result.score,
    )
    return resolve_active_user(explicit_user, config=config)
