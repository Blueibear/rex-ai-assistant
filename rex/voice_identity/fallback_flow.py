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
* Trusted identity provenance is staged separately from the user ID so
  output/media policy can distinguish recognized voice evidence from a
  fallback session/profile without treating either as permission.
* Audio-based confirmation (voice PIN, re-enrollment) is deferred to a
  future PR.
"""

from __future__ import annotations

import logging

from rex.identity import resolve_active_user, set_session_user, validate_user_id
from rex.runtime.invocation import stage_identity_resolution
from rex.runtime.turn import IdentityResolution
from rex.voice_identity.types import RecognitionDecision, RecognitionResult

logger = logging.getLogger(__name__)


def _stage_fallback_result(user_id: str | None) -> str | None:
    stage_identity_resolution(
        IdentityResolution.FALLBACK if user_id is not None else IdentityResolution.UNKNOWN
    )
    return user_id


def resolve_speaker_identity(
    result: RecognitionResult,
    *,
    explicit_user: str | None = None,
    config: dict | None = None,
) -> str | None:
    """Determine the active user and stage trusted resolution provenance.

    Decision logic:

    * **recognized** -- accept ``result.best_user_id``, update the session,
      and stage ``voice_recognized``.
    * **review** -- if the existing session/config already resolves to the
      same user as the best match, accept it and stage ``voice_review``.
      Otherwise use the existing identity chain and stage ``fallback`` (or
      ``unknown`` when no user resolves).
    * **unknown** -- use the existing identity chain and stage ``fallback``
      or ``unknown`` without setting a new session user.

    Returns:
        The resolved user ID, or ``None`` if no user could be determined.
    """
    # Fail safe before interpreting any result so malformed/exceptional paths
    # cannot retain a previous interaction's strong voice provenance.
    stage_identity_resolution(IdentityResolution.UNKNOWN)

    if result.decision == RecognitionDecision.RECOGNIZED and result.best_user_id:
        try:
            user_id = validate_user_id(result.best_user_id)
        except ValueError:
            logger.warning("Voice recognition returned an invalid user identity")
            return None
        logger.info(
            "Speaker recognized as %s (score=%.3f)",
            user_id,
            result.score,
        )
        set_session_user(user_id)
        stage_identity_resolution(IdentityResolution.VOICE_RECOGNIZED)
        return user_id

    if result.decision == RecognitionDecision.REVIEW and result.best_user_id:
        current = resolve_active_user(explicit_user, config=config)
        if current == result.best_user_id:
            logger.info(
                "Speaker review: existing session user matches best guess %s "
                "(score=%.3f); accepting.",
                result.best_user_id,
                result.score,
            )
            stage_identity_resolution(IdentityResolution.VOICE_REVIEW)
            return current
        logger.info(
            "Speaker review: score=%.3f for %s but session user is %s; "
            "falling back to identity resolution.",
            result.score,
            result.best_user_id,
            current,
        )
        return _stage_fallback_result(current)

    logger.info(
        "Speaker unknown (score=%.3f); using existing identity chain.",
        result.score,
    )
    return _stage_fallback_result(resolve_active_user(explicit_user, config=config))
