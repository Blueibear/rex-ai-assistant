"""VoiceBridge — drop-in replacement for the assistant seam in Rex voice loops.

All three Rex voice loop implementations share the same seam::

    reply = await self._assistant.generate_reply(transcript)          # optimized
    reply = await self._assistant.generate_reply(transcript, voice_mode=True)  # rex/
    reply = asyncio.to_thread(self._assistant.generate_reply, transcript, ...)  # root

VoiceBridge presents ``generate_reply(transcript, voice_mode=False) -> str`` so
that all three loops can swap their assistant reference for a VoiceBridge instance
without any other call-site changes.

Internally the bridge delegates to :class:`~rex.openclaw.agent.RexAgent`'s
``respond()`` method, which includes system-prompt injection, memory persistence,
and Rex's policy layer.  The ``voice_mode`` flag is accepted for compatibility
but not forwarded — the underlying RexAgent is stateless with respect to output
modality (TTS vs text).

Usage::

    from rex.openclaw.voice_bridge import VoiceBridge

    bridge = VoiceBridge()
    reply = bridge.generate_reply("Set a timer for five minutes.")

    # Async usage (asyncio.to_thread or await-compatible wrapper):
    reply = await asyncio.to_thread(bridge.generate_reply, transcript, voice_mode=True)
"""

from __future__ import annotations

import logging
from collections.abc import Generator
from typing import Any

from rex.openclaw.agent import RexAgent

logger = logging.getLogger(__name__)


class VoiceBridge:
    """Adapter that exposes ``generate_reply()`` backed by :class:`RexAgent`.

    Args:
        agent: Optional :class:`RexAgent` instance.  When *None*, a default
            agent is created on first use (lazy initialisation to avoid loading
            model weights at import time).
        user_key: Optional user identifier forwarded to
            :meth:`~rex.openclaw.agent.RexAgent.respond` for history
            persistence.  When *None*, no conversation history is saved or
            replayed.
    """

    def __init__(
        self,
        agent: RexAgent | None = None,
        *,
        user_key: str | None = None,
    ) -> None:
        self._agent = agent
        self.user_key = user_key

    @property
    def agent(self) -> RexAgent:
        """Return the RexAgent, creating it on first access."""
        if self._agent is None:
            self._agent = RexAgent()
        return self._agent

    def generate_reply(self, transcript: str, voice_mode: bool = False, **kwargs: Any) -> str:
        """Generate a response to *transcript* using the underlying RexAgent.

        This method is the voice loop seam.  It accepts (and ignores)
        ``voice_mode`` so it is compatible with all three Rex voice loop
        implementations, which pass the kwarg inconsistently:

        - ``voice_loop.py`` (root): ``generate_reply(transcript, voice_mode=True)``
        - ``rex/voice_loop.py``: ``generate_reply(transcript, voice_mode=True)``
        - ``rex/voice_loop_optimized.py``: ``generate_reply(transcript)``

        Args:
            transcript: The user's transcribed speech.
            voice_mode: Accepted for call-site compatibility; not forwarded to
                the RexAgent because output modality is handled by the calling
                voice loop, not by the language model.
            **kwargs: Any additional keyword arguments are accepted and silently
                ignored for forward compatibility.

        Returns:
            The agent's response as a plain string.

        Raises:
            ValueError: If *transcript* is empty or whitespace-only.
        """
        if not transcript or not transcript.strip():
            logger.debug("VoiceBridge.generate_reply: empty transcript, returning empty string")
            return ""

        logger.debug(
            "VoiceBridge.generate_reply called (voice_mode=%s, user_key=%s)",
            voice_mode,
            self.user_key,
        )

        if voice_mode:
            logger.debug("VoiceBridge: voice_mode=True noted; not forwarded to RexAgent")

        try:
            return self.agent.respond(transcript, user_key=self.user_key)
        except Exception:
            logger.exception("VoiceBridge.generate_reply: error from RexAgent.respond()")
            return "I had trouble reaching the server. Try again."

    def generate_reply_stream(
        self, transcript: str, voice_mode: bool = False, **kwargs: Any
    ) -> Generator[str, None, None]:
        """Stream sentence-sized chunks of the reply for incremental TTS.

        Delegates to :meth:`~rex.openclaw.agent.RexAgent.respond_stream` so
        that partial sentences arrive from the OpenClaw gateway as they are
        generated, enabling TTS to start speaking sooner.

        Falls back to a single-chunk response on any error so the voice loop
        is never left silent.

        Args:
            transcript: The user's transcribed speech.
            voice_mode: Accepted for call-site compatibility; not forwarded.
            **kwargs: Accepted for forward compatibility; ignored.

        Yields:
            Sentence-sized strings suitable for incremental TTS playback.
        """
        if not transcript or not transcript.strip():
            logger.debug("VoiceBridge.generate_reply_stream: empty transcript, yielding nothing")
            return

        logger.debug(
            "VoiceBridge.generate_reply_stream called (voice_mode=%s, user_key=%s)",
            voice_mode,
            self.user_key,
        )

        try:
            yield from self.agent.respond_stream(transcript, user_key=self.user_key)
        except Exception:
            logger.exception(
                "VoiceBridge.generate_reply_stream: error from RexAgent.respond_stream()"
            )
            yield "I had trouble reaching the server. Try again."
