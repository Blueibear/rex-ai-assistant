"""RexAgent — Rex registered as an OpenClaw agent.

Provides a thin wrapper that presents Rex's LLM client as an OpenClaw
agent.  When the ``openclaw`` package is not installed the class operates
in *standalone* mode and answers prompts directly via Rex's LLM client;
OpenClaw integration is a no-op that logs a warning.

Intended usage
--------------
::

    from rex.openclaw.agent import RexAgent

    agent = RexAgent()
    agent.register()          # registers with OpenClaw if available
    reply = agent.respond("What time is it?")

The ``register()`` hook will be filled-in once OpenClaw's agent
registration API is confirmed (see PRD Section 8.3 open dependency).
"""

from __future__ import annotations

import logging
from importlib.util import find_spec
from typing import Any

from rex.llm_client import LanguageModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional OpenClaw availability flag
# ---------------------------------------------------------------------------

OPENCLAW_AVAILABLE: bool = find_spec("openclaw") is not None

if OPENCLAW_AVAILABLE:  # pragma: no cover
    import openclaw as _openclaw  # type: ignore[import-not-found]
else:
    _openclaw = None  # type: ignore[assignment]


class RexAgent:
    """Rex AI Assistant registered as an OpenClaw agent.

    Args:
        llm: Optional ``LanguageModel`` instance.  When *None*, a new
            instance is created on first use (lazy, to avoid loading
            model weights at import time).
        agent_name: The name under which Rex registers with OpenClaw.
        system_prompt: Optional system/persona prompt injected into every
            ``respond`` call.  Defaults to a minimal Rex identity string;
            set via :class:`rex.openclaw.config` once that module exists.
    """

    def __init__(
        self,
        llm: LanguageModel | None = None,
        *,
        agent_name: str = "Rex",
        system_prompt: str | None = None,
    ) -> None:
        self._llm = llm
        self.agent_name = agent_name
        self.system_prompt = system_prompt or (
            "You are Rex, a helpful and friendly AI assistant."
        )
        self._registered = False

    # ------------------------------------------------------------------
    # LLM access
    # ------------------------------------------------------------------

    @property
    def llm(self) -> LanguageModel:
        """Return the language model, creating it on first access."""
        if self._llm is None:
            self._llm = LanguageModel()
        return self._llm

    # ------------------------------------------------------------------
    # OpenClaw registration
    # ------------------------------------------------------------------

    def register(self) -> Any:
        """Register this agent with OpenClaw.

        When the ``openclaw`` package is available this method calls
        OpenClaw's agent registration API.  When it is not available it
        logs a warning and returns ``None`` so that callers do not need
        to branch on availability.

        .. note::
            The exact OpenClaw registration call is a stub (see PRD
            Section 8.3 — *"Confirm OpenClaw's Python API for agent
            registration"*).  Replace the ``# TODO`` line below once
            the API is confirmed.

        Returns:
            The OpenClaw agent handle returned by the registration call,
            or ``None`` when OpenClaw is not available.
        """
        if not OPENCLAW_AVAILABLE:
            logger.warning(
                "openclaw package not installed — %s running in standalone mode",
                self.agent_name,
            )
            return None

        # TODO: replace with real OpenClaw agent registration once API is confirmed.
        # Expected shape (to be verified):
        #   handle = _openclaw.register_agent(
        #       name=self.agent_name,
        #       handler=self.respond,
        #   )
        #   self._registered = True
        #   return handle
        logger.warning(
            "OpenClaw agent registration stub — update once API is confirmed (PRD §8.3)"
        )
        self._registered = False
        return None

    # ------------------------------------------------------------------
    # Core response method
    # ------------------------------------------------------------------

    def respond(self, prompt: str) -> str:
        """Generate a response to *prompt* using Rex's LLM client.

        The system prompt (persona) is prepended to the message list so
        that the model always has Rex's identity context.

        Args:
            prompt: The user's input text.

        Returns:
            The model's response as a plain string.
        """
        if not prompt or not prompt.strip():
            raise ValueError("prompt must not be empty")

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        return self.llm.generate(messages=messages)
