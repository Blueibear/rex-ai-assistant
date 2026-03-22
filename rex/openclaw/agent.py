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

    # With conversation history persistence:
    reply = agent.respond("What time is it?", user_key="alice")

The ``register()`` hook will be filled-in once OpenClaw's agent
registration API is confirmed (see PRD Section 8.3 open dependency).
"""

from __future__ import annotations

import logging
from importlib.util import find_spec
from typing import Any

from rex.config import AppConfig
from rex.llm_client import LanguageModel
from rex.openclaw.memory_adapter import MemoryAdapter

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
            Defaults to the capitalised wakeword from ``AppConfig`` (``"Rex"``).
        system_prompt: System/persona prompt injected into every ``respond``
            call.  When *None*, :func:`rex.openclaw.config.build_system_prompt`
            derives the prompt from ``AppConfig`` fields (wakeword, profile,
            location, timezone, capabilities).
        config: Optional ``AppConfig`` instance used to derive ``agent_name``
            and ``system_prompt`` when those arguments are not supplied.
            Defaults to the global config loaded by ``rex.config.load_config()``.
        memory_adapter: Optional :class:`MemoryAdapter` instance for
            conversation-history persistence.  When *None*, a default adapter
            is created (uses Rex's file-based fallback).  Inject a custom
            adapter in tests to control the memory root directory.
    """

    def __init__(
        self,
        llm: LanguageModel | None = None,
        *,
        agent_name: str | None = None,
        system_prompt: str | None = None,
        config: AppConfig | None = None,
        memory_adapter: MemoryAdapter | None = None,
    ) -> None:
        self._llm = llm
        # Derive agent name and persona from Rex config when not supplied.
        from rex.openclaw.config import build_agent_config, build_system_prompt

        _cfg = build_agent_config(config)
        self.agent_name = agent_name or _cfg.get("agent_name", "Rex")
        self.system_prompt = system_prompt or build_system_prompt(config)
        self._registered = False
        self._memory = memory_adapter or MemoryAdapter()

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
        logger.warning("OpenClaw agent registration stub — update once API is confirmed (PRD §8.3)")
        self._registered = False
        return None

    # ------------------------------------------------------------------
    # Core response method
    # ------------------------------------------------------------------

    def respond(self, prompt: str, *, user_key: str | None = None) -> str:
        """Generate a response to *prompt* using Rex's LLM client.

        The system prompt (persona) is prepended to the message list so
        that the model always has Rex's identity context.

        When *user_key* is provided, previous conversation turns for that
        user are loaded from the memory adapter and prepended to the
        message list before the current prompt.  After the LLM responds,
        both the user turn and the assistant turn are persisted via the
        memory adapter so that subsequent calls accumulate context.

        Args:
            prompt: The user's input text.
            user_key: Optional user identifier for history persistence.
                When ``None``, no history is loaded or saved.

        Returns:
            The model's response as a plain string.
        """
        if not prompt or not prompt.strip():
            raise ValueError("prompt must not be empty")

        messages: list[dict[str, str]] = [
            {"role": "system", "content": self.system_prompt},
        ]

        # Prepend stored history when a user context is available.
        if user_key is not None:
            history = self._memory.load_recent(user_key)
            for turn in history:
                role = turn.get("role", "user")
                text = turn.get("text", "")
                messages.append({"role": role, "content": text})

        messages.append({"role": "user", "content": prompt})
        reply = self.llm.generate(messages=messages)

        # Persist the exchange so future calls have context.
        if user_key is not None:
            self._memory.append_entry(user_key, {"role": "user", "text": prompt})
            self._memory.append_entry(user_key, {"role": "assistant", "text": reply})

        return reply
