"""RexAgent -- Rex registered as an OpenClaw agent.

Provides a thin wrapper that presents Rex's LLM client as an OpenClaw
agent.  When the OpenClaw gateway is configured (``openclaw_gateway_url``
in config), LLM calls are routed through ``/v1/chat/completions`` over
HTTP.  Otherwise the class operates in *standalone* mode and answers
prompts directly via Rex's local LLM client.

The ``user`` field in OpenClaw chat completions is derived from
:meth:`~rex.openclaw.identity_adapter.IdentityAdapter.get_openclaw_user_key`
so that OpenClaw maintains a stable per-user session automatically.

Intended usage::

    from rex.openclaw.agent import RexAgent

    agent = RexAgent()
    reply = agent.respond("What time is it?")

    # With explicit user key for history persistence:
    reply = agent.respond("What time is it?", user_key="alice")
"""

from __future__ import annotations

import logging
from collections.abc import Generator
from typing import Any

from rex.config import AppConfig, load_config
from rex.llm_client import LanguageModel
from rex.openclaw.errors import OpenClawAPIError, OpenClawAuthError, OpenClawConnectionError
from rex.openclaw.http_client import get_openclaw_client, stream_sentences
from rex.openclaw.identity_adapter import IdentityAdapter
from rex.openclaw.memory_adapter import MemoryAdapter
from rex.openclaw.policy_adapter import PolicyAdapter

logger = logging.getLogger(__name__)


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
        policy_adapter: Optional :class:`PolicyAdapter` instance used by
            :meth:`call_tool` to enforce Rex's risk/approval policies before
            executing any tool.  When *None*, a default adapter is created
            (uses Rex's global :class:`~rex.policy_engine.PolicyEngine`).
            Inject a custom adapter in tests to control which policies apply.
        identity_adapter: Optional :class:`IdentityAdapter` instance for
            resolving the stable OpenClaw user key.  When *None*, a default
            adapter is created.  Inject a custom adapter in tests to control
            user resolution.
    """

    def __init__(
        self,
        llm: LanguageModel | None = None,
        *,
        agent_name: str | None = None,
        system_prompt: str | None = None,
        config: AppConfig | None = None,
        profile_name: str | None = None,
        memory_adapter: MemoryAdapter | None = None,
        policy_adapter: PolicyAdapter | None = None,
        identity_adapter: IdentityAdapter | None = None,
    ) -> None:
        self._llm = llm
        # Derive agent name and persona from Rex config when not supplied.
        from rex.openclaw.config import (
            apply_profile_to_config,
            build_agent_config,
            build_system_prompt,
        )

        _base_config = config or load_config()
        # Apply a named profile if requested — updates capabilities and active_profile.
        if profile_name is not None:
            _base_config = apply_profile_to_config(_base_config, profile_name)

        _cfg = build_agent_config(_base_config)
        self.agent_name = agent_name or _cfg.get("agent_name", "Rex")
        self.system_prompt = system_prompt or build_system_prompt(_base_config)
        self._config = _base_config
        self._registered = False
        self._memory = memory_adapter or MemoryAdapter()
        self._policy = policy_adapter or PolicyAdapter()
        self._identity = identity_adapter or IdentityAdapter()

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

        When the OpenClaw gateway is configured, this method will
        register Rex as an agent via the gateway API.  When no gateway
        is configured it logs a warning and returns ``None``.

        Returns:
            The OpenClaw agent handle, or ``None`` when the gateway is
            not configured.
        """
        if get_openclaw_client(load_config()) is None:
            logger.warning(
                "OpenClaw gateway not configured -- %s running in standalone mode",
                self.agent_name,
            )
            return None

        # FUTURE: OpenClaw agent registration via HTTP once the gateway
        # exposes a registration endpoint (not yet available as of v0.x).
        logger.warning("OpenClaw agent registration not yet available in gateway API")
        self._registered = False
        return None

    # ------------------------------------------------------------------
    # Policy-checked tool execution
    # ------------------------------------------------------------------

    def call_tool(
        self,
        tool_name: str,
        args: dict[str, Any] | None = None,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute *tool_name* after passing it through Rex's policy adapter.

        This is the policy-checked tool execution path for the OpenClaw
        agent.  The :class:`~rex.openclaw.policy_adapter.PolicyAdapter`
        is consulted first; if the tool is denied or requires approval,
        the appropriate exception is raised before any execution occurs.
        If the policy allows execution, the tool is dispatched via
        :func:`~rex.openclaw.tool_executor.execute_tool` with policy and audit
        checks disabled (the adapter has already run the policy check).

        Args:
            tool_name: Name of the tool to execute.
            args: Keyword arguments forwarded to the tool.  Defaults to
                an empty dict.
            metadata: Optional policy-evaluation context (e.g.
                ``{"recipient": "user@example.com"}``).  Forwarded to
                :meth:`~rex.openclaw.policy_adapter.PolicyAdapter.guard`.

        Returns:
            The tool result dict as returned by
            :func:`~rex.openclaw.tool_executor.execute_tool`.

        Raises:
            :exc:`~rex.openclaw.tool_executor.PolicyDeniedError`: If the policy
                denies the tool call.
            :exc:`~rex.openclaw.tool_executor.ApprovalRequiredError`: If the policy
                requires user approval before execution.
        """
        from rex.openclaw.tool_executor import execute_tool

        # Policy gate — raises PolicyDeniedError or ApprovalRequiredError if blocked.
        self._policy.guard(tool_name, metadata)

        return execute_tool(
            {"tool": tool_name, "args": args or {}},
            {},
            skip_policy_check=True,  # policy already enforced above
            skip_credential_check=False,
            skip_audit_log=False,
        )

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

        # --- OpenClaw HTTP path ---
        client = get_openclaw_client(self._config)
        if client is not None and self._config.use_openclaw_voice_backend:
            try:
                # Derive a stable user key: explicit user_key > identity chain > "rex"
                oc_user = (
                    user_key if user_key is not None else self._identity.get_openclaw_user_key()
                )
                payload: dict[str, Any] = {
                    "model": self._config.llm_model,
                    "messages": messages,
                    "user": oc_user,
                }
                response = client.post("/v1/chat/completions", json=payload)
                reply: str = response["choices"][0]["message"]["content"]
                logger.debug("OpenClaw responded via HTTP for user_key=%r", user_key)
            except (OpenClawConnectionError, OpenClawAuthError, OpenClawAPIError, KeyError) as exc:
                logger.warning(
                    "OpenClaw chat completions failed (%s) — falling back to local LLM",
                    exc,
                )
                reply = self.llm.generate(messages=messages)
                logger.info("Fallback to local LLM succeeded for user_key=%r", user_key)
        else:
            reply = self.llm.generate(messages=messages)

        # Persist the exchange so future calls have context.
        if user_key is not None:
            self._memory.append_entry(user_key, {"role": "user", "text": prompt})
            self._memory.append_entry(user_key, {"role": "assistant", "text": reply})

        return reply

    def respond_stream(
        self,
        prompt: str,
        *,
        user_key: str | None = None,
    ) -> Generator[str, None, None]:
        """Stream a response as sentence-sized chunks.

        Yields partial sentences as they arrive from the OpenClaw
        gateway.  Falls back to :meth:`respond` (non-streaming, single
        yield) if streaming is unavailable or fails mid-response.

        Args:
            prompt: The user's input text.
            user_key: Optional user identifier for history persistence.

        Yields:
            Sentence-sized strings suitable for incremental TTS.
        """
        if not prompt or not prompt.strip():
            raise ValueError("prompt must not be empty")

        messages: list[dict[str, str]] = [
            {"role": "system", "content": self.system_prompt},
        ]

        if user_key is not None:
            history = self._memory.load_recent(user_key)
            for turn in history:
                role = turn.get("role", "user")
                text = turn.get("text", "")
                messages.append({"role": role, "content": text})

        messages.append({"role": "user", "content": prompt})

        client = get_openclaw_client(self._config)
        collected: list[str] = []

        if client is not None and self._config.use_openclaw_voice_backend:
            oc_user = user_key if user_key is not None else self._identity.get_openclaw_user_key()
            payload: dict[str, Any] = {
                "model": self._config.llm_model,
                "messages": messages,
                "user": oc_user,
                "stream": True,
            }
            try:
                raw_chunks = client.post_stream("/v1/chat/completions", json=payload)
                for sentence in stream_sentences(raw_chunks):
                    collected.append(sentence)
                    yield sentence
            except (OpenClawConnectionError, OpenClawAuthError, OpenClawAPIError) as exc:
                logger.warning("OpenClaw stream failed (%s) -- falling back", exc)
                if collected:
                    # Partial content already yielded; fall back for remainder is not
                    # feasible, so just log and stop.
                    pass
                else:
                    # Nothing yielded yet -- fall back to non-streaming local LLM.
                    reply = self.llm.generate(messages=messages)
                    collected.append(reply)
                    yield reply
        else:
            reply = self.llm.generate(messages=messages)
            collected.append(reply)
            yield reply

        # Persist the full response.
        full_reply = " ".join(collected)
        if user_key is not None:
            self._memory.append_entry(user_key, {"role": "user", "text": prompt})
            self._memory.append_entry(user_key, {"role": "assistant", "text": full_reply})
