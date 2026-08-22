"""LLM context assembly for assistant requests (US-014).

Extracted from ``rex.assistant.Assistant._build_prompt``,
``_build_messages``, ``_build_system_context``,
``_get_active_personality_prompt``, and ``_load_user_profile_context``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from rex.context.cache import ContextArtifactCache, ContextCacheKey, ContextCacheMetrics
from rex.context.revisions import ContextCacheRequest, build_context_cache_versions
from rex.context.source_policy import ContextSourcePolicyStore
from rex.runtime.turn import TurnScope

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (previously class-level on Assistant)
# ---------------------------------------------------------------------------

VOICE_CONCISE_INSTRUCTION = (
    "[Respond in 1-3 sentences. Keep your reply short and conversational for voice output.]"
)
# Legacy alias kept for backward compatibility with tests/code that reference
# the underscore-prefixed name.
_VOICE_CONCISE_INSTRUCTION = VOICE_CONCISE_INSTRUCTION

_TOOL_INSTRUCTIONS = (
    "You have access to the following tools. When you need live data (current time, "
    "weather, or web search results), you MUST respond with ONLY a single-line tool "
    "request in this exact format — no other text on that line:\n"
    'TOOL_REQUEST: {"tool": "<name>", "args": {<arguments>}}\n'
    "\n"
    "Available tools:\n"
    "- time_now: Get the current local time for a location. "
    'Args: {"location": "City, Region"}\n'
    "- weather_now: Get current weather for a location. "
    'Args: {"location": "City, Region"}\n'
    "- web_search: Search the web. "
    'Args: {"query": "search terms"}\n'
    "\n"
    "IMPORTANT: When asked about the current time in ANY location, ALWAYS use "
    "the time_now tool. Do NOT guess or convert times yourself. Never claim "
    "you added, changed, sent, saved, or created something unless a tool result "
    "confirms that action succeeded."
)


# ---------------------------------------------------------------------------
# ContextPackage
# ---------------------------------------------------------------------------


@dataclass
class ContextPackage:
    """Ready-to-use LLM input produced by ContextBuilder.build()."""

    messages: list[dict] = field(default_factory=list)
    system_prompt: str = ""
    session_id: str = "default"
    user_facts: dict = field(default_factory=dict)
    prompt: str = ""  # text-format prompt for non-chat LLMs


@dataclass(frozen=True, slots=True)
class PrivateContextArtifacts:
    """Immutable user-scoped fragments safe under a validated cache key."""

    personality_prompt: str | None
    profile_context: str | None
    facts_context: str | None
    user_facts: tuple[tuple[str, str], ...] = ()

    def facts_dict(self) -> dict[str, str]:
        return dict(self.user_facts)


# ---------------------------------------------------------------------------
# ContextBuilder
# ---------------------------------------------------------------------------


class ContextBuilder:
    """Assembles LLM context from history, user profile, and personality.

    Args:
        settings:         App config/settings object (``AppConfig`` or ``Settings``).
        history:          Mutable reference to the in-memory conversation history list.
        user_id:          Default user/session identifier, or ``None`` when the
                          owning assistant is identity-unbound (issue #303); a
                          per-request ``active_user_id`` is then required for
                          user-scoped context.
        followup_engine:  Optional follow-up engine for ``format_followups()``.
        history_provider: Optional callable returning the current history list.
                          When given it takes precedence over *history*, so the
                          builder always reads the caller's live (per-user)
                          history instead of a snapshot taken at construction
                          time (issue #303).  Providers may optionally accept
                          the effective user ID as a single argument.
    """

    def __init__(
        self,
        settings: Any,
        history: list,
        user_id: str | None,
        *,
        followup_engine: Any = None,
        history_provider: Callable[..., list] | None = None,
        context_cache: ContextArtifactCache[PrivateContextArtifacts] | None = None,
        capability_registry: Any = None,
        source_policy_store: ContextSourcePolicyStore | None = None,
    ) -> None:
        self._settings = settings
        self._history = history
        self._user_id = user_id
        self._followup_engine = followup_engine
        self._history_provider = history_provider
        self._context_cache = context_cache or ContextArtifactCache(max_entries=128)
        self._capability_registry = capability_registry
        self._source_policy_store = source_policy_store

    def _current_history(self, user_id: str | None = None) -> list:
        """Return the live history list (provider-backed when configured).

        *user_id* is the effective request user; identity-aware providers
        receive it so per-request identities never route through shared
        mutable state (issue #303).  Zero-arg legacy providers keep working.
        """
        if self._history_provider is not None:
            try:
                return self._history_provider(user_id)
            except TypeError:
                return self._history_provider()
        return self._history

    # ------------------------------------------------------------------
    # Primary entry point
    # ------------------------------------------------------------------

    def build(
        self,
        user_message: str,
        *,
        voice_mode: bool = False,
        active_user_id: str | None = None,
        tool_context: str | None = None,
        cache_request: ContextCacheRequest | None = None,
    ) -> ContextPackage:
        """Build and return a :class:`ContextPackage` for LLM input."""
        system_prompt = self.build_system_context()
        session_id = active_user_id or self._user_id or ""
        private_artifacts = self._resolve_private_artifacts(active_user_id, cache_request)

        messages = self._build_messages(
            user_message,
            system_prompt=system_prompt,
            voice_mode=voice_mode,
            active_user_id=active_user_id,
            tool_context=tool_context,
            private_artifacts=private_artifacts,
        )
        prompt = self._build_prompt(
            user_message,
            system_prompt=system_prompt,
            voice_mode=voice_mode,
            active_user_id=active_user_id,
            tool_context=tool_context,
            private_artifacts=private_artifacts,
        )
        return ContextPackage(
            messages=messages,
            system_prompt=system_prompt,
            session_id=session_id,
            user_facts=private_artifacts.facts_dict(),
            prompt=prompt,
        )

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    @staticmethod
    def format_context_documents(documents: list[Any] | tuple[Any, ...]) -> str:
        """Render bounded document context with stable provenance markers."""
        sections: list[str] = []
        for doc in list(documents)[:5]:
            source_id = str(getattr(doc, "source_id", "")).strip()
            title = str(getattr(doc, "title", "")).strip()
            content = str(getattr(doc, "content", "")).strip()
            if not source_id or not content:
                continue
            if len(content) > 2000:
                content = content[:2000].rstrip() + "…"
            sections.append(f"[Context source: {source_id} | {title}]\n{content}")
        return "\n\n".join(sections)

    def build_system_context(self) -> str:
        """Return a system context string with current date/time and user location."""
        _settings = self._settings
        tz_name: str | None = getattr(_settings, "default_timezone", None)
        if not tz_name:
            from rex.geolocation import get_cached_timezone

            tz_name = get_cached_timezone()

        try:
            if tz_name:
                from zoneinfo import ZoneInfo

                now = datetime.now(tz=ZoneInfo(tz_name))
            else:
                now = datetime.now(tz=UTC)
                tz_name = "UTC"
        except Exception:
            now = datetime.now(tz=UTC)
            if not tz_name:
                tz_name = "UTC"

        lines = [f"Current date and time: {now.strftime('%Y-%m-%d %H:%M')} {tz_name}"]

        location: str | None = getattr(_settings, "default_location", None)
        if not location:
            from rex.geolocation import get_cached_city

            location = get_cached_city()
        if location:
            lines.append(f"User location: {location}")

        lines.append("")
        lines.append(_TOOL_INSTRUCTIONS)

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_user_facts(self, user_id: str | None) -> dict:
        """Return raw user facts dict for the given user_id.

        Without any identity no private facts are read (fail closed, #303).
        """
        uid = user_id or self._user_id
        if not uid:
            return {}
        try:
            from rex.user_facts import recall_all

            return recall_all(uid)
        except Exception:
            return {}

    @staticmethod
    def _format_user_facts_context(user_id: str, facts: dict[str, str]) -> str | None:
        if not facts:
            return None
        pairs = "; ".join(f"{key}={value}" for key, value in facts.items())
        return f"[Remembered facts about {user_id}: {pairs}]"

    def _build_private_artifacts(self, active_user_id: str | None) -> PrivateContextArtifacts:
        user_facts = self._get_user_facts(active_user_id)
        profile_context: str | None = None
        facts_context: str | None = None
        if active_user_id is not None:
            profile_context = self._load_user_profile_context(active_user_id)
            facts_context = self._format_user_facts_context(active_user_id, user_facts)
        return PrivateContextArtifacts(
            personality_prompt=self._get_active_personality_prompt(active_user_id),
            profile_context=profile_context,
            facts_context=facts_context,
            user_facts=tuple((str(key), str(value)) for key, value in user_facts.items()),
        )

    def _resolve_private_artifacts(
        self,
        active_user_id: str | None,
        cache_request: ContextCacheRequest | None,
    ) -> PrivateContextArtifacts:
        def build() -> PrivateContextArtifacts:
            return self._build_private_artifacts(active_user_id)

        effective_user = active_user_id or self._user_id
        if (
            cache_request is None
            or effective_user is None
            or cache_request.scope is not TurnScope.USER
            or cache_request.user_id != effective_user
        ):
            return build()
        try:
            source_policy_revision = (
                self._source_policy_store.revision_for_user(effective_user)
                if self._source_policy_store is not None
                else None
            )
            versions = build_context_cache_versions(
                cache_request,
                self._settings,
                self._capability_registry,
                source_policy_revision=source_policy_revision,
            )
            key = ContextCacheKey.private(effective_user, versions)
        except Exception as exc:
            logger.debug("Context artifact cache bypassed: %s", type(exc).__name__)
            return build()
        return self._context_cache.get_or_build(key, build)

    def context_cache_metrics(self) -> dict[str, ContextCacheMetrics]:
        return self._context_cache.metrics_snapshot()

    def _load_user_profile_context(self, user_id: str) -> str | None:
        """Load a user's memory profile and format it as a context string."""
        try:
            from rex.memory_utils import MEMORY_ROOT, load_memory_profile

            profile = load_memory_profile(user_id, MEMORY_ROOT)
        except Exception:
            return None

        parts: list[str] = []
        name = profile.get("name")
        if name:
            parts.append(f"name={name}")

        prefs = profile.get("preferences")
        if isinstance(prefs, dict):
            tone = prefs.get("tone")
            if tone:
                parts.append(f"tone={tone}")
            topics = prefs.get("topics")
            if isinstance(topics, list) and topics:
                parts.append(f"interests={', '.join(str(t) for t in topics[:5])}")

        if not parts:
            return f"[Active user: {user_id}]"
        return f"[Active user: {user_id} — {'; '.join(parts)}]"

    def _get_active_personality_prompt(self, active_user_id: str | None) -> str | None:
        """Return the system prompt for the user's configured personality, or None."""
        from rex.personality import get_personality

        personality_name: str | None = None
        uid = active_user_id or self._user_id
        if uid:
            try:
                from rex.identity import get_user_profile

                profile = get_user_profile(uid)
                if profile:
                    prefs = profile.get("preferences", {})
                    if isinstance(prefs, dict):
                        personality_name = prefs.get("personality")
            except Exception:
                pass

        if not personality_name:
            personality_name = getattr(self._settings, "personality", None)

        if not personality_name:
            from rex.personality import DEFAULT_PERSONALITY

            personality_name = DEFAULT_PERSONALITY

        return get_personality(personality_name).system_prompt

    def _build_messages(
        self,
        user_message: str,
        *,
        system_prompt: str,
        voice_mode: bool = False,
        active_user_id: str | None = None,
        tool_context: str | None = None,
        private_artifacts: PrivateContextArtifacts | None = None,
    ) -> list[dict]:
        messages: list[dict] = [{"role": "system", "content": system_prompt}]
        artifacts = private_artifacts or self._build_private_artifacts(active_user_id)

        if artifacts.personality_prompt:
            messages.append({"role": "system", "content": artifacts.personality_prompt})

        if active_user_id is not None:
            messages.append(
                {
                    "role": "system",
                    "content": artifacts.profile_context or f"[Active user: {active_user_id}]",
                }
            )
            if artifacts.facts_context:
                messages.append({"role": "system", "content": artifacts.facts_context})

        if tool_context:
            messages.append({"role": "system", "content": tool_context})

        engine = self._followup_engine
        effective_user = active_user_id or self._user_id
        if engine and effective_user and hasattr(engine, "format_followups"):
            try:
                followups = engine.format_followups(effective_user)
                if followups:
                    messages.append({"role": "system", "content": str(followups)})
            except Exception as exc:
                logger.debug("format_followups failed: %s", exc)

        if voice_mode:
            messages.append({"role": "system", "content": _VOICE_CONCISE_INSTRUCTION})

        for turn in self._current_history(active_user_id)[-4:]:
            speaker = str(turn.speaker).strip().lower()
            role = "assistant" if speaker in {"assistant", "rex"} else "user"
            messages.append({"role": role, "content": turn.text})

        messages.append({"role": "user", "content": user_message})
        return messages

    def _build_prompt(
        self,
        user_message: str,
        *,
        system_prompt: str,
        voice_mode: bool = False,
        active_user_id: str | None = None,
        tool_context: str | None = None,
        private_artifacts: PrivateContextArtifacts | None = None,
    ) -> str:
        history_lines = [system_prompt]
        artifacts = private_artifacts or self._build_private_artifacts(active_user_id)

        if artifacts.personality_prompt:
            history_lines.append(artifacts.personality_prompt)

        if active_user_id is not None:
            history_lines.append(artifacts.profile_context or f"[Active user: {active_user_id}]")
            if artifacts.facts_context:
                history_lines.append(artifacts.facts_context)

        if tool_context:
            history_lines.append(tool_context)

        effective_user = active_user_id or self._user_id
        history_lines += [
            f"{turn.speaker}: {turn.text}" for turn in self._current_history(active_user_id)[-4:]
        ]
        history_lines.append(f"user: {user_message}")

        if voice_mode:
            history_lines.append(_VOICE_CONCISE_INSTRUCTION)

        engine = self._followup_engine
        if engine and effective_user and hasattr(engine, "format_followups"):
            try:
                followups = engine.format_followups(effective_user)
                if followups:
                    history_lines.append(str(followups))
            except Exception as exc:
                logger.debug("format_followups failed: %s", exc)

        history_lines.append("assistant:")
        return "\n".join(history_lines)
