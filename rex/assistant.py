"""Async assistant orchestration."""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from .calendar_service import get_calendar_service
from .config import Settings, settings
from .followup_engine import FollowupEngine
from .ha_bridge import HABridge
from .history_store import HistoryStore
from .llm_client import LanguageModel
from .memory import trim_history
from .model_router import ModelRouter, TaskCategory
from .plugins import PluginSpec

logger = logging.getLogger(__name__)


@dataclass
class ConversationTurn:
    speaker: str
    text: str


class Assistant:
    """Coordinate wake-word, transcription, language model, and plugins."""

    def __init__(
        self,
        *,
        history_limit: int | None = None,
        plugins: Iterable[PluginSpec] | None = None,
        settings_obj: Settings | None = None,
        transcripts_dir: str | Path | None = None,
        user_id: str | None = None,
    ) -> None:
        self._settings = settings_obj or settings
        self._llm = LanguageModel(config=self._settings)
        self._history: list[ConversationTurn] = []
        self._history_limit = history_limit or self._settings.max_memory_items
        self._plugins = list(plugins or [])
        self._transcripts_dir = Path(transcripts_dir or self._settings.transcripts_dir)

        # Prefer explicit user_id, then settings.user_id, then "default"
        self._user_id = user_id or getattr(self._settings, "user_id", None) or "default"

        # Conversation history persistence
        self._history_store: HistoryStore | None = None
        self._prune_timer: threading.Timer | None = None
        if getattr(self._settings, "persist_history", True):
            try:
                db_path = getattr(self._settings, "history_db_path", None)
                if db_path is None:
                    from pathlib import Path as _Path

                    db_path = _Path("data/history.db")
                self._history_store = HistoryStore(db_path=db_path)
                # Preload the last 50 turns into in-memory history
                stored = self._history_store.load_history(self._user_id, limit=50)
                self._history = [
                    ConversationTurn(speaker=row["role"], text=row["content"]) for row in stored
                ]
                # Run an initial prune at startup and schedule daily repeats
                self._schedule_daily_prune()
            except Exception as exc:
                logger.warning("Failed to initialize HistoryStore: %s", exc)
                self._history_store = None

        # Follow-up engine for natural conversation cues
        self._followup_engine: object | None = None
        self._pending_followup: str | None = None
        # Lock protects the one-shot followup injection across concurrent generate_reply calls
        self._followup_lock = asyncio.Lock()
        self._init_followup_engine()

        # Model router for task-category-based model selection.
        # Pass routing config so the router can probe Ollama availability at init.
        _routing_cfg = getattr(self._settings, "model_routing", None)
        _ollama_url: str = getattr(
            self._settings, "ollama_base_url", ModelRouter._DEFAULT_OLLAMA_URL
        )
        self._router = ModelRouter(
            ollama_base_url=_ollama_url,
            routing_config=_routing_cfg,
        )

        # Route all tool calls through OpenClaw ToolBridge (US-P7-008)
        from .openclaw.tool_bridge import ToolBridge

        self._tool_router_fn = ToolBridge().route_if_tool_request

        # Skill trainer and registry for natural language skill creation (US-SK-003)
        from .skills.registry import SkillRegistry
        from .skills.trainer import SkillTrainer

        _skills_path = getattr(self._settings, "skills_path", None)
        self._skill_registry = SkillRegistry(skills_path=_skills_path)
        self._skill_trainer = SkillTrainer()

        # Only create HABridge if HA is configured
        self._ha_bridge: HABridge | None = None
        if self._settings.ha_base_url and self._settings.ha_token:
            try:
                self._ha_bridge = HABridge()
                logger.info("Home Assistant bridge initialized")
            except Exception as exc:
                logger.warning("Failed to initialize Home Assistant bridge: %s", exc)
                self._ha_bridge = None

    def _schedule_daily_prune(self) -> None:
        """Prune old history turns and schedule the next prune in 24 hours.

        Runs once immediately at startup, then repeats daily via a daemon thread.
        Safe to call if ``_history_store`` is None (no-op).
        """
        if self._history_store is None:
            return
        retention_days = int(getattr(self._settings, "history_retention_days", 30))
        try:
            deleted = self._history_store.prune(self._user_id, keep_days=retention_days)
            if deleted:
                logger.debug(
                    "Pruned %d old history turns for user %s (retention=%d days)",
                    deleted,
                    self._user_id,
                    retention_days,
                )
        except Exception as exc:
            logger.warning("History prune failed: %s", exc)
        # Schedule next prune in 24 hours (daemon so it doesn't block process exit)
        timer = threading.Timer(86400, self._schedule_daily_prune)
        timer.daemon = True
        timer.start()
        self._prune_timer = timer

    def _followups_enabled(self) -> bool:
        """
        Best-effort check for whether follow-ups are enabled.

        Supports multiple possible Settings layouts without hard dependency:
        - settings.followups_enabled (legacy)
        - settings.conversation.followups.enabled (newer config-backed)
        """
        # Legacy direct flag
        legacy = getattr(self._settings, "followups_enabled", None)
        if isinstance(legacy, bool):
            return legacy

        # Common nested config patterns
        conv = getattr(self._settings, "conversation", None)
        if isinstance(conv, dict):
            followups = conv.get("followups")
            if isinstance(followups, dict):
                enabled = followups.get("enabled")
                if isinstance(enabled, bool):
                    return enabled

        # Some Settings objects may expose followups as dict directly
        fu = getattr(self._settings, "followups", None)
        if isinstance(fu, dict):
            enabled = fu.get("enabled")
            if isinstance(enabled, bool):
                return enabled

        # Safe default for v1
        return False

    def _init_followup_engine(self) -> None:
        """Initialize the follow-up engine for natural cue injection."""
        if not self._followups_enabled():
            self._followup_engine = None
            self._pending_followup = None
            return

        # Preferred path: singleton/getter engine API
        try:
            from .followup_engine import get_followup_engine

            engine = get_followup_engine()
            self._followup_engine = engine

            # Start session if supported
            if hasattr(engine, "start_session"):
                engine.start_session(self._user_id)

            # Fetch a single pending follow-up prompt if supported
            if hasattr(engine, "get_followup_prompt"):
                self._pending_followup = engine.get_followup_prompt(self._user_id)

            if self._pending_followup:
                logger.debug("Pending followup for session: %s", self._pending_followup)
            return
        except Exception as exc:
            logger.debug("Follow-up engine getter not available: %s", exc)

        # Fallback path: construct engine directly from settings (older API)
        try:
            try:
                calendar_service = get_calendar_service()
            except Exception as exc:  # pragma: no cover - defensive guard
                logger.warning("Failed to initialize calendar service: %s", exc)
                calendar_service = None

            engine = FollowupEngine.from_settings(
                self._settings,
                calendar_service=calendar_service,
            )
            self._followup_engine = engine

            # If the fallback engine supports session and single prompt, use it
            if hasattr(engine, "start_session"):
                engine.start_session(self._user_id)
            if hasattr(engine, "get_followup_prompt"):
                self._pending_followup = engine.get_followup_prompt(self._user_id)

            if self._pending_followup:
                logger.debug("Pending followup for session: %s", self._pending_followup)
        except Exception as exc:
            logger.debug("Follow-up engine not available: %s", exc)
            self._followup_engine = None
            self._pending_followup = None

    @property
    def user_id(self) -> str:
        """Get the user ID for this assistant session."""
        return self._user_id

    @property
    def has_pending_followup(self) -> bool:
        """Check if there's a pending follow-up for this session."""
        return self._pending_followup is not None

    @property
    def pending_followup_prompt(self) -> str | None:
        """Get the pending follow-up prompt if any."""
        return self._pending_followup

    def _get_followup_lock(self) -> asyncio.Lock:
        """Return the follow-up lock, creating it lazily when needed."""
        lock = getattr(self, "_followup_lock", None)
        if lock is None:
            lock = asyncio.Lock()
            self._followup_lock = lock
        return lock

    async def _prepare_prompt(
        self,
        transcript: str,
        *,
        voice_mode: bool = False,
        active_user_id: str | None = None,
    ) -> str:
        if not transcript.strip():
            raise ValueError("Transcript must not be empty")

        prompt = self._build_prompt(
            transcript, voice_mode=voice_mode, active_user_id=active_user_id
        )

        async with self._get_followup_lock():
            if self._pending_followup:
                followup_hint = (
                    f'\n[Note: You may want to ask the user: "{self._pending_followup}" '
                    "as a natural conversation starter.]"
                )
                prompt = prompt + followup_hint
                self._pending_followup = None
                engine = self._followup_engine
                if engine and hasattr(engine, "mark_current_cue_asked"):
                    try:
                        engine.mark_current_cue_asked(self._user_id)
                    except Exception as exc:
                        logger.debug("mark_current_cue_asked failed: %s", exc)

        return prompt

    async def _post_process_completion(self, transcript: str, completion: str) -> str:
        loop = asyncio.get_running_loop()

        completion = await loop.run_in_executor(
            None,
            self._tool_router_fn,
            completion,
            self._build_tool_context(),
            self._build_tool_model_call(transcript),
        )

        plugin_enrichments = await self._run_plugins(transcript)
        if plugin_enrichments:
            completion = f"{completion}\n\nAdditional info:\n" + "\n".join(plugin_enrichments)

        if self._ha_bridge and self._ha_bridge.enabled:
            completion = await loop.run_in_executor(
                None,
                self._ha_bridge.post_process_response,
                completion,
            )

        return completion

    def _record_completion(self, transcript: str, completion: str) -> None:
        now = datetime.utcnow()
        history_store = getattr(self, "_history_store", None)
        if history_store is not None:
            try:
                history_store.save_turn(self._user_id, "user", transcript, now)
                history_store.save_turn(self._user_id, "assistant", completion, now)
            except Exception as exc:
                logger.warning("Failed to persist conversation turn: %s", exc)

        self._history.append(ConversationTurn("user", transcript))
        self._history.append(ConversationTurn("assistant", completion))
        self._history = [
            ConversationTurn(**item) if isinstance(item, dict) else item
            for item in trim_history(self._history, limit=self._history_limit)  # type: ignore[arg-type]
        ]

        self._log_turn(transcript, completion)

    def _resolve_model(self, category: TaskCategory) -> str:
        """Resolve the LLM model identifier for the given task category.

        Looks up *category* in ``settings.model_routing``.  If the category
        has no configured model, falls back to ``model_routing.default``.
        Returns an empty string when no routing overrides are configured,
        meaning the global ``llm_model`` is used unchanged.
        """
        routing = getattr(self._settings, "model_routing", None)
        if routing is None:
            return ""
        model: str = getattr(routing, str(category), "") or ""
        if model:
            return model
        default: str = getattr(routing, "default", "") or ""
        if default:
            logger.warning(
                "model_router: no model configured for category %r, falling back to default %r",
                str(category),
                default,
            )
        return default

    async def stream_reply(
        self, transcript: str, *, voice_mode: bool = False
    ) -> AsyncIterator[str]:
        loop = asyncio.get_running_loop()
        completion: str | None = None

        if self._ha_bridge and self._ha_bridge.enabled:
            completion = await loop.run_in_executor(
                None,
                self._ha_bridge.process_transcript,
                transcript,
            )

        if completion is not None:
            completion = await self._post_process_completion(transcript, completion)
            self._record_completion(transcript, completion)
            yield completion
            return

        prompt = await self._prepare_prompt(transcript, voice_mode=voice_mode)

        try:
            token_iterator = self._llm.stream(prompt)
        except NotImplementedError:
            completion = await loop.run_in_executor(None, self._llm.generate, prompt)
            completion = await self._post_process_completion(transcript, completion)
            self._record_completion(transcript, completion)
            yield completion
            return

        queue: asyncio.Queue[object] = asyncio.Queue()
        sentinel = object()
        collected_tokens: list[str] = []

        def _pump_tokens() -> None:
            try:
                for token in token_iterator:
                    if token:
                        loop.call_soon_threadsafe(queue.put_nowait, token)
            except Exception as exc:
                loop.call_soon_threadsafe(queue.put_nowait, exc)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, sentinel)

        pump_task = asyncio.create_task(asyncio.to_thread(_pump_tokens))
        try:
            while True:
                item = await queue.get()
                if item is sentinel:
                    break
                if isinstance(item, Exception):
                    raise item
                token = str(item)
                collected_tokens.append(token)
                yield token
        finally:
            await pump_task

        completion = "".join(collected_tokens).strip() or "(silence)"
        completion = await self._post_process_completion(transcript, completion)
        self._record_completion(transcript, completion)

    async def generate_reply(
        self,
        transcript: str,
        *,
        voice_mode: bool = False,
        active_user_id: str | None = None,
    ) -> str:
        loop = asyncio.get_running_loop()
        completion: str | None = None

        # Model routing: classify the message and resolve the target model.
        category = self._router.classify(transcript)
        _routing_cfg = getattr(self._settings, "model_routing", None)
        resolved_model = self._router.resolve_model(category, _routing_cfg)
        prev_model: str | None = getattr(self._llm, "model_name", None)
        if resolved_model and resolved_model != prev_model:
            logger.debug("model_router: classified as %s, using %s", category, resolved_model)
            if hasattr(self._llm, "model_name"):
                self._llm.model_name = resolved_model
        else:
            logger.debug(
                "model_router: classified as %s, using %s",
                category,
                prev_model or "default",
            )

        # Skill training: intercept natural language skill creation requests
        # before routing to the LLM (US-SK-003).
        _skill_trainer = getattr(self, "_skill_trainer", None)
        _skill_registry = getattr(self, "_skill_registry", None)
        if _skill_trainer is not None and _skill_registry is not None:
            training_response = _skill_trainer.handle_if_training_request(
                transcript, _skill_registry
            )
            if training_response is not None:
                self._record_completion(transcript, training_response)
                return training_response

        # Per-user credential/history scoping: swap self._user_id for the
        # duration of this call so history, transcripts, and tool calls use
        # the identified user's context.  Restore unconditionally in finally.
        prev_user_id = self._user_id
        if active_user_id is not None:
            self._user_id = active_user_id
            logger.debug("voice_identity: switching active user to %r", active_user_id)

        try:
            if self._ha_bridge and self._ha_bridge.enabled:
                completion = await loop.run_in_executor(
                    None,
                    self._ha_bridge.process_transcript,
                    transcript,
                )

            if completion is None:
                prompt = await self._prepare_prompt(
                    transcript, voice_mode=voice_mode, active_user_id=active_user_id
                )

                completion = await loop.run_in_executor(None, self._llm.generate, prompt)
                completion = await self._post_process_completion(transcript, completion)
        finally:
            # Restore the previous model name and user ID after this call.
            if prev_model is not None and hasattr(self._llm, "model_name"):
                self._llm.model_name = prev_model
            self._user_id = prev_user_id

        self._record_completion(transcript, completion)
        return completion

    async def _run_plugins(self, transcript: str) -> list[str]:
        loop = asyncio.get_running_loop()
        results: list[str] = []
        for spec in self._plugins:
            try:
                result = await loop.run_in_executor(None, spec.plugin.process, transcript)
            except Exception as exc:  # pragma: no cover - defensive guard
                logger.warning("Plugin %s failed: %s", spec.name, exc)
                continue
            if result:
                results.append(str(result))
        return results

    _VOICE_CONCISE_INSTRUCTION = (
        "[Respond in 1-3 sentences. Keep your reply short and conversational for voice output.]"
    )

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
        "the time_now tool. Do NOT guess or convert times yourself."
    )

    def _build_system_context(self) -> str:
        """Return a system context string with current date/time and user location."""

        _settings = getattr(self, "_settings", None)
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
            tz_name = "UTC"

        lines = [f"Current date and time: {now.strftime('%Y-%m-%d %H:%M')} {tz_name}"]

        location: str | None = getattr(_settings, "default_location", None)
        if not location:
            from rex.geolocation import get_cached_city

            location = get_cached_city()
        if location:
            lines.append(f"User location: {location}")

        lines.append("")
        lines.append(self._TOOL_INSTRUCTIONS)

        return "\n".join(lines)

    def _build_tool_context(self) -> dict[str, str]:
        """Return default_context dict for tool execution with location/timezone."""
        ctx: dict[str, str] = {}
        _settings = getattr(self, "_settings", None)

        location: str | None = getattr(_settings, "default_location", None)
        if not location:
            from rex.geolocation import get_cached_city

            location = get_cached_city()
        if location:
            ctx["location"] = location

        tz_name: str | None = getattr(_settings, "default_timezone", None)
        if not tz_name:
            from rex.geolocation import get_cached_timezone

            tz_name = get_cached_timezone()
        if tz_name:
            ctx["timezone"] = tz_name

        return ctx

    def _load_user_profile_context(self, user_id: str) -> str | None:
        """Load a user's memory profile and format it as a context string.

        Returns a short context string suitable for injection into the system
        prompt, or ``None`` if no profile is found.
        """
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

    def _build_prompt(
        self,
        transcript: str,
        *,
        voice_mode: bool = False,
        active_user_id: str | None = None,
    ) -> str:
        system_context = self._build_system_context()
        history_lines = [system_context]
        if active_user_id is not None:
            user_ctx = self._load_user_profile_context(active_user_id)
            if user_ctx:
                history_lines.append(user_ctx)
            else:
                history_lines.append(f"[Active user: {active_user_id}]")
        history_lines += [f"{turn.speaker}: {turn.text}" for turn in self._history[-4:]]
        history_lines.append(f"user: {transcript}")
        if voice_mode:
            history_lines.append(self._VOICE_CONCISE_INSTRUCTION)

        # Optional: bulk followup formatting if engine supports it
        engine = self._followup_engine
        if engine and hasattr(engine, "format_followups"):
            try:
                followups = engine.format_followups()
                if followups:
                    history_lines.append(str(followups))
            except Exception as exc:
                logger.debug("format_followups failed: %s", exc)

        history_lines.append("assistant:")
        return "\n".join(history_lines)

    def _build_tool_model_call(self, transcript: str):
        base_messages = [
            {"role": turn.speaker, "content": turn.text} for turn in self._history[-4:]
        ]
        base_messages.append({"role": "user", "content": transcript})

        def model_call(tool_message: dict[str, str]) -> str:
            messages = base_messages + [tool_message]
            return self._llm.generate(messages=messages)

        return model_call

    def history(self) -> list[ConversationTurn]:
        return list(self._history)

    def _log_turn(self, transcript: str, reply: str) -> None:
        try:
            self._transcripts_dir.mkdir(parents=True, exist_ok=True)
            user_dir = self._transcripts_dir / self._user_id
            user_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.utcnow()
            file_path = user_dir / f"{timestamp:%Y-%m-%d}.txt"
            with file_path.open("a", encoding="utf-8") as handle:
                handle.write(f"{timestamp:%H:%M:%S} user: {transcript.strip()}\n")
                handle.write(f"{timestamp:%H:%M:%S} assistant: {reply.strip()}\n\n")
        except Exception:  # pragma: no cover - logging must not break replies
            logger.exception("Failed to persist transcript entry")


__all__ = ["Assistant", "ConversationTurn", "PluginSpec"]
