"""Async assistant orchestration."""

from __future__ import annotations

import asyncio
import logging
import re
import threading
import time
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from .assistant_errors import IntegrationNotConfiguredError
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

# Matches bare "undo" or "undo that" utterances for HA command reversal
_UNDO_PATTERN = re.compile(r"^\s*undo\s*(?:that)?\s*$", re.IGNORECASE)
_DIRECT_TIME_PATTERNS = (
    re.compile(r"\bwhat\s+time\s+is\s+it\b", re.IGNORECASE),
    re.compile(r"\bwhat(?:'s| is)\s+(?:the\s+)?time\b", re.IGNORECASE),
    re.compile(r"\bcurrent\s+(?:local\s+)?time\b", re.IGNORECASE),
    re.compile(r"\btime\s+(?:is\s+it\s+)?(?:now|currently)\b", re.IGNORECASE),
)
_DIRECT_DATE_PATTERNS = (
    re.compile(r"\bwhat(?:'s|s| is)\s+(?:today'?s\s+|todays\s+|the\s+)?date\b", re.IGNORECASE),
    re.compile(r"\bwhat\s+date\s+is\s+(?:it\s+)?(?:today)?\b", re.IGNORECASE),
    re.compile(r"\bcurrent\s+date\b", re.IGNORECASE),
    re.compile(r"\bdate\s+today\b", re.IGNORECASE),
    re.compile(r"\b(?:today'?s|todays)\s+date\b", re.IGNORECASE),
)
_DIRECT_DAY_PATTERNS = (
    re.compile(r"\bwhat\s+day\s+is\s+it(?:\s+today)?\b", re.IGNORECASE),
    re.compile(r"\bwhat\s+day\s+is\s+today\b", re.IGNORECASE),
    re.compile(r"\bwhat(?:'s|s| is)\s+(?:the\s+)?day(?:\s+today)?\b", re.IGNORECASE),
    re.compile(r"\bday\s+of\s+(?:the\s+)?week\b", re.IGNORECASE),
)
_TIME_LOCATION_PATTERN = re.compile(r"\bin\s+([^?.!]+)[?.!]*\s*$", re.IGNORECASE)
_TIME_LOCATION_SUFFIXES = (
    "right now",
    "now",
    "today",
    "currently",
    "please",
    "for me",
    "at the moment",
)
_DIRECT_GREETING_PATTERN = re.compile(r"^\s*(?:hello|hey)\s*[!.?]*\s*$", re.IGNORECASE)
_DIRECT_WELLBEING_PATTERN = re.compile(
    r"^\s*(?:how\s+are\s+you|how'?s\s+it\s+going|how\s+are\s+things)\s*[?.!]*\s*$",
    re.IGNORECASE,
)
_DIRECT_CREATOR_PATTERN = re.compile(
    r"^\s*who\s+(?:created|made|built)\s+you\s*[?.!]*\s*$",
    re.IGNORECASE,
)
_RECIPE_REQUEST_PATTERN = re.compile(
    r"\b(?:need|want|give\s+me|find\s+me|show\s+me|make|bake|cook|how\s+(?:do\s+i|to))\b"
    r".*\b(?:recipe|make|bake|cook)\b",
    re.IGNORECASE,
)
_CHOCOLATE_CAKE_PATTERN = re.compile(r"\bchocolate\s+cake\b", re.IGNORECASE)
_SHOPPING_LIST_REFERENCE_PATTERN = re.compile(r"\b(?:shopping\s+)?list\b", re.IGNORECASE)
_UNVERIFIED_ACTION_CLAIM_PATTERNS = (
    re.compile(
        r"\bi(?:'ve| have| just)?\s+"
        r"(?:added|put|saved|sent|scheduled|set|changed|updated|deleted|removed)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:done|okay|sure)[,.\s]+(?:i(?:'ve| have| just)\s+)?"
        r"(?:added|put|saved|sent|scheduled|set|changed|updated|deleted|removed)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:it|that|this)\s+(?:has\s+been|was)\s+"
        r"(?:added|put|saved|sent|scheduled|set|changed|updated|deleted|removed)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:i(?:'ve| have| just)|done|okay|sure)[^.!?\n]{0,80}"
        r"\bcreated\s+(?:an?\s+)?(?:event|task|reminder|calendar|file|note)\b",
        re.IGNORECASE,
    ),
)
_EXPLICIT_MUTATION_REQUEST_PATTERN = re.compile(
    r"\b(?:add|put|save|send|create|schedule|set|change|update|delete|remove)\b",
    re.IGNORECASE,
)
_INTERNAL_TOOL_SYNTAX_PATTERN = re.compile(r"\bTOOL_(?:REQUEST|RESULT)\s*:", re.IGNORECASE)
_TOOL_REQUEST_PREFIX = "TOOL_REQUEST:"
_ACTION_CLAIM_STREAM_PREFIXES = (
    "i",
    "i'",
    "i have",
    "i've",
    "i just",
    "i added",
    "done",
    "okay",
    "sure",
    "it",
    "that",
    "this",
)


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

        # Skill trainer, registry, and router (US-SK-003, US-SK-004)
        from .skills.registry import SkillRegistry
        from .skills.router import SkillRouter
        from .skills.trainer import SkillTrainer

        _skills_path = getattr(self._settings, "skills_path", None)
        self._skill_registry = SkillRegistry(skills_path=_skills_path)
        self._skill_trainer = SkillTrainer()
        self._skill_router = SkillRouter(self._skill_registry)

        # Auto tool dispatcher (US-TD-002)
        from .tools.dispatcher import ToolDispatcher
        from .tools.registry import get_default_registry

        self._tool_dispatcher = ToolDispatcher(get_default_registry(), config=self._settings)

        # Shopping list voice handler (US-SL-002)
        from .shopping_list import ShoppingList
        from .shopping_list_handler import ShoppingListHandler

        _sl_path = getattr(self._settings, "shopping_list_path", None)
        _shopping_list = ShoppingList(path=_sl_path) if _sl_path else ShoppingList()
        self._shopping_list_handler: ShoppingListHandler | None = ShoppingListHandler(
            _shopping_list
        )

        # Music Assistant voice handler (US-022)
        from .integrations.music_assistant import MusicAssistantClient
        from .music_handler import MusicHandler

        _ma_url = getattr(self._settings, "music_assistant_url", None)
        _ma_token = getattr(self._settings, "music_assistant_token", None)
        _ma_client = MusicAssistantClient(base_url=_ma_url, token=_ma_token)
        self._music_handler: MusicHandler | None = MusicHandler(_ma_client)

        # Device state query handler (US-028)
        from .ha.state_handler import DeviceStateHandler

        _ha_url = getattr(self._settings, "ha_base_url", None)
        _ha_token = getattr(self._settings, "ha_token", None)
        self._device_state_handler: DeviceStateHandler | None = DeviceStateHandler(
            base_url=_ha_url,
            token=_ha_token,
        )

        # Response cache for repeated factual queries (US-LAT-004)
        from .response_cache import ResponseCache

        _cache_ttl = float(getattr(self._settings, "response_cache_ttl", 300.0))
        self._response_cache: ResponseCache | None = (
            ResponseCache(ttl=_cache_ttl) if _cache_ttl > 0 else None
        )

        # Only create HABridge if HA is configured
        self._ha_bridge: HABridge | None = None
        if self._settings.ha_base_url and self._settings.ha_token:
            try:
                self._ha_bridge = HABridge()
                logger.info("Home Assistant bridge initialized")
            except Exception as exc:
                logger.warning("Failed to initialize Home Assistant bridge: %s", exc)
                self._ha_bridge = None

        # Tool result post-processing handler (US-013)
        from .tools.result_handler import ToolResultHandler

        self._result_handler = ToolResultHandler(
            tool_router_fn=self._tool_router_fn,
            ha_bridge=self._ha_bridge,
        )

        # Proactive suggestion engine (US-036)
        from .suggestions.engine import SuggestionEngine
        from .suggestions.pattern_detector import PatternEntry

        _dismissed_path = getattr(self._settings, "dismissed_suggestions_path", None)
        _automations_path = getattr(self._settings, "automations_path", None)
        self._suggestion_engine: SuggestionEngine | None = SuggestionEngine(
            dismissed_path=_dismissed_path,
            automations_path=_automations_path,
        )
        # In-memory command log for pattern detection (wall-clock timestamps)
        self._pattern_entries: list[PatternEntry] = []

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
            except IntegrationNotConfiguredError:
                logger.info("Calendar: not configured")
                calendar_service = None
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

    async def _prepare_model_input(
        self,
        transcript: str,
        *,
        voice_mode: bool = False,
        active_user_id: str | None = None,
        tool_context: str | None = None,
    ) -> tuple[str, list[dict[str, str]]]:
        if not transcript.strip():
            raise ValueError("Transcript must not be empty")

        prompt = self._build_prompt(
            transcript,
            voice_mode=voice_mode,
            active_user_id=active_user_id,
            tool_context=tool_context,
        )
        messages = self._build_messages(
            transcript,
            voice_mode=voice_mode,
            active_user_id=active_user_id,
            tool_context=tool_context,
        )

        async with self._get_followup_lock():
            if self._pending_followup:
                followup_text = (
                    f'You may want to ask the user: "{self._pending_followup}" '
                    "as a natural conversation starter."
                )
                followup_hint = f"\n[Note: {followup_text}]"
                prompt = prompt + followup_hint
                messages.insert(-1, {"role": "system", "content": followup_text})
                self._pending_followup = None
                engine = self._followup_engine
                if engine and hasattr(engine, "mark_current_cue_asked"):
                    try:
                        engine.mark_current_cue_asked(self._user_id)
                    except Exception as exc:
                        logger.debug("mark_current_cue_asked failed: %s", exc)

        return prompt, messages

    async def _prepare_prompt(
        self,
        transcript: str,
        *,
        voice_mode: bool = False,
        active_user_id: str | None = None,
        tool_context: str | None = None,
    ) -> str:
        prompt, _messages = await self._prepare_model_input(
            transcript,
            voice_mode=voice_mode,
            active_user_id=active_user_id,
            tool_context=tool_context,
        )
        return prompt

    def _generate_model_reply(self, prompt: str, messages: list[dict[str, str]]) -> str:
        try:
            return self._llm.generate(messages=messages)
        except TypeError:
            return self._llm.generate(prompt)

    def _stream_model_reply(self, prompt: str, messages: list[dict[str, str]]) -> Iterable[str]:
        try:
            return self._llm.stream(messages=messages)
        except TypeError:
            return self._llm.stream(prompt)

    async def _post_process_completion(self, transcript: str, completion: str) -> str:
        plugin_enrichments = await self._run_plugins(transcript)
        return await self._result_handler.process(
            transcript,
            completion,
            tool_context=self._build_tool_context(),
            model_call_fn=self._build_tool_model_call(transcript),
            plugin_enrichments=plugin_enrichments,
        )

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

    def _try_direct_time_reply(self, transcript: str) -> str | None:
        """Answer simple clock/date queries without an LLM round trip."""
        text = transcript.strip()
        if not text:
            return None

        wants_time = any(pattern.search(text) for pattern in _DIRECT_TIME_PATTERNS)
        wants_date = any(pattern.search(text) for pattern in _DIRECT_DATE_PATTERNS)
        wants_day = any(pattern.search(text) for pattern in _DIRECT_DAY_PATTERNS)
        if not wants_time and not wants_date and not wants_day:
            return None

        context = self._build_tool_context()
        location = self._extract_direct_time_location(text) or context.get("location")
        args = {"location": location} if location else {}

        try:
            from .openclaw.tool_executor import execute_tool

            result = execute_tool(
                {"tool": "time_now", "args": args},
                context,
                skip_policy_check=True,
                skip_credential_check=True,
                skip_audit_log=True,
            )
        except Exception as exc:
            logger.debug("direct time reply failed: %s", exc)
            return None

        if "error" in result:
            fallback = self._fallback_local_time_result(location, context)
            if fallback is None:
                logger.debug("direct time reply returned error: %s", result["error"])
                return None
            result = fallback

        return self._format_direct_time_reply(
            result,
            location=location,
            wants_date=wants_date and not wants_time,
            wants_day=wants_day and not wants_time,
        )

    def _fallback_local_time_result(
        self,
        location: str | None,
        context: dict[str, str],
    ) -> dict[str, object] | None:
        configured_location = context.get("location")
        if location and configured_location:
            same_location = location.strip().casefold() == configured_location.strip().casefold()
            if not same_location:
                return None

        try:
            now = datetime.now().astimezone()
        except Exception as exc:
            logger.debug("local clock fallback failed: %s", exc)
            return None

        timezone = str(now.tzinfo) if now.tzinfo is not None else context.get("timezone", "local")
        return {
            "local_time": now.strftime("%Y-%m-%d %H:%M"),
            "date": now.strftime("%Y-%m-%d"),
            "timezone": timezone,
        }

    def _extract_direct_time_location(self, transcript: str) -> str | None:
        match = _TIME_LOCATION_PATTERN.search(transcript)
        if not match:
            return None
        location = match.group(1).strip(" \t,")
        while True:
            updated = location
            for suffix in _TIME_LOCATION_SUFFIXES:
                suffix_text = f" {suffix}"
                if updated.lower().endswith(suffix_text):
                    updated = updated[: -len(suffix_text)].strip(" \t,")
                    break
            if updated == location:
                break
            location = updated
        if not location:
            return None
        if location.lower().split(maxsplit=1)[0] in {"my", "your", "the", "a", "an"}:
            return None
        return location

    def _format_direct_time_reply(
        self,
        result: dict[str, object],
        *,
        location: str | None,
        wants_date: bool,
        wants_day: bool,
    ) -> str:
        local_time = str(result.get("local_time") or "")
        try:
            when = datetime.strptime(local_time, "%Y-%m-%d %H:%M")
        except ValueError:
            return str(result.get("local_time") or result.get("date") or "")

        place = f" in {location}" if location else ""
        if wants_day:
            date_text = f"{when.strftime('%B')} {when.day}, {when.year}"
            return f"Today is {when.strftime('%A')}, {date_text}{place}."

        if wants_date:
            date_text = f"{when.strftime('%B')} {when.day}, {when.year}"
            return f"Today is {date_text}{place}."

        time_text = when.strftime("%I:%M %p").lstrip("0")
        return f"It's {time_text}{place}."

    def _try_direct_conversation_reply(self, transcript: str) -> str | None:
        """Handle common greetings without invoking an unstable chat model."""
        text = transcript.strip()
        if not text:
            return None
        if _DIRECT_GREETING_PATTERN.match(text):
            return "Hello. How can I help?"
        if _DIRECT_WELLBEING_PATTERN.match(text):
            return "I'm here and ready to help."
        if _DIRECT_CREATOR_PATTERN.match(text):
            return (
                "I'm AskRex, a local assistant running from this project. "
                "The repo owner and project contributors configure the models and integrations I use."
            )
        return None

    def _try_direct_recipe_reply(self, transcript: str) -> str | None:
        """Handle common recipe requests without tool or shopping-list routing."""
        text = transcript.strip()
        if not text:
            return None
        if _SHOPPING_LIST_REFERENCE_PATTERN.search(text):
            return None
        if not _RECIPE_REQUEST_PATTERN.search(text):
            return None
        if not _CHOCOLATE_CAKE_PATTERN.search(text):
            return None
        return (
            "Here is a simple chocolate cake recipe: mix 1 and 3/4 cups flour, "
            "2 cups sugar, 3/4 cup cocoa, 1 and 1/2 teaspoons baking powder, "
            "1 and 1/2 teaspoons baking soda, and 1 teaspoon salt. Add 2 eggs, "
            "1 cup milk, 1/2 cup oil, and 2 teaspoons vanilla, then stir in "
            "1 cup hot water. Bake in two greased 9-inch pans at 350 F for "
            "30 to 35 minutes, cool, and frost."
        )

    def _looks_like_unverified_action_claim(self, completion: str) -> bool:
        return any(pattern.search(completion) for pattern in _UNVERIFIED_ACTION_CLAIM_PATTERNS)

    def _stream_tool_prefix_state(self, text: str) -> str:
        stripped = text.lstrip().upper()
        if not stripped:
            return "pending"
        prefix = _TOOL_REQUEST_PREFIX
        if prefix.startswith(stripped):
            return "pending"
        if stripped.startswith(prefix):
            return "pending"

        lowered = text.lstrip().lower()
        if self._looks_like_unverified_action_claim(text):
            return "pending"
        if len(lowered) < 80 and any(
            prefix.startswith(lowered) or lowered.startswith(prefix)
            for prefix in _ACTION_CLAIM_STREAM_PREFIXES
        ):
            return "pending"
        return "text"

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

        direct_reply = self._try_direct_time_reply(transcript)
        if direct_reply is not None:
            self._record_completion(transcript, direct_reply)
            yield direct_reply
            return

        direct_reply = self._try_direct_conversation_reply(transcript)
        if direct_reply is not None:
            self._record_completion(transcript, direct_reply)
            yield direct_reply
            return

        direct_reply = self._try_direct_recipe_reply(transcript)
        if direct_reply is not None:
            self._record_completion(transcript, direct_reply)
            yield direct_reply
            return

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

        prompt, messages = await self._prepare_model_input(transcript, voice_mode=voice_mode)

        try:
            token_iterator = self._stream_model_reply(prompt, messages)
        except NotImplementedError:
            completion = await loop.run_in_executor(
                None, self._generate_model_reply, prompt, messages
            )
            completion = await self._post_process_completion(transcript, completion)
            self._record_completion(transcript, completion)
            yield completion
            return

        queue: asyncio.Queue[object] = asyncio.Queue()
        sentinel = object()
        collected_tokens: list[str] = []
        pending_stream_text = ""
        stream_released = False

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
                if stream_released:
                    yield token
                    continue

                pending_stream_text += token
                prefix_state = self._stream_tool_prefix_state(pending_stream_text)
                if prefix_state == "text":
                    stream_released = True
                    yield pending_stream_text
                    pending_stream_text = ""
        finally:
            await pump_task

        completion = "".join(collected_tokens).strip() or "(silence)"
        completion = await self._post_process_completion(transcript, completion)
        self._record_completion(transcript, completion)
        if not stream_released:
            yield completion

    async def generate_reply(
        self,
        transcript: str,
        *,
        voice_mode: bool = False,
        active_user_id: str | None = None,
    ) -> str:
        loop = asyncio.get_running_loop()
        completion: str | None = None

        # Capability query: "What can you do?" (US-038) — intercept before any
        # model routing or tool dispatch so the answer is always fast and accurate.
        from .capabilities.registry import get_capability_registry
        from .capabilities.responder import build_capability_response, is_capability_query

        if is_capability_query(transcript):
            _registry = get_capability_registry(config=self._settings)
            _cap_reply = build_capability_response(_registry)
            self._record_completion(transcript, _cap_reply)
            return _cap_reply

        direct_reply = self._try_direct_time_reply(transcript)
        if direct_reply is not None:
            self._record_completion(transcript, direct_reply)
            return direct_reply

        if active_user_id is None:
            direct_reply = self._try_direct_conversation_reply(transcript)
            if direct_reply is not None:
                self._record_completion(transcript, direct_reply)
                return direct_reply

        direct_reply = self._try_direct_recipe_reply(transcript)
        if direct_reply is not None:
            self._record_completion(transcript, direct_reply)
            return direct_reply

        # Proactive suggestion response handling (US-036): intercept yes/no
        # answers while a suggestion is pending, before any other processing.
        _sug_engine = getattr(self, "_suggestion_engine", None)
        if _sug_engine is not None and _sug_engine.has_pending:
            if _sug_engine.is_accept(transcript):
                reply = str(_sug_engine.handle_yes())
                self._record_completion(transcript, reply)
                return reply
            elif _sug_engine.is_dismiss(transcript):
                reply = str(_sug_engine.handle_dismiss())
                self._record_completion(transcript, reply)
                return reply

        # Model routing: classify the message and resolve the target model.
        _router = getattr(self, "_router", None)
        category = _router.classify(transcript) if _router is not None else TaskCategory.default
        _routing_cfg = getattr(getattr(self, "_settings", None), "model_routing", None)
        resolved_model = (
            _router.resolve_model(category, _routing_cfg) if _router is not None else None
        )
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
                return str(training_response)

        # Skill invocation: check registered skill triggers before the LLM
        # (US-SK-004).  On a match the skill handler is executed and the
        # result returned directly; on no match normal routing proceeds.
        _skill_router = getattr(self, "_skill_router", None)
        if _skill_router is not None:
            matched_skill = _skill_router.match(transcript)
            if matched_skill is not None:
                skill_response = str(_skill_router.execute(matched_skill, transcript))
                self._record_completion(transcript, skill_response)
                return skill_response

        # Shopping list voice commands (US-SL-002)
        _sl_handler = getattr(self, "_shopping_list_handler", None)
        if _sl_handler is not None:
            _sl_response = _sl_handler.handle(transcript, user_id=active_user_id or self._user_id)
            if _sl_response is not None:
                sl_response = str(_sl_response)
                self._record_completion(transcript, sl_response)
                return sl_response

        # Music Assistant voice commands (US-022)
        _music_handler = getattr(self, "_music_handler", None)
        if _music_handler is not None:
            _music_response = _music_handler.handle(transcript)
            if _music_response is not None:
                music_response = str(_music_response)
                self._record_completion(transcript, music_response)
                return music_response

        # Device state queries (US-028)
        _ds_handler = getattr(self, "_device_state_handler", None)
        if _ds_handler is not None:
            _ds_response = _ds_handler.handle(transcript)
            if _ds_response is not None:
                ds_response = str(_ds_response)
                self._record_completion(transcript, ds_response)
                return ds_response

        # Per-user credential/history scoping: swap self._user_id for the
        # duration of this call so history, transcripts, and tool calls use
        # the identified user's context.  Restore unconditionally in finally.
        prev_user_id = self._user_id
        if active_user_id is not None:
            self._user_id = active_user_id
            logger.debug("voice_identity: switching active user to %r", active_user_id)

        # Auto tool dispatch: select tools by intent, execute, aggregate context
        # (US-TD-002).  Uses getattr guard so __new__-constructed test instances
        # without _tool_dispatcher still work.
        _tool_context: str | None = None
        _dispatcher = getattr(self, "_tool_dispatcher", None)
        if _dispatcher is not None:
            _selected_tools = _dispatcher.select_tools(transcript)
            if _selected_tools:
                _effective_user = active_user_id or self._user_id
                import functools

                _tool_results = await loop.run_in_executor(
                    None,
                    functools.partial(
                        _dispatcher.execute_tools,
                        _selected_tools,
                        transcript,
                        user_id=_effective_user,
                    ),
                )
                _tool_context = _dispatcher.format_tool_context(_tool_results) or None

        # Check response cache before hitting the LLM.
        _cache = getattr(self, "_response_cache", None)
        _cached: str | None = _cache.get(transcript) if _cache is not None else None
        if _cached is not None:
            self._user_id = prev_user_id
            self._record_completion(transcript, _cached)
            return _cached

        try:
            if self._ha_bridge and self._ha_bridge.enabled:
                # Check for undo intent before other HA processing
                if _UNDO_PATTERN.match(transcript):
                    completion = await loop.run_in_executor(
                        None,
                        self._ha_bridge.undo_last,
                    )
                else:
                    _hist_len_before = len(getattr(self._ha_bridge, "_command_history", None) or [])
                    completion = await loop.run_in_executor(
                        None,
                        self._ha_bridge.process_transcript,
                        transcript,
                    )
                    # If a new HA command succeeded, record it for pattern detection
                    if completion is not None:
                        _cmd_hist = getattr(self._ha_bridge, "_command_history", None)
                        if _cmd_hist is not None and len(_cmd_hist) > _hist_len_before:
                            _last = _cmd_hist._entries[-1]
                            from .suggestions.pattern_detector import PatternEntry

                            self._pattern_entries.append(
                                PatternEntry(
                                    entity_id=_last.entity_id,
                                    service=_last.service,
                                    timestamp=time.time(),
                                )
                            )
                        # Check if a proactive suggestion is due (US-036)
                        _sug_engine2 = getattr(self, "_suggestion_engine", None)
                        if _sug_engine2 is not None:
                            from .suggestions.pattern_detector import detect_patterns

                            _patterns = detect_patterns(self._pattern_entries)
                            _sug = _sug_engine2.get_suggestion(_patterns)
                            if _sug is not None:
                                _, _spoken = _sug
                                completion = f"{completion} {_spoken}"

            if completion is None:
                prompt, messages = await self._prepare_model_input(
                    transcript,
                    voice_mode=voice_mode,
                    active_user_id=active_user_id,
                    tool_context=_tool_context,
                )

                completion = await loop.run_in_executor(
                    None, self._generate_model_reply, prompt, messages
                )
                completion = await self._post_process_completion(transcript, completion)
        finally:
            # Restore the previous model name and user ID after this call.
            if prev_model is not None and hasattr(self._llm, "model_name"):
                self._llm.model_name = prev_model
            self._user_id = prev_user_id

        # Store result in cache for future identical queries.
        if _cache is not None:
            _cache.put(transcript, completion)

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
        "the time_now tool. Do NOT guess or convert times yourself. Never claim "
        "you added, changed, sent, saved, or created something unless a tool result "
        "confirms that action succeeded."
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
            # ZoneInfo lookup failed (e.g. tzdata not installed); fall back to UTC
            # time but keep the configured timezone name so it appears in context.
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

    def _get_active_personality_prompt(self, active_user_id: str | None) -> str | None:
        """Return the system prompt for the user's configured personality, or None."""
        from .personality import get_personality

        # Resolve personality name: check per-user preferences first, then settings
        personality_name: str | None = None
        uid = active_user_id or getattr(self, "_user_id", None)
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
            personality_name = getattr(getattr(self, "_settings", None), "personality", None)

        if not personality_name:
            from .personality import DEFAULT_PERSONALITY

            personality_name = DEFAULT_PERSONALITY

        return get_personality(personality_name).system_prompt

    def _build_prompt(
        self,
        transcript: str,
        *,
        voice_mode: bool = False,
        active_user_id: str | None = None,
        tool_context: str | None = None,
    ) -> str:
        system_context = self._build_system_context()
        history_lines = [system_context]

        # Inject active personality system prompt
        personality_prompt = self._get_active_personality_prompt(active_user_id)
        if personality_prompt:
            history_lines.append(personality_prompt)
        if active_user_id is not None:
            user_ctx = self._load_user_profile_context(active_user_id)
            if user_ctx:
                history_lines.append(user_ctx)
            else:
                history_lines.append(f"[Active user: {active_user_id}]")
            # Inject per-user remembered facts so the LLM can reference them.
            try:
                from rex.user_facts import format_facts_for_prompt

                facts_ctx = format_facts_for_prompt(active_user_id)
                if facts_ctx:
                    history_lines.append(facts_ctx)
            except Exception as exc:
                logger.debug("Failed to load user facts: %s", exc)
        if tool_context:
            history_lines.append(tool_context)
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

    def _build_messages(
        self,
        transcript: str,
        *,
        voice_mode: bool = False,
        active_user_id: str | None = None,
        tool_context: str | None = None,
    ) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self._build_system_context()}
        ]

        personality_prompt = self._get_active_personality_prompt(active_user_id)
        if personality_prompt:
            messages.append({"role": "system", "content": personality_prompt})

        if active_user_id is not None:
            user_ctx = self._load_user_profile_context(active_user_id)
            messages.append(
                {
                    "role": "system",
                    "content": user_ctx if user_ctx else f"[Active user: {active_user_id}]",
                }
            )
            try:
                from rex.user_facts import format_facts_for_prompt

                facts_ctx = format_facts_for_prompt(active_user_id)
                if facts_ctx:
                    messages.append({"role": "system", "content": facts_ctx})
            except Exception as exc:
                logger.debug("Failed to load user facts: %s", exc)

        if tool_context:
            messages.append({"role": "system", "content": tool_context})

        engine = self._followup_engine
        if engine and hasattr(engine, "format_followups"):
            try:
                followups = engine.format_followups()
                if followups:
                    messages.append({"role": "system", "content": str(followups)})
            except Exception as exc:
                logger.debug("format_followups failed: %s", exc)

        if voice_mode:
            messages.append({"role": "system", "content": self._VOICE_CONCISE_INSTRUCTION})

        for turn in self._history[-4:]:
            speaker = str(turn.speaker).strip().lower()
            role = "assistant" if speaker in {"assistant", "rex"} else "user"
            messages.append({"role": role, "content": turn.text})

        messages.append({"role": "user", "content": transcript})
        return messages

    def _build_tool_model_call(self, transcript: str):
        base_messages = [
            {"role": "system", "content": self._build_system_context()},
            {
                "role": "system",
                "content": (
                    "A tool has already been executed for the user's request. "
                    "Use the tool result to answer the user directly in plain language. "
                    "Do not emit TOOL_REQUEST. "
                    "Do not output JSON. "
                    "Do not request another tool unless the tool result is missing or invalid."
                ),
            },
            *[{"role": turn.speaker, "content": turn.text} for turn in self._history[-4:]],
            {"role": "user", "content": transcript},
        ]

        def model_call(tool_message: dict[str, str]) -> str:
            messages = base_messages + [
                {
                    "role": "system",
                    "content": (
                        "The next message is the tool result. "
                        "Answer the user naturally using that result. "
                        "Do not emit TOOL_REQUEST. "
                        "Do not output JSON."
                    ),
                },
                tool_message,
            ]

            reply = self._llm.generate(messages=messages)

            if isinstance(reply, str) and reply.lstrip().startswith("TOOL_REQUEST:"):
                messages.extend(
                    [
                        {"role": "assistant", "content": reply},
                        {
                            "role": "system",
                            "content": (
                                "That was incorrect. "
                                "You already have the tool result. "
                                "Answer the user directly in one short natural sentence. "
                                "Do not emit TOOL_REQUEST. "
                                "Do not output JSON."
                            ),
                        },
                    ]
                )
                reply = self._llm.generate(messages=messages)

            return reply

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
