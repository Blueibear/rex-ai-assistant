"""Async assistant orchestration."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from .calendar_service import get_calendar_service
from .config import Settings, settings
from .followup_engine import FollowupEngine
from .ha_bridge import HABridge
from .llm_client import LanguageModel
from .memory import trim_history
from .plugins import PluginSpec
from .tool_router import route_if_tool_request

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

        # Follow-up engine for natural conversation cues
        self._followup_engine: Optional[object] = None
        self._pending_followup: Optional[str] = None
        self._followup_injected = False
        self._init_followup_engine()

        # Only create HABridge if HA is configured
        self._ha_bridge: HABridge | None = None
        if self._settings.ha_base_url and self._settings.ha_token:
            try:
                self._ha_bridge = HABridge()
                logger.info("Home Assistant bridge initialized")
            except Exception as exc:
                logger.warning("Failed to initialize Home Assistant bridge: %s", exc)
                self._ha_bridge = None

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
            from .followup_engine import get_followup_engine  # type: ignore

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

            engine = FollowupEngine.from_settings(  # type: ignore[attr-defined]
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
        return self._pending_followup is not None and not self._followup_injected

    @property
    def pending_followup_prompt(self) -> Optional[str]:
        """Get the pending follow-up prompt if any."""
        return self._pending_followup if self.has_pending_followup else None

    async def generate_reply(self, transcript: str) -> str:
        if not transcript.strip():
            raise ValueError("Transcript must not be empty")

        loop = asyncio.get_running_loop()
        completion: str | None = None

        if self._ha_bridge and self._ha_bridge.enabled:
            completion = await loop.run_in_executor(
                None,
                self._ha_bridge.process_transcript,
                transcript,
            )

        if completion is None:
            prompt = self._build_prompt(transcript)
            completion = await loop.run_in_executor(None, self._llm.generate, prompt)
            completion = await loop.run_in_executor(
                None,
                route_if_tool_request,
                completion,
                {},
                self._build_tool_model_call(transcript),
            )

            plugin_enrichments = await self._run_plugins(transcript)
            if plugin_enrichments:
                completion = (
                    f"{completion}\n\nAdditional info:\n" + "\n".join(plugin_enrichments)
                )

            if self._ha_bridge and self._ha_bridge.enabled:
                completion = await loop.run_in_executor(
                    None,
                    self._ha_bridge.post_process_response,
                    completion,
                )

        self._history.append(ConversationTurn("user", transcript))
        self._history.append(ConversationTurn("assistant", completion))
        self._history = [
            ConversationTurn(**item) if isinstance(item, dict) else item
            for item in trim_history(self._history, limit=self._history_limit)
        ]

        self._log_turn(transcript, completion)
        return completion

    async def _run_plugins(self, transcript: str) -> list[str]:
        loop = asyncio.get_running_loop()
        results: list[str] = []
        for spec in self._plugins:
            try:
                result = await loop.run_in_executor(
                    None, spec.plugin.process, transcript
                )
            except Exception as exc:  # pragma: no cover - defensive guard
                logger.warning("Plugin %s failed: %s", spec.name, exc)
                continue
            if result:
                results.append(str(result))
        return results

    def _build_prompt(self, transcript: str) -> str:
        history_lines = [f"{turn.speaker}: {turn.text}" for turn in self._history[-4:]]
        history_lines.append(f"user: {transcript}")

        # Optional: bulk followup formatting if engine supports it
        engine = self._followup_engine
        if engine and hasattr(engine, "format_followups"):
            try:
                followups = engine.format_followups()  # type: ignore[call-arg]
                if followups:
                    history_lines.append(str(followups))
            except Exception as exc:
                logger.debug("format_followups failed: %s", exc)

        history_lines.append("assistant:")
        prompt = "\n".join(history_lines)

        # Inject one follow-up cue on first exchange if available
        if self._pending_followup and not self._followup_injected:
            followup_hint = (
                f'\n[Note: You may want to ask the user: "{self._pending_followup}" '
                "as a natural conversation starter.]"
            )
            prompt = prompt + followup_hint
            self._followup_injected = True

            # Mark cue as asked if supported
            if engine and hasattr(engine, "mark_current_cue_asked"):
                try:
                    engine.mark_current_cue_asked(self._user_id)  # type: ignore[call-arg]
                except Exception as exc:
                    logger.debug("mark_current_cue_asked failed: %s", exc)

        return prompt

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

