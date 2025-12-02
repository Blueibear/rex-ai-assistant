"""Async assistant orchestration."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .config import Settings, settings
from .llm_client import LanguageModel
from .memory import trim_history
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
    ) -> None:
        self._settings = settings_obj or settings
        self._llm = LanguageModel(
            model_name=self._settings.llm_model,
            backend=self._settings.llm_backend,
            temperature=self._settings.temperature,
        )
        self._history: list[ConversationTurn] = []
        self._history_limit = history_limit or self._settings.max_memory_items
        self._plugins = list(plugins or [])
        self._transcripts_dir = Path(transcripts_dir or self._settings.transcripts_dir)

    async def generate_reply(self, transcript: str) -> str:
        if not transcript.strip():
            raise ValueError("Transcript must not be empty")

        loop = asyncio.get_running_loop()
        prompt = self._build_prompt(transcript)

        completion = await loop.run_in_executor(None, self._llm.generate, prompt)
        self._history.append(ConversationTurn("user", transcript))
        self._history.append(ConversationTurn("assistant", completion))
        self._history = [
            ConversationTurn(**item) if isinstance(item, dict) else item
            for item in trim_history(self._history, limit=self._history_limit)
        ]

        plugin_enrichments = await self._run_plugins(transcript)
        if plugin_enrichments:
            completion = f"{completion}\n\nAdditional info:\n" + "\n".join(plugin_enrichments)
        self._log_turn(transcript, completion)
        return completion

    async def _run_plugins(self, transcript: str) -> list[str]:
        loop = asyncio.get_running_loop()
        results = []
        for spec in self._plugins:
            try:
                result = await loop.run_in_executor(None, spec.plugin.process, transcript)
            except Exception as exc:  # pragma: no cover - defensive guard
                logger.warning("Plugin %s failed: %s", spec.name, exc)
                continue
            if result:
                results.append(str(result))
        return results

    def _build_prompt(self, transcript: str) -> str:
        history_lines = [f"{turn.speaker}: {turn.text}" for turn in self._history[-4:]]
        history_lines.append(f"user: {transcript}")
        history_lines.append("assistant:")
        return "\n".join(history_lines)

    def history(self) -> list[ConversationTurn]:
        return list(self._history)

    def _log_turn(self, transcript: str, reply: str) -> None:
        try:
            self._transcripts_dir.mkdir(parents=True, exist_ok=True)
            user_dir = self._transcripts_dir / self._settings.user_id
            user_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.utcnow()
            file_path = user_dir / f"{timestamp:%Y-%m-%d}.txt"
            with file_path.open("a", encoding="utf-8") as handle:
                handle.write(f"{timestamp:%H:%M:%S} user: {transcript.strip()}\n")
                handle.write(f"{timestamp:%H:%M:%S} assistant: {reply.strip()}\n\n")
        except Exception:  # pragma: no cover - logging must not break replies
            logger.exception("Failed to persist transcript entry")


__all__ = ["Assistant", "ConversationTurn", "PluginSpec"]
