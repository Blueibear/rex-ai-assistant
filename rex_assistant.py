"""Command-line entry point for the Rex assistant.

The script keeps the lightweight interactive loop that originally lived at the
repository root while delegating the heavy lifting to the refactored
``rex.assistant`` helpers. Keeping the documentation here helps developers
understand how to launch the assistant without digging through historical
commits.
"""

import importlib
import importlib.util
import os
import tempfile
import textwrap
from typing import Optional, Iterable

import numpy as np
import sounddevice as sd
import simpleaudio as sa
import soundfile as sf
import whisper
from TTS.api import TTS

from config import load_config
from llm_client import LanguageModel
from wakeword_utils import detect_wakeword, load_wakeword_model
from memory_utils import (
    extract_voice_reference,
    load_all_profiles,
    load_users_map,
    resolve_user_key,
)

import asyncio
import logging

from rex import settings
from rex.assistant import Assistant
from rex.logging_utils import configure_logging
from rex.plugins import PluginSpec, load_plugins, shutdown_plugins

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CONFIG = load_config()
USERS_MAP = load_users_map()
USER_PROFILES = load_all_profiles()
ACTIVE_USER = resolve_user_key(CONFIG.default_user, USERS_MAP, profiles=USER_PROFILES)

if not ACTIVE_USER:
    if USER_PROFILES:
        ACTIVE_USER = sorted(USER_PROFILES.keys())[0]
    else:
        ACTIVE_USER = "james"

ACTIVE_PROFILE = USER_PROFILES.get(ACTIVE_USER, {})
ACTIVE_USER_DISPLAY = (
    ACTIVE_PROFILE.get("name") if isinstance(ACTIVE_PROFILE, dict) else None
)

# Wake word settings
WAKEWORD = CONFIG.wakeword
WAKEWORD_THRESHOLD = CONFIG.wakeword_threshold

# Path to wake confirmation sound
WAKE_SOUND_PATH = CONFIG.wake_sound_path or os.path.join(
    os.path.dirname(__file__), "assets", "rex_wake_acknowledgment (1).wav"
)

# Speaker reference WAVs
SPEAKER_VOICES = {
    user: extract_voice_reference(profile)
    for user, profile in USER_PROFILES.items()
}
if ACTIVE_USER not in SPEAKER_VOICES:
    SPEAKER_VOICES[ACTIVE_USER] = None

# Recording length
COMMAND_DURATION = CONFIG.command_duration

# Whisper model to load
WHISPER_MODEL_NAME = CONFIG.whisper_model

# LLM
LLM = LanguageModel(CONFIG)

ASSISTANT_PERSONA = textwrap.dedent(
    """
    You are Rex, a focused AI voice assistant that keeps responses concise.
    Reference the active user's preferences when it helps personalise your
    answer. Always respond in natural English prose.
    """
)

# Optional plugin: web search
_WEB_SEARCH_SPEC = importlib.util.find_spec("plugins.web_search")
if _WEB_SEARCH_SPEC is not None:
    search_web = getattr(importlib.import_module("plugins.web_search"), "search_web", None)
else:
    search_web = None


# ---------------------------------------------------------------------------
# Async CLI loop
# ---------------------------------------------------------------------------

async def _chat_loop(assistant: Assistant) -> None:
    print("Rex assistant ready. Type 'exit' or 'quit' to stop.")
    while True:
        try:
            user_input = input("You: ")
        except EOFError:
            break

        if user_input.strip().lower() in {"exit", "quit"}:
            break
        if not user_input.strip():
            print("(please enter a prompt)")
            continue

        try:
            reply = await assistant.generate_reply(user_input)
        except Exception as exc:  # pragma: no cover
            logger.exception("Assistant failed to generate a reply: %s", exc)
            print(f"[error] {exc}")
            continue

        print(f"Rex: {reply}")


async def _run() -> None:
    configure_logging()
    plugin_specs: Iterable[PluginSpec] = load_plugins()
    assistant = Assistant(history_limit=settings.max_memory_items, plugins=plugin_specs)
    try:
        await _chat_loop(assistant)
    finally:
        shutdown_plugins(plugin_specs)


def main() -> int:
    try:
        asyncio.run(_run())
    except KeyboardInterrupt:  # pragma: no cover
        print("\nInterrupted.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

