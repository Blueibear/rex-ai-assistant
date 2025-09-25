"""Command-line entry point for the Rex assistant.

The script keeps the lightweight interactive loop that originally lived at the
repository root while delegating the heavy lifting to the refactored
``rex.assistant`` helpers.  Keeping the documentation here helps developers
understand how to launch the assistant without digging through historical
commits.
"""

import importlib
import importlib.util
import os
import tempfile
import textwrap
from typing import Optional

import numpy as np
import sounddevice as sd
import simpleaudio as sa
import soundfile as sf
import whisper
from TTS.api import TTS

from config import load_config
from llm_client import LanguageModel
from wakeword_utils import detect_wakeword, load_wakeword_model

_WEB_SEARCH_SPEC = importlib.util.find_spec("plugins.web_search")
if _WEB_SEARCH_SPEC is not None:
    search_web = getattr(importlib.import_module("plugins.web_search"), "search_web", None)
else:
    search_web = None

from memory_utils import (
    extract_voice_reference,
    load_all_profiles,
    load_users_map,
    resolve_user_key,
)

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

WAKEWORD = CONFIG.wakeword
WAKEWORD_THRESHOLD = CONFIG.wakeword_threshold

if CONFIG.wake_sound_path:
    WAKE_SOUND_PATH = CONFIG.wake_sound_path
else:
    WAKE_SOUND_PATH = os.path.join(
        os.path.dirname(__file__), "assets", "rex_wake_acknowledgment (1).wav"
    )

SPEAKER_VOICES = {
    user: extract_voice_reference(profile)
    for user, profile in USER_PROFILES.items()
}

if ACTIVE_USER not in SPEAKER_VOICES:
    SPEAKER_VOICES[ACTIVE_USER] = None

COMMAND_DURATION = CONFIG.command_duration
WHISPER_MODEL_NAME = CONFIG.whisper_model

LLM = LanguageModel(CONFIG)
ASSISTANT_PERSONA = textwrap.dedent(
    """
    You are Rex, a focused AI voice assistant that keeps responses concise.
    Reference the active user's preferences when it helps personalise your
    answer. Always respond in natural English prose.
    """
)

# TODO: Actual implementation (wakeword, record, transcribe, respond) goes here.
