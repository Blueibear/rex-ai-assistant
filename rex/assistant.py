"""High level orchestration for the Rex voice assistant."""

from __future__ import annotations

import logging
from contextlib import suppress
from dataclasses import dataclass
from typing import Dict, Optional

try:  # pragma: no cover - prefer the real dependency when available
    from tqdm import tqdm
except ModuleNotFoundError:  # pragma: no cover - fallback in constrained environments
    class _NullProgress:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def __enter__(self) -> "_NullProgress":
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

        def update(self, *_args, **_kwargs) -> None:
            return None

    def tqdm(*_args, **_kwargs):  # type: ignore
        return _NullProgress()

from .config import settings
from .llm_client import GenerationConfig, LLMClient
from .logging_utils import configure_logger
from .memory import (
    append_history_entry,
    extract_voice_reference,
    load_all_profiles,
    load_recent_history,
    load_users_map,
    resolve_user_key,
)
from .plugins.base import Plugin, PluginContext
from .plugins.loader import load_plugins, shutdown_plugins
from .wakeword.listener import WakeWordListener

LOGGER = configure_logger(__name__)


@dataclass
class AssistantState:
    """Runtime state for the assistant."""

    user_id: str
    voice: Optional[str]
    enable_search: bool


class MicIndicator:
    """Lightweight helper that exposes microphone state changes."""

    def __init__(self) -> None:
        self.state = "idle"
        self._logger = logging.getLogger("rex.mic")

    def update(self, state: str) -> None:
        self.state = state
        self._logger.info("Microphone state: %s", state)


class VoiceAssistant:
    def __init__(
        self,
        *,
        wake_listener: WakeWordListener | None = None,
        llm_client: LLMClient | None = None,
        plugins: Dict[str, Plugin] | None = None,
    ) -> None:
        profiles = load_all_profiles()
        users_map = load_users_map()
        user_id = resolve_user_key(settings.user_id, users_map, profiles=profiles) or settings.user_id
        voice = extract_voice_reference(profiles.get(user_id, {})) if user_id in profiles else None
        self.state = AssistantState(user_id=user_id, voice=voice, enable_search=settings.enable_search_plugin)
        self._profiles = profiles
        self._plugins = plugins or load_plugins()
        self._llm = llm_client or LLMClient()
        self._mic_indicator = MicIndicator()
        self._wake_listener = wake_listener or WakeWordListener()

    def run(self) -> None:
        """Main blocking loop that waits for wake word then handles audio."""

        LOGGER.info("Assistant ready for user '%s'", self.state.user_id)
        try:
            while True:
                self._mic_indicator.update("listening")
                triggered = self._wake_listener.listen()
                if not triggered:
                    LOGGER.warning("Wake listener aborted; retrying")
                    continue
                self._mic_indicator.update("capturing")
                text = self._capture_transcript()
                if text:
                    self.handle_text(text)
                else:
                    LOGGER.warning("No speech detected after wake word")
        except KeyboardInterrupt:
            LOGGER.info("Assistant stopped via keyboard interrupt")
        finally:
            shutdown_plugins(self._plugins)

    def handle_text(self, text: str) -> str:
        """Process a chunk of text as if it was transcribed audio."""

        if not text.strip():
            raise ValueError("Text prompt must not be empty")
        append_history_entry(self.state.user_id, {"role": "user", "text": text})
        context = PluginContext(user_id=self.state.user_id, text=text)
        plugin_result = None
        if self.state.enable_search:
            for plugin in self._plugins.values():
                with suppress(Exception):
                    plugin_result = plugin.process(context)
                    if plugin_result:
                        LOGGER.info("Plugin %s provided supplemental context", plugin.name)
                        break
        history = load_recent_history(self.state.user_id, limit=5)
        prompt = self._build_prompt(text, history, plugin_result)
        response = self._generate_response(prompt)
        append_history_entry(self.state.user_id, {"role": "assistant", "text": response})
        self._speak(response)
        return response

    def _capture_transcript(self) -> str:
        """Placeholder microphone capture logic with error handling."""

        try:
            # In the real assistant this would trigger speech-to-text.  For now we simply
            # raise ``NotImplementedError`` to make the limitation explicit while ensuring
            # callers handle the exception gracefully.
            raise NotImplementedError("Audio capture is not implemented in the test environment")
        except NotImplementedError as exc:
            LOGGER.error("Audio capture unavailable: %s", exc)
            return ""
        except Exception as exc:  # pragma: no cover - hardware specific
            LOGGER.error("Unexpected audio capture failure: %s", exc)
            return ""

    def _generate_response(self, prompt: str) -> str:
        with tqdm(total=1, desc="Generating response", leave=False) as progress:
            result = self._llm.generate(prompt, config=GenerationConfig())
            progress.update(1)
        return result

    def _build_prompt(self, text: str, history: list[dict], plugin_result: Optional[str]) -> str:
        lines = ["You are Rex, a helpful voice assistant."]
        if history:
            lines.append("Recent conversation:")
            for entry in history:
                role = entry.get("role", "user").title()
                content = entry.get("text", "")
                lines.append(f"- {role}: {content}")
        if plugin_result:
            lines.append("Supplemental context from plugins:")
            lines.append(plugin_result)
        lines.append(f"User: {text}")
        lines.append("Assistant:")
        return "\n".join(lines)

    def _speak(self, text: str) -> None:
        try:
            # Text-to-speech is optional in the CI environment.  We simply log output to
            # provide deterministic feedback.
            LOGGER.info("Assistant response: %s", text)
        except Exception as exc:  # pragma: no cover - hardware specific
            LOGGER.error("Failed to speak response: %s", exc)

    def toggle_search_plugin(self, enabled: bool) -> None:
        self.state.enable_search = enabled
        LOGGER.info("Search plugin %s", "enabled" if enabled else "disabled")

    def change_user(self, user_id: str) -> None:
        resolved = resolve_user_key(user_id, load_users_map(), profiles=self._profiles)
        if not resolved:
            raise ValueError(f"Unknown user profile: {user_id}")
        self.state.user_id = resolved
        LOGGER.info("Switched active user to %s", resolved)

    def change_voice(self, voice_path: Optional[str]) -> None:
        self.state.voice = voice_path
        LOGGER.info("Voice sample changed to %s", voice_path or "default")


def build_cli_menu(assistant: VoiceAssistant) -> Dict[str, str]:
    return {
        "1": "Change voice sample",
        "2": "Switch active user",
        "3": "Toggle search plugin",
        "4": "Show microphone state",
        "0": "Exit",
    }


def run_cli() -> None:
    assistant = VoiceAssistant()
    menu = build_cli_menu(assistant)
    LOGGER.info("Launching Rex CLI menu")
    while True:
        for key, label in menu.items():
            print(f"{key}. {label}")
        choice = input("Select an option: ").strip()
        if choice == "0":
            break
        if choice == "1":
            path = input("Enter path to voice sample (blank for default): ").strip() or None
            assistant.change_voice(path)
        elif choice == "2":
            user = input("Enter user identifier: ").strip()
            try:
                assistant.change_user(user)
            except ValueError as exc:
                LOGGER.error("%s", exc)
        elif choice == "3":
            assistant.toggle_search_plugin(not assistant.state.enable_search)
        elif choice == "4":
            LOGGER.info("Microphone state: %s", assistant._mic_indicator.state)
        else:
            LOGGER.warning("Unknown menu option: %s", choice)
    shutdown_plugins(assistant._plugins)
    LOGGER.info("CLI menu closed")


__all__ = ["VoiceAssistant", "run_cli"]
