"""Command-line entry point for the Rex assistant.

This module intentionally mirrors the historical top-level script so existing
documentation that imports :mod:ex_assistant continues to work.
"""

from __future__ import annotations

# Load .env before accessing any environment variables
from utils.env_loader import load as _load_env
_load_env()

import asyncio
import logging
from collections.abc import Iterable, Sequence
from typing import Callable

from rex import settings
from rex.llm_client import LanguageModel
from rex.assistant import Assistant
from rex.logging_utils import configure_logging
from rex.plugins import PluginSpec, load_plugins, shutdown_plugins

logger = logging.getLogger(__name__)


class FunctionRouter:
    """Simple command router that maps predicates to handlers."""

    def __init__(self) -> None:
        self._routes: list[tuple[Callable[[str], bool], Callable[[str], str]]] = []

    def register(self, predicate: Callable[[str], bool], handler: Callable[[str], str]) -> None:
        self._routes.append((predicate, handler))

    def route(self, text: str) -> str | None:
        for predicate, handler in self._routes:
            try:
                if predicate(text):
                    return handler(text)
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.exception("Command handler failed: %s", exc)
        return None


ROUTER = FunctionRouter()
LLM: LanguageModel | None = None


def _ensure_llm() -> LanguageModel:
    global LLM
    if LLM is None or not hasattr(LLM, "generate"):
        LLM = LanguageModel()
    return LLM


def handle_command(text: str) -> str | None:
    """Dispatch `text` through the registered command handlers."""

    return ROUTER.route(text)


def generate_response(prompt: str, *, messages: Sequence[dict] | None = None) -> str:
    """Generate an assistant response, with safe fallback on failure."""

    routed = handle_command(prompt)
    if routed is not None:
        return routed

    try:
        if messages is not None:
            normalised = [
                {
                    "role": str(entry.get("role", "")),
                    "content": str(entry.get("content", "")),
                }
                for entry in messages
                if isinstance(entry, dict)
            ]
            return _ensure_llm().generate(messages=normalised)
        return _ensure_llm().generate(prompt)
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.exception("LLM generation failed: %s", exc)
        cleaned = prompt.strip()
        return f"I heard you say: {cleaned or '(silence)'}"


async def _chat_loop(assistant: Assistant) -> None:
    """Interactive CLI loop for chatting with Rex."""

    print("🎤 Rex assistant ready. Type 'exit' or 'quit' to stop.")
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
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception("Assistant failed to generate a reply: %s", exc)
            print(f"[error] {exc}")
            continue

        print(f"Rex: {reply}")


async def _run() -> None:
    """Configure logging, load plugins, and run the assistant loop."""

    configure_logging()
    plugin_specs: Iterable[PluginSpec] = load_plugins()
    assistant = Assistant(history_limit=settings.max_memory_items, plugins=plugin_specs)
    try:
        await _chat_loop(assistant)
    finally:
        shutdown_plugins(plugin_specs)


def main() -> int:
    """Main entry point for Rex assistant CLI."""

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        print("\nInterrupted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
