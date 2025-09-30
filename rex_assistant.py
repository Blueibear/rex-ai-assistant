"""Command-line entry point for the Rex assistant.

This module intentionally mirrors the historical top-level script so existing
documentation that imports :mod:`rex_assistant` continues to work. A leading
triple-quoted docstring is required so that Python does not treat the module
preamble as bare text, which would otherwise raise a :class:`SyntaxError` at
import time. The helper functions continue to delegate the heavy lifting to
``rex.assistant``.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Iterable

from rex import settings
from rex.assistant import Assistant
from rex.logging_utils import configure_logging
from rex.plugins import PluginSpec, load_plugins, shutdown_plugins

logger = logging.getLogger(__name__)


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
        except Exception as exc:  # pragma: no cover - runtime safeguard
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
    except KeyboardInterrupt:  # pragma: no cover - manual interruption
        print("\nInterrupted.")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
