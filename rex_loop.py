"""Async CLI loop driving the full Rex voice experience.

This preserves the semantics of the original ``rex_loop.py`` entry point while
bridging to the refactored voice loop package introduced during previous
iterations.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
from typing import Iterable

import rex
from rex.assistant import Assistant
from rex.assistant_errors import AssistantError, WakeWordError
from rex.logging_utils import configure_logging
from rex.mqtt_audio_router import MqttAudioRouter
from rex.plugins import PluginSpec, load_plugins, shutdown_plugins
from rex.voice_loop import build_voice_loop

logger = logging.getLogger(__name__)


def _select_plugins(enabled: Iterable[str] | None) -> list[PluginSpec]:
    specs = load_plugins()
    if not enabled:
        return specs
    enabled_set = {name.strip() for name in enabled if name}
    return [spec for spec in specs if spec.name in enabled_set]


async def _run(args) -> None:
    configure_logging()
    plugin_specs = _select_plugins(args.enable_plugin)

    if args.user:
        os.environ["REX_ACTIVE_USER"] = args.user
        rex.reload_settings()

    assistant = Assistant(
        history_limit=rex.settings.max_memory_items,
        plugins=plugin_specs
    )

    try:
        voice_loop = build_voice_loop(assistant)
    except (AssistantError, WakeWordError) as exc:
        logger.error("Unable to initialise voice loop: %s", exc)
        return

    mqtt_router: MqttAudioRouter | None = None
    try:
        mqtt_router = MqttAudioRouter(assistant=assistant)
        await mqtt_router.start()
        logger.info("MQTT audio router started.")
    except Exception as exc:  # pragma: no cover - defensive startup log
        logger.error("Unable to start MQTT audio router: %s", exc)
        mqtt_router = None

    logger.info("🎙️ Voice loop started. Press Ctrl+C to exit.")
    try:
        await voice_loop.run()
    finally:
        if mqtt_router is not None:
            await mqtt_router.stop()
        shutdown_plugins(plugin_specs)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the Rex voice assistant loop.")
    parser.add_argument("--user", help="Override the active user profile")
    parser.add_argument(
        "--enable-plugin",
        action="append",
        metavar="NAME",
        help="Explicitly enable a plugin by name (omit to load all)",
    )

    args = parser.parse_args(argv)

    try:
        asyncio.run(_run(args))
    except KeyboardInterrupt:
        print("\nInterrupted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

