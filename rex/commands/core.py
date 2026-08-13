"""Core (doctor, chat, version, tools) commands for the Rex CLI.

Extracted verbatim from ``rex/cli.py`` (US-REM-027). Handler behavior,
argument definitions, help text, defaults, and exit codes are unchanged.
"""

from __future__ import annotations

import argparse
import sys

from rex.commands._helpers import _get_version


def _cli():
    """Return the ``rex.cli`` module at call time.

    ``rex.cli`` is the single patch point for service getters and command
    handlers (tests monkeypatch ``rex.cli.<name>``). Resolving through the
    module at call time preserves that behavior without creating an import
    cycle at module load time.
    """
    from rex import cli

    return cli


def cmd_doctor(args: argparse.Namespace) -> int:
    """Run environment diagnostics or the lightweight liveness probe."""
    from rex.doctor import run_diagnostics, run_healthcheck

    if getattr(args, "healthcheck", False):
        return run_healthcheck()

    debug = getattr(args, "debug", False)
    return run_diagnostics(
        verbose=args.verbose,
        debug=debug,
        release_gate=getattr(args, "release_gate", False),
    )


def cmd_chat(args: argparse.Namespace) -> int:
    """Start the interactive chat interface."""
    import asyncio

    from rex import settings
    from rex.assistant import Assistant
    from rex.logging_utils import configure_logging
    from rex.plugins import load_plugins, shutdown_plugins
    from rex.runtime.invocation import turn_invocation
    from rex.runtime.status import TurnStatusProjector
    from rex.runtime.turn import TurnSource
    from rex.services import initialize_services

    async def _chat_loop(assistant: Assistant) -> None:
        """Interactive CLI loop for chatting with Rex."""
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

            def show_status(update) -> None:  # noqa: ANN001
                print(f"Rex [{update.status.value.capitalize()}]")

            status_projector = TurnStatusProjector(show_status)
            try:
                with turn_invocation(TurnSource.CLI):
                    reply = await assistant.generate_reply(
                        user_input,
                        event_observer=status_projector.observe,
                    )
            except Exception as exc:
                print(f"[error] {exc}")
                continue

            print(f"Rex: {reply}")

    async def _run() -> None:
        """Configure logging, load plugins, and run the assistant loop."""
        from rex.identity import resolve_entrypoint_user_id

        configure_logging()
        initialize_services()
        plugin_specs = load_plugins()
        # Deliberate single-user profile selection (issue #303): Assistant no
        # longer invents an identity when user_id is omitted.
        assistant = Assistant(
            history_limit=settings.max_memory_items,
            plugins=plugin_specs,
            user_id=resolve_entrypoint_user_id(settings),
        )
        try:
            await _chat_loop(assistant)
        finally:
            shutdown_plugins(plugin_specs)

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        print("\nInterrupted.")
    return 0


def cmd_version(args: argparse.Namespace) -> int:
    """Show version information."""
    version = _get_version()
    print(f"rex-ai-assistant {version}")

    if args.verbose:
        print(f"Python: {sys.version}")
        try:
            import torch

            print(f"PyTorch: {torch.__version__}")
        except ImportError:
            print("PyTorch: not installed")
        try:
            import transformers

            print(f"Transformers: {transformers.__version__}")
        except ImportError:
            print("Transformers: not installed")

    return 0


def cmd_tools(args: argparse.Namespace) -> int:
    """List registered tools and their status."""
    from rex.openclaw.tool_registry import get_tool_registry

    registry = get_tool_registry()
    tools = registry.list_tools(include_disabled=args.all)

    if not tools:
        print("No tools registered.")
        return 0

    print("Rex Tool Registry")
    print("=" * 60)
    print()

    for tool in tools:
        status = registry.get_tool_status(tool.name)

        if status["ready"]:
            ready_icon = "[READY]"
        elif not status["enabled"]:
            ready_icon = "[DISABLED]"
        elif not status["credentials_available"]:
            ready_icon = "[NO CREDS]"
        elif not status["health_ok"]:
            ready_icon = "[UNHEALTHY]"
        else:
            ready_icon = "[UNKNOWN]"

        print(f"{tool.name} {ready_icon}")
        print(f"  Description: {tool.description}")

        if args.verbose:
            print(f"  Version: {tool.version}")
            if tool.capabilities:
                print(f"  Capabilities: {', '.join(tool.capabilities)}")
            if tool.required_credentials:
                print(f"  Required credentials: {', '.join(tool.required_credentials)}")
            if status.get("missing_credentials"):
                print(f"  Missing credentials: {', '.join(status['missing_credentials'])}")
            print(f"  Health: {status.get('health_message', 'n/a')}")

        print()

    total = len(tools)
    ready = sum(1 for t in tools if registry.get_tool_status(t.name)["ready"])
    print(f"Total: {total} tools, {ready} ready")

    return 0


def cmd_integrations(_args: argparse.Namespace) -> int:
    """Print evidence-based integration readiness without saying configured is connected."""
    from rex.config import load_config
    from rex.integration_state import build_integration_inventory

    print("Rex Integration Status")
    print("=" * 60)
    for integration in build_integration_inventory(load_config()):
        print(f"{integration.name}: {integration.state.value}")
        if integration.detail:
            print(f"  {integration.detail}")
    return 0


def register(subparsers: argparse._SubParsersAction) -> None:
    """Register this domain's subcommands on the top-level subparsers."""
    # doctor
    doctor_parser = subparsers.add_parser(
        "doctor",
        help="Run environment diagnostics",
        description="Check Python version, config files, environment variables, and external dependencies.",
    )
    doctor_parser.add_argument(
        "-v", "--verbose", action="store_true", help="Show detailed diagnostic information"
    )
    doctor_parser.add_argument(
        "--debug",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Include additional low-level diagnostic info (log level, env vars)",
    )
    doctor_parser.add_argument(
        "--release-gate",
        action="store_true",
        help="Exit nonzero for errors and actionable warnings",
    )
    doctor_parser.add_argument(
        "--healthcheck",
        action="store_true",
        help="Run a lightweight core-runtime liveness probe and exit nonzero on failure",
    )
    doctor_parser.set_defaults(func=_cli().cmd_doctor)

    # chat
    chat_parser = subparsers.add_parser(
        "chat",
        help="Start interactive chat (default)",
        description="Start an interactive text chat session with Rex.",
    )
    chat_parser.add_argument(
        "--no-tts",
        action="store_true",
        default=False,
        help="Disable TTS output (text-only mode; useful in CI or headless environments)",
    )
    chat_parser.set_defaults(func=_cli().cmd_chat)

    # version
    version_parser = subparsers.add_parser(
        "version",
        help="Show version information",
        description="Display Rex version and optionally dependency versions.",
    )
    version_parser.add_argument(
        "-v", "--verbose", action="store_true", help="Show dependency versions"
    )
    version_parser.set_defaults(func=_cli().cmd_version)

    # tools
    tools_parser = subparsers.add_parser(
        "tools",
        help="List registered tools and their status",
        description="Display all registered tools with health status and credential availability.",
    )
    tools_parser.add_argument(
        "-v", "--verbose", action="store_true", help="Show detailed tool information"
    )
    tools_parser.add_argument("-a", "--all", action="store_true", help="Include disabled tools")
    tools_parser.set_defaults(func=_cli().cmd_tools)

    integrations_parser = subparsers.add_parser(
        "integrations",
        help="Show evidence-based integration readiness",
        description="Distinguish configuration from reachability, authentication, and tested writes.",
    )
    integrations_parser.set_defaults(func=cmd_integrations)
