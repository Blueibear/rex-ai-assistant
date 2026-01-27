"""Command-line interface for Rex AI Assistant.

This module provides the main CLI entry point with subcommands:
    rex doctor  - Run environment diagnostics
    rex chat    - Start interactive chat (default)
    rex version - Show version information
    rex tools   - List registered tools and their status

Usage:
    rex [command] [options]

If no command is specified, the chat interface is started.
"""

from __future__ import annotations

import argparse
import sys
from typing import Sequence


def _get_version() -> str:
    """Return the current Rex version."""
    try:
        from rex.contracts.version import CONTRACT_VERSION

        return CONTRACT_VERSION
    except ImportError:
        return "0.1.0"


def cmd_doctor(args: argparse.Namespace) -> int:
    """Run environment diagnostics."""
    from rex.doctor import run_diagnostics

    return run_diagnostics(verbose=args.verbose)


def cmd_chat(args: argparse.Namespace) -> int:
    """Start the interactive chat interface."""
    # Import here to avoid loading heavy dependencies unless needed
    import asyncio

    from rex.assistant import Assistant
    from rex.logging_utils import configure_logging
    from rex.plugins import load_plugins, shutdown_plugins
    from rex import settings

    async def _chat_loop(assistant: "Assistant") -> None:
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

            try:
                reply = await assistant.generate_reply(user_input)
            except Exception as exc:
                print(f"[error] {exc}")
                continue

            print(f"Rex: {reply}")

    async def _run() -> None:
        """Configure logging, load plugins, and run the assistant loop."""
        configure_logging()
        plugin_specs = load_plugins()
        assistant = Assistant(history_limit=settings.max_memory_items, plugins=plugin_specs)
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
        import sys

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
    from rex.tool_registry import get_tool_registry

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

        # Status indicators
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
            if status["missing_credentials"]:
                print(f"  Missing credentials: {', '.join(status['missing_credentials'])}")
            print(f"  Health: {status['health_message']}")

        print()

    # Summary
    total = len(tools)
    ready = sum(1 for t in tools if registry.get_tool_status(t.name)["ready"])
    print(f"Total: {total} tools, {ready} ready")

    return 0


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser with all subcommands."""
    parser = argparse.ArgumentParser(
        prog="rex",
        description="Rex AI Assistant - Voice-activated AI assistant with speech recognition and synthesis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  rex doctor          Run environment diagnostics
  rex doctor -v       Run diagnostics with verbose output
  rex chat            Start interactive chat
  rex version         Show version information
  rex tools           List registered tools and status
  rex tools -v        Show detailed tool information

For more information, visit: https://github.com/Blueibear/rex-ai-assistant
""",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {_get_version()}",
    )

    subparsers = parser.add_subparsers(
        title="commands",
        description="Available commands",
        dest="command",
        metavar="COMMAND",
    )

    # doctor subcommand
    doctor_parser = subparsers.add_parser(
        "doctor",
        help="Run environment diagnostics",
        description="Check Python version, config files, environment variables, and external dependencies.",
    )
    doctor_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show detailed diagnostic information",
    )
    doctor_parser.set_defaults(func=cmd_doctor)

    # chat subcommand
    chat_parser = subparsers.add_parser(
        "chat",
        help="Start interactive chat (default)",
        description="Start an interactive text chat session with Rex.",
    )
    chat_parser.set_defaults(func=cmd_chat)

    # version subcommand
    version_parser = subparsers.add_parser(
        "version",
        help="Show version information",
        description="Display Rex version and optionally dependency versions.",
    )
    version_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show dependency versions",
    )
    version_parser.set_defaults(func=cmd_version)

    # tools subcommand
    tools_parser = subparsers.add_parser(
        "tools",
        help="List registered tools and their status",
        description="Display all registered tools with health status and credential availability.",
    )
    tools_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show detailed tool information",
    )
    tools_parser.add_argument(
        "-a",
        "--all",
        action="store_true",
        help="Include disabled tools",
    )
    tools_parser.set_defaults(func=cmd_tools)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Main entry point for the Rex CLI.

    Args:
        argv: Command-line arguments (defaults to sys.argv[1:])

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    parser = create_parser()
    args = parser.parse_args(argv)

    # If no command specified, default to chat
    if args.command is None:
        # Check if just -v/--verbose was passed without a command
        if hasattr(args, "verbose") and args.verbose:
            # Show help with verbose flag but no command
            parser.print_help()
            return 0
        # Default to chat mode
        args.func = cmd_chat
        args.verbose = getattr(args, "verbose", False)

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
