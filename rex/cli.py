"""Command-line interface for Rex AI Assistant.

This module provides the main CLI entry point with subcommands:
    rex doctor       - Run environment diagnostics
    rex chat         - Start interactive chat (default)
    rex version      - Show version information
    rex tools        - List registered tools and their status
    rex run-workflow - Run a workflow from a JSON file
    rex approvals    - List and manage pending approvals

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


def cmd_run_workflow(args: argparse.Namespace) -> int:
    """Run a workflow from a JSON file."""
    from pathlib import Path

    from rex.workflow import Workflow
    from rex.workflow_runner import WorkflowRunner

    workflow_path = Path(args.workflow)

    if not workflow_path.exists():
        print(f"Error: Workflow file not found: {workflow_path}")
        return 1

    try:
        workflow = Workflow.load_from_file(workflow_path)
    except Exception as e:
        print(f"Error: Failed to load workflow: {e}")
        return 1

    print(f"Loaded workflow: {workflow.title}")
    print(f"  ID: {workflow.workflow_id}")
    print(f"  Status: {workflow.status}")
    print(f"  Steps: {len(workflow.steps)}")
    print()

    runner = WorkflowRunner(workflow)

    if args.dry_run:
        print("Running in dry-run mode (no changes will be made)")
        print("-" * 60)

        result = runner.dry_run()

        for step in result.steps:
            status_icon = "[WOULD RUN]" if step.would_execute else "[SKIP]"
            print(f"{status_icon} {step.step_id}: {step.description}")
            if step.tool:
                print(f"    Tool: {step.tool}")
            print(f"    Policy: {step.policy_decision}")
            print(f"    Reason: {step.reason}")
            print()

        print("-" * 60)
        if result.would_complete:
            print("Workflow would complete successfully.")
        else:
            print(f"Workflow would not complete: {result.blocking_reason}")

        return 0

    if args.resume:
        if workflow.status != "blocked":
            print(f"Error: Cannot resume workflow in status '{workflow.status}'")
            print("Only 'blocked' workflows can be resumed.")
            return 1

        print(f"Resuming blocked workflow...")
        print(f"  Blocking approval: {workflow.blocking_approval_id}")
        print()

        try:
            result = runner.resume()
        except ValueError as e:
            print(f"Error: {e}")
            return 1
    else:
        print("Running workflow...")
        print("-" * 60)
        result = runner.run()

    print()
    print("-" * 60)
    print(f"Workflow finished")
    print(f"  Status: {result.status}")
    print(f"  Steps executed: {result.steps_executed}/{result.steps_total}")

    if result.error:
        print(f"  Error: {result.error}")
        return 1

    if result.blocking_approval_id:
        print(f"  Blocked on approval: {result.blocking_approval_id}")
        print()
        print("To approve, run:")
        print(f"  rex approvals approve {result.blocking_approval_id}")
        print()
        print("Then resume with:")
        print(f"  rex run-workflow {workflow_path} --resume")
        return 0

    return 0


def cmd_approvals(args: argparse.Namespace) -> int:
    """List and manage pending approvals."""
    from rex.workflow_runner import (
        list_pending_approvals,
        approve_workflow,
        deny_workflow,
    )
    from rex.workflow import WorkflowApproval

    if args.approve:
        approval_id = args.approve
        if approve_workflow(approval_id, decided_by="cli_user", reason=args.reason):
            print(f"Approved: {approval_id}")
            return 0
        else:
            print(f"Error: Approval not found: {approval_id}")
            return 1

    if args.deny:
        approval_id = args.deny
        reason = args.reason or "Denied via CLI"
        if deny_workflow(approval_id, decided_by="cli_user", reason=reason):
            print(f"Denied: {approval_id}")
            return 0
        else:
            print(f"Error: Approval not found: {approval_id}")
            return 1

    if args.show:
        approval_id = args.show
        approval = WorkflowApproval.load(approval_id)
        if approval is None:
            print(f"Error: Approval not found: {approval_id}")
            return 1

        print(f"Approval: {approval.approval_id}")
        print(f"  Status: {approval.status}")
        print(f"  Workflow: {approval.workflow_id}")
        print(f"  Step: {approval.step_id}")
        print(f"  Description: {approval.step_description}")
        print(f"  Tool: {approval.tool_call_summary}")
        print(f"  Requested at: {approval.requested_at}")
        if approval.decided_at:
            print(f"  Decided at: {approval.decided_at}")
            print(f"  Decided by: {approval.decided_by}")
        if approval.reason:
            print(f"  Reason: {approval.reason}")
        return 0

    # List pending approvals
    pending = list_pending_approvals()

    if not pending:
        print("No pending approvals.")
        return 0

    print("Pending Approvals")
    print("=" * 60)
    print()

    for approval in pending:
        print(f"{approval.approval_id}")
        print(f"  Workflow: {approval.workflow_id}")
        print(f"  Step: {approval.step_id}")
        print(f"  Description: {approval.step_description}")
        if approval.tool_call_summary:
            print(f"  Tool: {approval.tool_call_summary}")
        print(f"  Requested: {approval.requested_at}")
        print()

    print(f"Total: {len(pending)} pending approval(s)")
    print()
    print("To approve: rex approvals --approve <approval_id>")
    print("To deny:    rex approvals --deny <approval_id> --reason 'reason'")

    return 0


def cmd_workflows(args: argparse.Namespace) -> int:
    """List workflows."""
    from rex.workflow import Workflow

    workflows = Workflow.list_workflows(status=args.status)

    if not workflows:
        print("No workflows found.")
        return 0

    print("Workflows")
    print("=" * 60)
    print()

    for wf in workflows:
        status_icon = {
            "queued": "[QUEUED]",
            "running": "[RUNNING]",
            "blocked": "[BLOCKED]",
            "completed": "[DONE]",
            "failed": "[FAILED]",
            "canceled": "[CANCELED]",
        }.get(wf.status, f"[{wf.status.upper()}]")

        print(f"{status_icon} {wf.workflow_id}")
        print(f"  Title: {wf.title}")
        print(f"  Steps: {wf.current_step_index}/{len(wf.steps)}")
        print(f"  Created: {wf.created_at}")
        if wf.error:
            print(f"  Error: {wf.error}")
        print()

    print(f"Total: {len(workflows)} workflow(s)")

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
  rex run-workflow workflow.json        Run a workflow
  rex run-workflow workflow.json --dry-run  Preview workflow
  rex run-workflow workflow.json --resume   Resume blocked workflow
  rex approvals       List pending approvals
  rex approvals --approve <id>   Approve a pending request
  rex workflows       List all workflows

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

    # run-workflow subcommand
    workflow_parser = subparsers.add_parser(
        "run-workflow",
        help="Run a workflow from a JSON file",
        description="Load and execute a workflow definition. Supports dry-run and resume modes.",
    )
    workflow_parser.add_argument(
        "workflow",
        type=str,
        help="Path to the workflow JSON file",
    )
    workflow_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview workflow without executing (no changes made)",
    )
    workflow_parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume a blocked workflow after approval",
    )
    workflow_parser.set_defaults(func=cmd_run_workflow)

    # approvals subcommand
    approvals_parser = subparsers.add_parser(
        "approvals",
        help="List and manage pending approvals",
        description="View, approve, or deny pending workflow approvals.",
    )
    approvals_parser.add_argument(
        "--approve",
        type=str,
        metavar="ID",
        help="Approve the specified approval request",
    )
    approvals_parser.add_argument(
        "--deny",
        type=str,
        metavar="ID",
        help="Deny the specified approval request",
    )
    approvals_parser.add_argument(
        "--show",
        type=str,
        metavar="ID",
        help="Show details of a specific approval",
    )
    approvals_parser.add_argument(
        "--reason",
        type=str,
        help="Reason for approval/denial decision",
    )
    approvals_parser.set_defaults(func=cmd_approvals)

    # workflows subcommand
    workflows_parser = subparsers.add_parser(
        "workflows",
        help="List workflows",
        description="List all workflows with their status.",
    )
    workflows_parser.add_argument(
        "--status",
        type=str,
        choices=["queued", "running", "blocked", "completed", "failed", "canceled"],
        help="Filter by workflow status",
    )
    workflows_parser.set_defaults(func=cmd_workflows)

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
