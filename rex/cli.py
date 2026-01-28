"""Command-line interface for Rex AI Assistant.

This module provides the main CLI entry point with subcommands:
    rex doctor       - Run environment diagnostics
    rex chat         - Start interactive chat (default)
    rex version      - Show version information
    rex tools        - List registered tools and their status
    rex run-workflow - Run a workflow from a JSON file
    rex approvals    - List and manage pending approvals
    rex memory       - Manage working and long-term memory
    rex kb           - Manage knowledge base documents

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
    from rex.services import initialize_services
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
        initialize_services()
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


def cmd_scheduler(args: argparse.Namespace) -> int:
    """Manage scheduler jobs."""
    from rex.services import initialize_services

    services = initialize_services(storage_path=args.storage)
    scheduler = services.scheduler

    if args.scheduler_command == "list":
        jobs = scheduler.list_jobs()
        if not jobs:
            print("No jobs scheduled.")
            return 0
        print("Scheduled Jobs")
        print("=" * 60)
        for job in jobs:
            next_run = job.next_run_at.isoformat() if job.next_run_at else "n/a"
            status = "enabled" if job.enabled else "disabled"
            print(f"{job.job_id} | {job.name} | every {job.interval_seconds}s | {status}")
            print(f"  Next run: {next_run}")
        return 0

    if args.scheduler_command == "run":
        if scheduler.run_job(args.job_id, manual=True):
            print(f"Job {args.job_id} executed.")
            return 0
        print(f"Job not found: {args.job_id}")
        return 1

    print("Unknown scheduler command.")
    return 1


def cmd_email(args: argparse.Namespace) -> int:
    """Email triage commands."""
    from rex.services import initialize_services

    services = initialize_services()
    email_service = services.email

    if args.email_command == "unread":
        triaged = email_service.triage_unread()
        if not triaged:
            print("No unread messages.")
            return 0
        print("Unread Email Summary")
        print("=" * 60)
        for item in triaged:
            received = item["received_at"].isoformat()
            print(f"{item['message_id']} | {item['subject']} | {item['sender']}")
            print(f"  Category: {item['category']} | Received: {received}")
            print(f"  Summary: {item['summary']}")
        return 0

    print("Unknown email command.")
    return 1


def cmd_calendar(args: argparse.Namespace) -> int:
    """Calendar commands."""
    from rex.services import initialize_services

    services = initialize_services()
    calendar = services.calendar

    if args.calendar_command == "upcoming":
        events = calendar.list_upcoming()
        if not events:
            print("No upcoming events.")
            return 0
        print("Upcoming Events")
        print("=" * 60)
        for event in events:
            start = event.start_time.isoformat()
            end = event.end_time.isoformat()
            location = event.location or "n/a"
            print(f"{event.event_id} | {event.title}")
            print(f"  {start} - {end} | {location}")
        return 0

    print("Unknown calendar command.")
    return 1


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
        # Prefer persisted workflow state for resume
        persisted = Workflow.load(workflow.workflow_id)
        if persisted is not None:
            workflow = persisted
            runner = WorkflowRunner(workflow)

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
        print(f"  rex approvals --approve {result.blocking_approval_id}")
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


def cmd_memory(args: argparse.Namespace) -> int:
    """Manage working and long-term memory."""
    import json
    from datetime import timedelta

    from rex.memory import (
        get_working_memory,
        get_long_term_memory,
    )

    subcommand = args.memory_command

    if subcommand == "recent":
        # Show recent working memory entries
        wm = get_working_memory()
        n = args.count or 10
        entries = wm.get_recent_with_timestamps(n)

        if not entries:
            print("No working memory entries.")
            return 0

        print("Recent Working Memory")
        print("=" * 60)
        print()

        for entry in entries:
            timestamp = entry.get("timestamp", "unknown")
            content = entry.get("content", "")
            print(f"[{timestamp}]")
            print(f"  {content}")
            print()

        print(f"Total: {len(entries)} entries shown (of {len(wm)})")
        return 0

    elif subcommand == "add":
        # Add a long-term memory entry
        ltm = get_long_term_memory()

        category = args.category
        try:
            content = json.loads(args.content)
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON content: {e}")
            return 1

        expires_in = None
        if args.ttl:
            expires_in = _parse_ttl(args.ttl)
            if expires_in is None:
                print(f"Error: Invalid TTL format: {args.ttl}")
                print("Use formats like: 7d, 24h, 30m, 1w, 10s")
                return 1

        entry = ltm.add_entry(
            category=category,
            content=content,
            expires_in=expires_in,
            sensitive=args.sensitive,
        )

        print(f"Added memory entry: {entry.entry_id}")
        print(f"  Category: {entry.category}")
        print(f"  Expires: {entry.expires_at or 'never'}")
        print(f"  Sensitive: {entry.sensitive}")
        return 0

    elif subcommand == "search":
        # Search long-term memory
        ltm = get_long_term_memory()

        keyword = args.keyword
        category = args.category
        show_sensitive = args.show_sensitive

        results = ltm.search(
            category=category,
            keyword=keyword,
            include_sensitive=True,  # Always include, but filter display
        )

        if not results:
            print("No matching memory entries found.")
            return 0

        print("Long-Term Memory Search Results")
        print("=" * 60)
        print()

        for entry in results:
            print(f"{entry.entry_id} [{entry.category}]")
            print(f"  Created: {entry.created_at}")
            if entry.expires_at:
                print(f"  Expires: {entry.expires_at}")
            if entry.sensitive:
                print("  [SENSITIVE]")
                if show_sensitive:
                    print(f"  Content: {json.dumps(entry.content, indent=4)}")
                else:
                    print("  Content: <hidden - use --show-sensitive>")
            else:
                print(f"  Content: {json.dumps(entry.content, indent=4)}")
            print()

        print(f"Total: {len(results)} entries found")
        return 0

    elif subcommand == "forget":
        # Delete a memory entry
        ltm = get_long_term_memory()

        entry_id = args.entry_id
        if ltm.forget(entry_id):
            print(f"Deleted memory entry: {entry_id}")
            return 0
        else:
            print(f"Error: Entry not found: {entry_id}")
            return 1

    elif subcommand == "clear":
        # Clear working memory
        wm = get_working_memory()
        count = len(wm)
        wm.clear()
        print(f"Cleared {count} working memory entries.")
        return 0

    elif subcommand == "retention":
        # Run retention policy
        ltm = get_long_term_memory()
        deleted = ltm.run_retention_policy()
        print(f"Retention policy deleted {deleted} expired entries.")
        return 0

    elif subcommand == "stats":
        # Show memory statistics
        wm = get_working_memory()
        ltm = get_long_term_memory()

        print("Memory Statistics")
        print("=" * 60)
        print()
        print(f"Working Memory: {len(wm)} entries")
        print(f"Long-Term Memory: {len(ltm)} entries")
        print()

        counts = ltm.count_by_category()
        if counts:
            print("Long-Term Memory by Category:")
            for cat, count in sorted(counts.items()):
                print(f"  {cat}: {count}")

        return 0

    else:
        print("Unknown memory subcommand. Use 'rex memory --help'")
        return 1


def cmd_kb(args: argparse.Namespace) -> int:
    """Manage knowledge base."""
    from rex.knowledge_base import get_knowledge_base

    kb = get_knowledge_base()
    subcommand = args.kb_command

    if subcommand == "ingest":
        # Ingest a file
        path = args.path
        title = args.title
        tags = args.tags.split(",") if args.tags else None

        try:
            doc = kb.ingest_file(path, title=title, tags=tags)
        except FileNotFoundError:
            print(f"Error: File not found: {path}")
            return 1
        except ValueError as e:
            print(f"Error: {e}")
            return 1

        print(f"Ingested document: {doc.doc_id}")
        print(f"  Title: {doc.title}")
        print(f"  Words: {doc.word_count}")
        if doc.tags:
            print(f"  Tags: {', '.join(doc.tags)}")
        return 0

    elif subcommand == "search":
        # Search documents
        query = args.query
        max_results = args.max_results or 5

        results = kb.search(query, max_results=max_results)

        if not results:
            print("No matching documents found.")
            return 0

        print("Knowledge Base Search Results")
        print("=" * 60)
        print()

        for doc in results:
            print(f"{doc.doc_id}: {doc.title}")
            if doc.tags:
                print(f"  Tags: {', '.join(doc.tags)}")
            print(f"  Words: {doc.word_count}")
            if args.verbose:
                # Show snippet
                snippet = doc.content[:200].replace("\n", " ")
                if len(doc.content) > 200:
                    snippet += "..."
                print(f"  Snippet: {snippet}")
            print()

        print(f"Total: {len(results)} documents found")
        return 0

    elif subcommand == "list":
        # List documents
        limit = args.limit or 20
        docs = kb.list_documents(limit=limit)

        if not docs:
            print("No documents in knowledge base.")
            return 0

        print("Knowledge Base Documents")
        print("=" * 60)
        print()

        for doc in docs:
            print(f"{doc.doc_id}: {doc.title}")
            if doc.tags:
                print(f"  Tags: {', '.join(doc.tags)}")
            print(f"  Words: {doc.word_count}")
            print(f"  Created: {doc.created_at}")
            print()

        print(f"Total: {len(kb)} documents")
        return 0

    elif subcommand == "show":
        # Show a specific document
        doc_id = args.doc_id
        doc = kb.get_document(doc_id)

        if not doc:
            print(f"Error: Document not found: {doc_id}")
            return 1

        print(f"Document: {doc.doc_id}")
        print("=" * 60)
        print(f"Title: {doc.title}")
        if doc.source_path:
            print(f"Source: {doc.source_path}")
        if doc.tags:
            print(f"Tags: {', '.join(doc.tags)}")
        print(f"Words: {doc.word_count}")
        print(f"Created: {doc.created_at}")
        print()
        print("Content:")
        print("-" * 60)
        print(doc.content)
        return 0

    elif subcommand == "delete":
        # Delete a document
        doc_id = args.doc_id
        if kb.delete_document(doc_id):
            print(f"Deleted document: {doc_id}")
            return 0
        else:
            print(f"Error: Document not found: {doc_id}")
            return 1

    elif subcommand == "cite":
        # Get citations for a query
        query = args.query
        citations = kb.get_citations(query)

        if not citations:
            print("No citations found.")
            return 0

        print("Citations")
        print("=" * 60)
        print()

        for doc_id in citations:
            doc = kb.get_document(doc_id)
            if doc:
                print(f"{doc_id}: {doc.title}")

        print()
        print(f"Total: {len(citations)} documents can be cited")
        return 0

    elif subcommand == "tags":
        # List all tags
        tags = kb.list_tags()

        if not tags:
            print("No tags in knowledge base.")
            return 0

        print("Knowledge Base Tags")
        print("=" * 60)
        print()
        for tag in tags:
            print(f"  {tag}")
        print()
        print(f"Total: {len(tags)} tags")
        return 0

    else:
        print("Unknown kb subcommand. Use 'rex kb --help'")
        return 1


def _parse_ttl(ttl_str: str):
    """Parse TTL string like '7d', '24h', '30m', '1w', '10s' into timedelta."""
    from datetime import timedelta

    ttl_str = ttl_str.strip().lower()
    if not ttl_str:
        return None

    try:
        if ttl_str.endswith("w"):
            return timedelta(weeks=int(ttl_str[:-1]))
        elif ttl_str.endswith("d"):
            return timedelta(days=int(ttl_str[:-1]))
        elif ttl_str.endswith("h"):
            return timedelta(hours=int(ttl_str[:-1]))
        elif ttl_str.endswith("m"):
            return timedelta(minutes=int(ttl_str[:-1]))
        elif ttl_str.endswith("s"):
            return timedelta(seconds=int(ttl_str[:-1]))
        else:
            # Try parsing as days
            return timedelta(days=int(ttl_str))
    except ValueError:
        return None


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

Memory commands:
  rex memory recent 5                     Show last 5 working memory entries
  rex memory add "facts" '{"key":"value"}'  Add a long-term memory entry
  rex memory add "secrets" '{"api":"key"}' --sensitive --ttl=7d
  rex memory search "keyword"             Search long-term memory
  rex memory search --category preferences
  rex memory forget <entry_id>            Delete a memory entry
  rex memory stats                        Show memory statistics

Knowledge base commands:
  rex kb ingest /path/to/file.txt --title "My Doc" --tags notes,project
  rex kb search "query"                   Search documents
  rex kb search "query" -v                Search with content snippets
  rex kb list                             List all documents
  rex kb show <doc_id>                    Show document content
  rex kb delete <doc_id>                  Delete a document
  rex kb cite "phrase"                    Get citations for a phrase

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

    # memory subcommand
    memory_parser = subparsers.add_parser(
        "memory",
        help="Manage working and long-term memory",
        description="Manage Rex's working memory (recent context) and long-term structured memory.",
    )
    memory_subparsers = memory_parser.add_subparsers(
        title="memory commands",
        dest="memory_command",
        metavar="COMMAND",
    )

    # memory recent
    memory_recent = memory_subparsers.add_parser(
        "recent",
        help="Show recent working memory entries",
        description="Display the most recent entries from working memory.",
    )
    memory_recent.add_argument(
        "count",
        type=int,
        nargs="?",
        default=10,
        help="Number of entries to show (default: 10)",
    )
    memory_recent.add_argument(
        "--show-sensitive",
        action="store_true",
        help="No-op for working memory (kept for CLI compatibility)",
    )
    memory_recent.set_defaults(func=cmd_memory)

    # memory add
    memory_add = memory_subparsers.add_parser(
        "add",
        help="Add a long-term memory entry",
        description="Add a new entry to long-term memory with category and JSON content.",
    )
    memory_add.add_argument(
        "category",
        type=str,
        help="Category for the entry (e.g., 'preferences', 'facts')",
    )
    memory_add.add_argument(
        "content",
        type=str,
        help='JSON content for the entry (e.g., \'{"key": "value"}\')',
    )
    memory_add.add_argument(
        "--ttl",
        type=str,
        help="Time to live (e.g., '7d', '24h', '30m', '1w', '10s')",
    )
    memory_add.add_argument(
        "--sensitive",
        action="store_true",
        help="Mark this entry as containing sensitive data",
    )
    memory_add.set_defaults(func=cmd_memory)

    # memory search
    memory_search = memory_subparsers.add_parser(
        "search",
        help="Search long-term memory",
        description="Search for entries in long-term memory by keyword or category.",
    )
    memory_search.add_argument(
        "keyword",
        type=str,
        nargs="?",
        help="Keyword to search for in content",
    )
    memory_search.add_argument(
        "--category",
        type=str,
        help="Filter by category",
    )
    memory_search.add_argument(
        "--show-sensitive",
        action="store_true",
        help="Show content of sensitive entries",
    )
    memory_search.set_defaults(func=cmd_memory)

    # memory forget
    memory_forget = memory_subparsers.add_parser(
        "forget",
        help="Delete a memory entry",
        description="Delete a specific entry from long-term memory.",
    )
    memory_forget.add_argument(
        "entry_id",
        type=str,
        help="ID of the entry to delete",
    )
    memory_forget.set_defaults(func=cmd_memory)

    # memory clear
    memory_clear = memory_subparsers.add_parser(
        "clear",
        help="Clear all working memory",
        description="Remove all entries from working memory.",
    )
    memory_clear.set_defaults(func=cmd_memory)

    # memory retention
    memory_retention = memory_subparsers.add_parser(
        "retention",
        help="Run retention policy",
        description="Delete all expired entries from long-term memory.",
    )
    memory_retention.set_defaults(func=cmd_memory)

    # memory stats
    memory_stats = memory_subparsers.add_parser(
        "stats",
        help="Show memory statistics",
        description="Display statistics about working and long-term memory.",
    )
    memory_stats.set_defaults(func=cmd_memory)

    memory_parser.set_defaults(func=cmd_memory, memory_command="stats")

    # kb (knowledge base) subcommand
    kb_parser = subparsers.add_parser(
        "kb",
        help="Manage knowledge base",
        description="Manage Rex's knowledge base for document storage and retrieval.",
    )
    kb_subparsers = kb_parser.add_subparsers(
        title="knowledge base commands",
        dest="kb_command",
        metavar="COMMAND",
    )

    # kb ingest
    kb_ingest = kb_subparsers.add_parser(
        "ingest",
        help="Ingest a file into the knowledge base",
        description="Read and index a text file for later search and retrieval.",
    )
    kb_ingest.add_argument(
        "path",
        type=str,
        help="Path to the file to ingest",
    )
    kb_ingest.add_argument(
        "--title",
        type=str,
        help="Title for the document (defaults to filename)",
    )
    kb_ingest.add_argument(
        "--tags",
        type=str,
        help="Comma-separated list of tags",
    )
    kb_ingest.set_defaults(func=cmd_kb)

    # kb search
    kb_search = kb_subparsers.add_parser(
        "search",
        help="Search the knowledge base",
        description="Search for documents matching a query.",
    )
    kb_search.add_argument(
        "query",
        type=str,
        help="Search query",
    )
    kb_search.add_argument(
        "--max-results",
        type=int,
        default=5,
        help="Maximum number of results (default: 5)",
    )
    kb_search.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show content snippets",
    )
    kb_search.set_defaults(func=cmd_kb)

    # kb list
    kb_list = kb_subparsers.add_parser(
        "list",
        help="List all documents",
        description="List all documents in the knowledge base.",
    )
    kb_list.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Maximum number of documents to list (default: 20)",
    )
    kb_list.set_defaults(func=cmd_kb)

    # kb show
    kb_show = kb_subparsers.add_parser(
        "show",
        help="Show a document's content",
        description="Display the full content of a document.",
    )
    kb_show.add_argument(
        "doc_id",
        type=str,
        help="Document ID",
    )
    kb_show.set_defaults(func=cmd_kb)

    # kb delete
    kb_delete = kb_subparsers.add_parser(
        "delete",
        help="Delete a document",
        description="Remove a document from the knowledge base.",
    )
    kb_delete.add_argument(
        "doc_id",
        type=str,
        help="Document ID to delete",
    )
    kb_delete.set_defaults(func=cmd_kb)

    # kb cite
    kb_cite = kb_subparsers.add_parser(
        "cite",
        help="Get citations for a query",
        description="Find documents that can be cited for a specific phrase or term.",
    )
    kb_cite.add_argument(
        "query",
        type=str,
        help="Text to find citations for",
    )
    kb_cite.set_defaults(func=cmd_kb)

    # kb tags
    kb_tags = kb_subparsers.add_parser(
        "tags",
        help="List all tags",
        description="List all unique tags in the knowledge base.",
    )
    kb_tags.set_defaults(func=cmd_kb)

    kb_parser.set_defaults(func=cmd_kb, kb_command="list")

    # scheduler subcommand
    scheduler_parser = subparsers.add_parser(
        "scheduler",
        help="Manage scheduled jobs",
        description="List or run scheduled jobs.",
    )
    scheduler_parser.add_argument(
        "--storage",
        type=str,
        default=None,
        help="Path to scheduler storage (default: data/scheduler/jobs.json).",
    )
    scheduler_subparsers = scheduler_parser.add_subparsers(dest="scheduler_command")

    scheduler_list = scheduler_subparsers.add_parser(
        "list",
        help="List scheduled jobs",
    )
    scheduler_list.set_defaults(func=cmd_scheduler, scheduler_command="list")

    scheduler_run = scheduler_subparsers.add_parser(
        "run",
        help="Run a scheduled job immediately",
    )
    scheduler_run.add_argument("job_id", type=str, help="Job ID to run")
    scheduler_run.set_defaults(func=cmd_scheduler, scheduler_command="run")

    scheduler_parser.set_defaults(func=cmd_scheduler, scheduler_command="list")

    # email subcommand
    email_parser = subparsers.add_parser(
        "email",
        help="Inspect unread email (mock data)",
        description="List unread emails and triage them.",
    )
    email_subparsers = email_parser.add_subparsers(dest="email_command")

    email_unread = email_subparsers.add_parser(
        "unread",
        help="List unread emails",
    )
    email_unread.set_defaults(func=cmd_email, email_command="unread")

    email_parser.set_defaults(func=cmd_email, email_command="unread")

    # calendar subcommand
    calendar_parser = subparsers.add_parser(
        "calendar",
        help="Inspect upcoming calendar events (mock data)",
        description="List upcoming calendar events.",
    )
    calendar_subparsers = calendar_parser.add_subparsers(dest="calendar_command")

    calendar_upcoming = calendar_subparsers.add_parser(
        "upcoming",
        help="List upcoming calendar events",
    )
    calendar_upcoming.set_defaults(func=cmd_calendar, calendar_command="upcoming")

    calendar_parser.set_defaults(func=cmd_calendar, calendar_command="upcoming")

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
