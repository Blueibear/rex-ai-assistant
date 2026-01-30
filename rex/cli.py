"""
Command-line interface for Rex AI Assistant.

This module provides the main CLI entry point with subcommands:
    rex doctor       - Run environment diagnostics
    rex chat         - Start interactive chat (default)
    rex version      - Show version information
    rex tools        - List registered tools and their status
    rex run-workflow - Run a workflow from a JSON file
    rex approvals    - List and manage pending approvals
    rex workflows    - List workflows
    rex memory       - Manage working and long-term memory
    rex kb           - Manage knowledge base documents
    rex scheduler    - List and manage scheduler jobs
    rex email        - Manage email
    rex calendar     - Manage calendar
    rex reminders    - Manage reminders
    rex cues         - Manage follow-up cues
    rex browser      - Browser automation
    rex os           - OS automation
    rex gh           - GitHub integration
    rex code         - VS Code operations
    rex msg          - Messaging (SMS)
    rex notify       - Notifications

Usage:
    rex [command] [options]

If no command is specified, the chat interface is started.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from datetime import datetime, timedelta


def get_browser_service():
    from rex.browser_automation import get_browser_service as _get_browser_service

    return _get_browser_service()


def get_os_service():
    from rex.os_automation import get_os_service as _get_os_service

    return _get_os_service()


def get_github_service():
    from rex.github_service import get_github_service as _get_github_service

    return _get_github_service()


def get_vscode_service():
    from rex.vscode_service import get_vscode_service as _get_vscode_service

    return _get_vscode_service()


def get_scheduler():
    from rex.scheduler import get_scheduler as _get_scheduler

    return _get_scheduler()


def get_email_service():
    from rex.email_service import get_email_service as _get_email_service

    return _get_email_service()


def get_calendar_service():
    from rex.calendar_service import get_calendar_service as _get_calendar_service

    return _get_calendar_service()


def get_reminder_service():
    from rex.reminder_service import get_reminder_service as _get_reminder_service

    return _get_reminder_service()


def get_cue_store():
    from rex.cue_store import get_cue_store as _get_cue_store

    return _get_cue_store()


def initialize_scheduler_system(*args, **kwargs):
    from rex.integrations import initialize_scheduler_system as _initialize_scheduler_system

    return _initialize_scheduler_system(*args, **kwargs)


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
    import asyncio

    from rex import settings
    from rex.assistant import Assistant
    from rex.logging_utils import configure_logging
    from rex.plugins import load_plugins, shutdown_plugins
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
        persisted = Workflow.load(workflow.workflow_id)
        if persisted is not None:
            workflow = persisted
            runner = WorkflowRunner(workflow)

        if workflow.status != "blocked":
            print(f"Error: Cannot resume workflow in status '{workflow.status}'")
            print("Only 'blocked' workflows can be resumed.")
            return 1

        print("Resuming blocked workflow...")
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
    print("Workflow finished")
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
    from rex.workflow import WorkflowApproval
    from rex.workflow_runner import approve_workflow, deny_workflow, list_pending_approvals

    if args.approve:
        approval_id = args.approve
        if approve_workflow(approval_id, decided_by="cli_user", reason=args.reason):
            print(f"Approved: {approval_id}")
            return 0
        print(f"Error: Approval not found: {approval_id}")
        return 1

    if args.deny:
        approval_id = args.deny
        reason = args.reason or "Denied via CLI"
        if deny_workflow(approval_id, decided_by="cli_user", reason=reason):
            print(f"Denied: {approval_id}")
            return 0
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


def cmd_plan(args: argparse.Namespace) -> int:
    """Generate a workflow plan from a high-level goal."""
    from rex.autonomy_modes import AutonomyMode, get_mode
    from rex.executor import ExecutionBudget, Executor
    from rex.planner import Planner, UnableToPlanError
    from rex.policy_engine import get_policy_engine
    from rex.tool_registry import get_tool_registry

    goal = args.goal
    print(f"Planning workflow for goal: {goal}")
    print("-" * 60)

    planner = Planner(
        tool_registry=get_tool_registry(),
        policy_engine=get_policy_engine(),
    )

    try:
        workflow = planner.plan(goal, requested_by="cli_user")
    except UnableToPlanError as e:
        print(f"Error: {e}")
        return 1

    print(f"Generated workflow: {workflow.title}")
    print(f"  ID: {workflow.workflow_id}")
    print(f"  Steps: {len(workflow.steps)}")
    print()

    for i, step in enumerate(workflow.steps, 1):
        print(f"{i}. {step.description}")
        if step.tool_call:
            print(f"   Tool: {step.tool_call.tool}")
            print(f"   Args: {step.tool_call.args}")
        if step.requires_approval:
            print("   [REQUIRES APPROVAL]")
        print()

    print("Validating workflow...")
    if not planner.validate_workflow(workflow):
        print("Error: Workflow validation failed.")
        print(
            "The workflow contains steps that cannot be executed (missing tools or policy denials)."
        )
        return 1

    print("Validation passed.")
    print()

    autonomy_mode = get_mode(workflow)
    print(f"Autonomy mode: {autonomy_mode.value}")
    print()

    if args.save or args.execute:
        workflow.save()
        print(f"Saved workflow to: data/workflows/{workflow.workflow_id}.json")
        print()

    if args.execute:
        if autonomy_mode == AutonomyMode.OFF:
            print("Autonomy mode is OFF for this workflow category.")
            print("Manual execution is required.")
            if not args.force:
                print("Use --force to execute anyway.")
                return 0

        budget = ExecutionBudget(
            max_actions=args.max_actions,
            max_messages=args.max_messages,
            max_time_seconds=args.max_time,
        )

        print(f"Executing workflow with budget: {budget}")
        print("-" * 60)

        executor = Executor(workflow, budget)
        result = executor.run()

        print()
        print("-" * 60)
        print("Execution complete")
        print(result)

        if result.status == "completed":
            return 0
        if result.status == "blocked":
            print()
            print("To approve, run:")
            print(f"  rex approvals --approve {result.blocking_approval_id}")
            print()
            print("Then resume with:")
            print(f"  rex executor resume {workflow.workflow_id}")
            return 0
        return 1

    print("Workflow planned successfully.")
    print()
    print("To execute, run:")
    print(f'  rex plan "{goal}" --execute')
    print()
    print("Or run the workflow file:")
    print(f"  rex run-workflow data/workflows/{workflow.workflow_id}.json")
    return 0


def cmd_executor_resume(args: argparse.Namespace) -> int:
    """Resume a blocked executor workflow."""
    from rex.executor import ExecutionBudget, Executor
    from rex.workflow import Workflow

    workflow_id = args.workflow_id

    workflow = Workflow.load(workflow_id)
    if workflow is None:
        print(f"Error: Workflow not found: {workflow_id}")
        return 1

    if workflow.status != "blocked":
        print(f"Error: Cannot resume workflow in status '{workflow.status}'")
        print("Only 'blocked' workflows can be resumed.")
        return 1

    print(f"Resuming workflow: {workflow.title}")
    print(f"  ID: {workflow.workflow_id}")
    print(f"  Blocking approval: {workflow.blocking_approval_id}")
    print()

    budget = ExecutionBudget(
        max_actions=args.max_actions,
        max_messages=args.max_messages,
        max_time_seconds=args.max_time,
    )

    print(f"Executing with budget: {budget}")
    print("-" * 60)

    executor = Executor(workflow, budget)
    result = executor.run()

    print()
    print("-" * 60)
    print("Execution complete")
    print(result)

    if result.status == "completed":
        return 0
    if result.status == "blocked":
        print()
        print("To approve, run:")
        print(f"  rex approvals --approve {result.blocking_approval_id}")
        print()
        print("Then resume with:")
        print(f"  rex executor resume {workflow.workflow_id}")
        return 0
    return 1


def cmd_memory(args: argparse.Namespace) -> int:
    """Manage working and long-term memory."""
    import json

    from rex.memory import get_long_term_memory, get_working_memory

    subcommand = args.memory_command

    if subcommand == "recent":
        wm = get_working_memory()
        n = int(getattr(args, "count", 10))
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

    if subcommand == "add":
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

    if subcommand == "search":
        ltm = get_long_term_memory()
        keyword = args.keyword
        category = args.category
        show_sensitive = args.show_sensitive

        results = ltm.search(
            category=category,
            keyword=keyword,
            include_sensitive=True,
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

    if subcommand == "forget":
        ltm = get_long_term_memory()
        entry_id = args.entry_id
        if ltm.forget(entry_id):
            print(f"Deleted memory entry: {entry_id}")
            return 0
        print(f"Error: Entry not found: {entry_id}")
        return 1

    if subcommand == "clear":
        wm = get_working_memory()
        count = len(wm)
        wm.clear()
        print(f"Cleared {count} working memory entries.")
        return 0

    if subcommand == "retention":
        ltm = get_long_term_memory()
        deleted = ltm.run_retention_policy()
        print(f"Retention policy deleted {deleted} expired entries.")
        return 0

    if subcommand == "stats":
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

    print("Unknown memory subcommand. Use 'rex memory --help'")
    return 1


def cmd_kb(args: argparse.Namespace) -> int:
    """Manage knowledge base."""
    from rex.knowledge_base import get_knowledge_base

    kb = get_knowledge_base()
    subcommand = args.kb_command

    if subcommand == "ingest":
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

    if subcommand == "search":
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
            if getattr(args, "verbose", False):
                snippet = doc.content[:200].replace("\n", " ")
                if len(doc.content) > 200:
                    snippet += "..."
                print(f"  Snippet: {snippet}")
            print()

        print(f"Total: {len(results)} documents found")
        return 0

    if subcommand == "list":
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

    if subcommand == "show":
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

    if subcommand == "delete":
        doc_id = args.doc_id
        if kb.delete_document(doc_id):
            print(f"Deleted document: {doc_id}")
            return 0
        print(f"Error: Document not found: {doc_id}")
        return 1

    if subcommand == "cite":
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

    if subcommand == "tags":
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

    print("Unknown kb subcommand. Use 'rex kb --help'")
    return 1


def cmd_scheduler(args: argparse.Namespace) -> int:
    """Manage scheduler jobs."""
    scheduler = get_scheduler()
    subcommand = args.scheduler_command

    if subcommand == "list":
        jobs = scheduler.list_jobs()
        if not jobs:
            print("No scheduled jobs.")
            return 0

        print("Scheduled Jobs")
        print("=" * 80)
        print()

        for job in jobs:
            status = "enabled" if job.enabled else "disabled"
            next_run = getattr(job, "next_run", None)
            next_run_str = next_run.strftime("%Y-%m-%d %H:%M:%S") if next_run else "n/a"

            print(f"{job.job_id}: {job.name} [{status}]")
            if hasattr(job, "schedule"):
                print(f"  Schedule: {job.schedule}")
            print(f"  Next run: {next_run_str}")
            if hasattr(job, "run_count"):
                print(f"  Run count: {job.run_count}", end="")
                if getattr(job, "max_runs", None):
                    print(f" / {job.max_runs}")
                else:
                    print(" (unlimited)")
            if getattr(args, "verbose", False):
                callback_name = getattr(job, "callback_name", None)
                workflow_id = getattr(job, "workflow_id", None)
                if callback_name:
                    print(f"  Callback: {callback_name}")
                if workflow_id:
                    print(f"  Workflow: {workflow_id}")
            print()

        print(f"Total: {len(jobs)} jobs")
        return 0

    if subcommand == "run":
        job_id = args.job_id
        initialize_scheduler_system(start_scheduler=False)

        if scheduler.run_job(job_id, force=True):
            print(f"Job {job_id} executed successfully")
            return 0
        print(f"Error: Failed to run job {job_id}")
        return 1

    if subcommand == "init":
        initialize_scheduler_system(start_scheduler=False)
        print("Scheduler system initialized with default jobs")
        print("Use 'rex scheduler list' to see registered jobs")
        return 0

    print("Unknown scheduler subcommand. Use 'rex scheduler --help'")
    return 1


def cmd_email(args: argparse.Namespace) -> int:
    """Manage email."""
    email_service = get_email_service()
    subcommand = args.email_command

    if subcommand == "unread":
        if not email_service.connected:
            if not email_service.connect():
                print("Error: Failed to connect to email service")
                return 1

        limit = getattr(args, "limit", None) or 10
        unread = email_service.fetch_unread(limit=limit)
        print("Unread Email Summary")
        print("=" * 80)
        print()

        if not unread:
            print("No unread emails.")
            return 0

        for email in unread:
            category = email_service.categorize(email)
            importance = "!! " if getattr(email, "importance_score", 0.0) >= 0.8 else ""
            print(f"{importance}{email.id}: {email.subject}")
            print(f"  From: {email.from_addr}")
            print(f"  Received: {email.received_at.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"  Category: {category}")
            if getattr(args, "verbose", False):
                score = getattr(email, "importance_score", None)
                if score is not None:
                    print(f"  Importance: {score:.2f}")
                print(f"  Snippet: {email.snippet}")
            print()

        print(f"Total: {len(unread)} unread emails")
        return 0

    print("Unknown email subcommand. Use 'rex email --help'")
    return 1


def cmd_calendar(args: argparse.Namespace) -> int:
    """Manage calendar."""
    calendar_service = get_calendar_service()
    subcommand = args.calendar_command

    if subcommand == "upcoming":
        if not calendar_service.connected:
            if not calendar_service.connect():
                print("Error: Failed to connect to calendar service")
                return 1

        days = getattr(args, "days", None) or 7
        events = calendar_service.get_upcoming_events(days=days)

        print(f"Upcoming Events (next {days} days)")
        print("=" * 80)
        print()

        if not events:
            print(f"No upcoming events in the next {days} days.")
            return 0

        for event in events:
            if getattr(event, "all_day", False):
                time_str = event.start_time.strftime("%Y-%m-%d") + " (All day)"
            else:
                time_str = (
                    event.start_time.strftime("%Y-%m-%d %H:%M")
                    + " - "
                    + event.end_time.strftime("%H:%M")
                )

            print(f"{event.id}: {event.title}")
            print(f"  When: {time_str}")
            if event.location:
                print(f"  Location: {event.location}")
            if getattr(event, "attendees", None):
                if event.attendees:
                    print(f"  Attendees: {', '.join(event.attendees)}")
            if getattr(args, "verbose", False) and getattr(event, "description", None):
                print(f"  Description: {event.description}")
            print()

        print(f"Total: {len(events)} events")

        if getattr(args, "conflicts", False):
            conflicts = calendar_service.find_conflicts(events)
            if conflicts:
                print()
                print("Conflicts Detected:")
                print("-" * 80)
                for event1, event2 in conflicts:
                    print(f"!! '{event1.title}' overlaps with '{event2.title}'")

        return 0

    print("Unknown calendar subcommand. Use 'rex calendar --help'")
    return 1


def _parse_ttl(ttl_str: str) -> timedelta | None:
    """Parse TTL string like '7d', '24h', '30m', '1w', '10s' into timedelta."""
    ttl_str = ttl_str.strip().lower()
    if not ttl_str:
        return None

    try:
        if ttl_str.endswith("w"):
            return timedelta(weeks=int(ttl_str[:-1]))
        if ttl_str.endswith("d"):
            return timedelta(days=int(ttl_str[:-1]))
        if ttl_str.endswith("h"):
            return timedelta(hours=int(ttl_str[:-1]))
        if ttl_str.endswith("m"):
            return timedelta(minutes=int(ttl_str[:-1]))
        if ttl_str.endswith("s"):
            return timedelta(seconds=int(ttl_str[:-1]))
        return timedelta(days=int(ttl_str))
    except ValueError:
        return None


def _parse_datetime_strict(dt_str: str) -> datetime:
    """
    Parse a datetime string in common formats and return a timezone-aware datetime.

    Accepted examples:
      - 2026-01-29 14:30
      - 2026-01-29 14:30:00
      - 2026/01/29 14:30
      - ISO-8601 (datetime.fromisoformat compatible)

    If no timezone is provided, local timezone is assumed.
    """
    dt_str = dt_str.strip()
    if not dt_str:
        raise ValueError("Datetime cannot be empty")

    # First, try ISO parsing (supports "YYYY-MM-DD HH:MM[:SS]" and true ISO forms)
    try:
        dt = datetime.fromisoformat(dt_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=datetime.now().astimezone().tzinfo)
        return dt
    except ValueError:
        pass

    # Then, try a few explicit formats
    formats = [
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d %H:%M",
        "%Y/%m/%d %H:%M:%S",
        "%m/%d/%Y %H:%M",
        "%d/%m/%Y %H:%M",
    ]

    for fmt in formats:
        try:
            dt = datetime.strptime(dt_str, fmt)
            return dt.replace(tzinfo=datetime.now().astimezone().tzinfo)
        except ValueError:
            continue

    raise ValueError("Invalid datetime format. Use YYYY-MM-DD HH:MM (or ISO-8601).")


def cmd_reminders(args: argparse.Namespace) -> int:
    """Manage reminders."""
    service = get_reminder_service()
    subcommand = args.reminders_command

    if subcommand == "add":
        title = args.title
        at_str = args.at

        try:
            remind_at = _parse_datetime_strict(at_str)
        except ValueError as exc:
            print(f"Error: {exc}")
            return 1

        followup = bool(getattr(args, "followup", False))

        try:
            from rex.config_manager import load_config

            config = load_config()
            user_id = config.get("runtime", {}).get("user_id", "default")
        except Exception:
            user_id = "default"

        # Compatibility: service might implement create_reminder(...) or add_reminder(...)
        if hasattr(service, "create_reminder"):
            reminder = service.create_reminder(
                user_id=user_id,
                title=title,
                remind_at=remind_at,
                followup_enabled=followup,
            )
        else:
            reminder = service.add_reminder(title, remind_at, follow_up=followup)

        reminder_id = getattr(reminder, "reminder_id", getattr(reminder, "id", "unknown"))
        remind_at_val = getattr(reminder, "remind_at", getattr(reminder, "remind_at", remind_at))
        if hasattr(remind_at_val, "isoformat"):
            remind_at_str = remind_at_val.isoformat()
        else:
            remind_at_str = str(remind_at_val)

        print(f"Created reminder: {reminder_id}")
        print(f"  Title: {getattr(reminder, 'title', title)}")
        print(f"  Remind at: {remind_at_str}")
        print(f"  Follow-up enabled: {getattr(reminder, 'followup_enabled', followup)}")
        return 0

    if subcommand == "list":
        status_filter = getattr(args, "status", None)
        reminders = service.list_reminders(status=status_filter)

        if not reminders:
            print("No reminders found.")
            return 0

        print("Reminders")
        print("=" * 60)
        print()

        for rem in reminders:
            status_indicator = {
                "pending": "[PENDING]",
                "fired": "[FIRED]",
                "done": "[DONE]",
                "canceled": "[CANCELED]",
            }.get(getattr(rem, "status", getattr(rem, "status", "unknown")), "[?]")

            rid = getattr(rem, "reminder_id", getattr(rem, "id", "unknown"))
            title = getattr(rem, "title", "")
            remind_at_dt = getattr(rem, "remind_at", getattr(rem, "remind_at", None))
            remind_at_str = remind_at_dt.strftime("%Y-%m-%d %H:%M") if remind_at_dt else "n/a"

            print(f"{rid}: {title} {status_indicator}")
            print(f"  Remind at: {remind_at_str}")

            if getattr(rem, "followup_enabled", False):
                print("  Follow-up: enabled")

            fired_at = getattr(rem, "fired_at", None)
            done_at = getattr(rem, "done_at", None)
            if fired_at:
                print(f"  Fired at: {fired_at.strftime('%Y-%m-%d %H:%M')}")
            if done_at:
                print(f"  Done at: {done_at.strftime('%Y-%m-%d %H:%M')}")
            print()

        print(f"Total: {len(reminders)} reminders")
        return 0

    if subcommand == "done":
        reminder_id = getattr(args, "reminder_id", None) or getattr(args, "id", None)
        if hasattr(service, "mark_done"):
            ok = service.mark_done(reminder_id)
        else:
            ok = service.complete_reminder(reminder_id)
        if ok:
            print(f"Marked reminder {reminder_id} as done.")
            return 0
        print(f"Error: Reminder not found: {reminder_id}")
        return 1

    if subcommand == "cancel":
        reminder_id = getattr(args, "reminder_id", None) or getattr(args, "id", None)
        if hasattr(service, "cancel_reminder"):
            ok = service.cancel_reminder(reminder_id)
        else:
            ok = service.cancel(reminder_id)
        if ok:
            print(f"Canceled reminder {reminder_id}.")
            return 0
        print(f"Error: Reminder not found: {reminder_id}")
        return 1

    print("Unknown reminders subcommand. Use 'rex reminders --help'")
    return 1


def cmd_cues(args: argparse.Namespace) -> int:
    """Manage follow-up cues."""
    store = get_cue_store()
    subcommand = args.cues_command

    if subcommand == "list":
        status_filter = getattr(args, "status", None)

        if hasattr(store, "list_all_cues"):
            cues = store.list_all_cues(status=status_filter)
        else:
            cues = store.list_cues(status=status_filter)

        if not cues:
            print("No cues found.")
            return 0

        print("Follow-up Cues")
        print("=" * 60)
        print()

        for cue in cues:
            status_indicator = {
                "pending": "[PENDING]",
                "asked": "[ASKED]",
                "dismissed": "[DISMISSED]",
            }.get(getattr(cue, "status", "unknown"), "[?]")

            source_type = getattr(cue, "source_type", None)
            source_label = f"[{source_type}]" if source_type else ""
            cue_id = getattr(cue, "cue_id", getattr(cue, "id", "unknown"))
            title = getattr(cue, "title", "(no title)")
            prompt = getattr(cue, "prompt", "")

            print(f"{cue_id}: {title} {source_label} {status_indicator}".strip())
            print(f"  Prompt: {prompt}")

            created_at = getattr(cue, "created_at", None)
            expires_at = getattr(cue, "expires_at", None)
            asked_at = getattr(cue, "asked_at", None)
            dismissed_at = getattr(cue, "dismissed_at", None)

            if created_at:
                print(f"  Created: {created_at.strftime('%Y-%m-%d %H:%M')}")
            if expires_at:
                print(f"  Expires: {expires_at.strftime('%Y-%m-%d %H:%M')}")
            if asked_at:
                print(f"  Asked at: {asked_at.strftime('%Y-%m-%d %H:%M')}")
            if dismissed_at:
                print(f"  Dismissed at: {dismissed_at.strftime('%Y-%m-%d %H:%M')}")
            print()

        print(f"Total: {len(cues)} cues")

        if hasattr(store, "stats"):
            stats = store.stats()
            by_status = stats.get("by_status", {}) if isinstance(stats, dict) else {}
            print(f"  Pending: {by_status.get('pending', 0)}")
            print(f"  Asked: {by_status.get('asked', 0)}")
            print(f"  Dismissed: {by_status.get('dismissed', 0)}")

        return 0

    if subcommand == "dismiss":
        cue_id = args.cue_id
        ok = store.dismiss(cue_id) if hasattr(store, "dismiss") else False
        if ok:
            print(f"Dismissed cue {cue_id}.")
            return 0
        print(f"Error: Cue not found: {cue_id}")
        return 1

    if subcommand == "prune":
        # Compatibility: store may implement prune_expired() or prune_expired(expire_hours=...)
        if hasattr(store, "prune_expired"):
            try:
                count = store.prune_expired()
            except TypeError:
                from rex import settings

                expire_hours = int(getattr(settings, "followups_expire_hours", 168))
                count = store.prune_expired(expire_hours=expire_hours)
        else:
            count = 0
        print(f"Pruned {count} expired cue(s).")
        return 0

    print("Unknown cues subcommand. Use 'rex cues --help'")
    return 1


def cmd_browser(args: argparse.Namespace) -> int:
    """Manage browser automation."""
    import json
    from pathlib import Path

    from rex.browser_automation import run_browser_script_sync

    subcommand = args.browser_command

    if subcommand == "run":
        script_path = Path(args.script)
        if not script_path.exists():
            print(f"Error: Script file not found: {script_path}")
            return 1

        try:
            with open(script_path, encoding="utf-8") as f:
                script_data = json.load(f)

            steps = script_data.get("steps", [])
            headless = not args.headed

            print(f"Running browser script: {script_path}")
            print(f"  Steps: {len(steps)}")
            print(f"  Headless: {headless}")
            print()

            results = run_browser_script_sync(steps, headless=headless)

            for result in results:
                step_num = result.get("step", "?")
                action = result.get("action", "unknown")
                status = result.get("status", "unknown")

                if status == "success":
                    print(f"✓ Step {step_num}: {action}")
                elif status == "error":
                    print(f"✗ Step {step_num}: {action} - {result.get('error', 'Unknown error')}")
                else:
                    print(f"⊘ Step {step_num}: {action} - {status}")

            print()
            success_count = sum(1 for r in results if r.get("status") == "success")
            print(f"Completed: {success_count}/{len(results)} steps successful")

            return 0 if success_count == len(results) else 1

        except Exception as e:
            print(f"Error: {e}")
            return 1

    if subcommand == "sessions":
        service = get_browser_service()
        sessions = service.list_sessions()

        if not sessions:
            print("No browser sessions found.")
            return 0

        print("Browser Sessions")
        print("=" * 60)
        print()

        for session in sessions:
            print(f"  {session}")

        print()
        print(f"Total: {len(sessions)} sessions")
        return 0

    if subcommand == "screenshots":
        service = get_browser_service()
        screenshots = service.list_screenshots()

        if not screenshots:
            print("No screenshots found.")
            return 0

        print("Screenshots")
        print("=" * 60)
        print()

        for screenshot in screenshots:
            print(f"  {screenshot}")

        print()
        print(f"Total: {len(screenshots)} screenshots")
        return 0

    print("Unknown browser subcommand. Use 'rex browser --help'")
    return 1


def cmd_os(args: argparse.Namespace) -> int:
    """Manage OS automation."""
    subcommand = args.os_command
    service = get_os_service()

    if subcommand == "run":
        command = args.command.split()

        try:
            print(f"Running command: {' '.join(command)}")
            print()

            result = service.run_command(command)

            if result.stdout:
                print(result.stdout)

            if result.stderr:
                print(f"[stderr] {result.stderr}", file=sys.stderr)

            print()
            print(f"Exit code: {result.returncode}")
            print(f"Duration: {result.duration_ms}ms")

            return result.returncode

        except PermissionError as e:
            print(f"Error: {e}")
            return 1
        except Exception as e:
            print(f"Error: {e}")
            return 1

    if subcommand == "copy":
        try:
            result = service.copy_file(args.src, args.dst)
            print(f"Copied: {result['src']} -> {result['dst']}")
            print(f"Size: {result['size']} bytes")
            return 0
        except Exception as e:
            print(f"Error: {e}")
            return 1

    if subcommand == "move":
        try:
            result = service.move_file(args.src, args.dst)
            print(f"Moved: {result['src']} -> {result['dst']}")
            return 0
        except Exception as e:
            print(f"Error: {e}")
            return 1

    if subcommand == "delete":
        try:
            backup = not args.permanent
            result = service.delete_file(args.path, backup=backup)

            if result["action"] == "moved_to_trash":
                print(f"Moved to trash: {result['path']}")
                print(f"Backup location: {result['backup_path']}")
            else:
                print(f"Permanently deleted: {result['path']}")

            return 0
        except Exception as e:
            print(f"Error: {e}")
            return 1

    if subcommand == "trash":
        files = service.list_trash()

        if not files:
            print("Trash is empty.")
            return 0

        print("Trash Contents")
        print("=" * 80)
        print()

        for file in files:
            print(f"{file['name']}")
            print(f"  Path: {file['path']}")
            print(f"  Size: {file['size']} bytes")
            print(f"  Modified: {file['modified']}")
            print()

        print(f"Total: {len(files)} files")
        return 0

    print("Unknown os subcommand. Use 'rex os --help'")
    return 1


def cmd_gh(args: argparse.Namespace) -> int:
    """Manage GitHub integration."""
    subcommand = args.gh_command
    service = get_github_service()

    if subcommand == "repos":
        try:
            repos = service.list_repos()

            if not repos:
                print("No repositories found.")
                return 0

            print("GitHub Repositories")
            print("=" * 80)
            print()

            for repo in repos:
                visibility = "private" if repo.private else "public"
                print(f"{repo.full_name} ({visibility})")
                if repo.description:
                    print(f"  {repo.description}")
                print(f"  URL: {repo.url}")
                print(f"  Default branch: {repo.default_branch}")
                print()

            print(f"Total: {len(repos)} repositories")
            return 0

        except Exception as e:
            print(f"Error: {e}")
            return 1

    if subcommand == "prs":
        try:
            prs = service.list_prs(args.repo, state=args.state)

            if not prs:
                print(f"No pull requests found ({args.state}).")
                return 0

            print(f"Pull Requests for {args.repo}")
            print("=" * 80)
            print()

            for pr in prs:
                print(f"#{pr.number}: {pr.title}")
                print(f"  State: {pr.state}")
                print(f"  Author: {pr.author}")
                print(f"  Branch: {pr.head_branch} -> {pr.base_branch}")
                print(f"  URL: {pr.url}")
                print(f"  Created: {pr.created_at}")
                print()

            print(f"Total: {len(prs)} pull requests")
            return 0

        except Exception as e:
            print(f"Error: {e}")
            return 1

    if subcommand == "issue-create":
        try:
            labels = args.labels.split(",") if args.labels else None
            issue = service.create_issue(args.repo, args.title, args.body, labels)

            print("Issue created successfully!")
            print(f"  Number: #{issue.number}")
            print(f"  Title: {issue.title}")
            print(f"  URL: {issue.url}")

            return 0

        except Exception as e:
            print(f"Error: {e}")
            return 1

    if subcommand == "pr-create":
        try:
            pr = service.create_pr(
                args.repo,
                args.head,
                args.base,
                args.title,
                args.body,
            )

            print("Pull request created successfully!")
            print(f"  Number: #{pr.number}")
            print(f"  Title: {pr.title}")
            print(f"  URL: {pr.url}")

            return 0

        except Exception as e:
            print(f"Error: {e}")
            return 1

    print("Unknown gh subcommand. Use 'rex gh --help'")
    return 1


def cmd_code(args: argparse.Namespace) -> int:
    """Manage VS Code operations."""
    subcommand = args.code_command
    service = get_vscode_service()

    if subcommand == "open":
        try:
            result = service.open_file(args.path)

            print(f"File: {result['path']}")
            print(f"Lines: {result['lines']}")
            print(f"Size: {result['size']} bytes")
            print(f"Modified: {result['modified']}")
            print()
            print("-" * 80)
            print(result["content"])
            print("-" * 80)

            return 0

        except Exception as e:
            print(f"Error: {e}")
            return 1

    if subcommand == "patch":
        try:
            from pathlib import Path

            patch_file = Path(args.patch_file)
            if not patch_file.exists():
                print(f"Error: Patch file not found: {patch_file}")
                return 1

            with open(patch_file, encoding="utf-8") as f:
                patch_content = f.read()

            result = service.apply_patch(args.path, patch_content)

            if result.success:
                print("Patch applied successfully!")
                print(f"  File: {result.file_path}")
                print(f"  Hunks applied: {result.hunks_applied}")
            else:
                print("Patch application failed!")
                print(f"  File: {result.file_path}")
                print(f"  Hunks applied: {result.hunks_applied}")
                print(f"  Hunks failed: {result.hunks_failed}")
                if result.errors:
                    print("  Errors:")
                    for error in result.errors:
                        print(f"    - {error}")
                return 1

            return 0

        except Exception as e:
            print(f"Error: {e}")
            return 1

    if subcommand == "test":
        try:
            result = service.run_tests(
                test_path=args.path,
                pattern=args.pattern,
                verbose=args.verbose,
            )

            print("Test Results")
            print("=" * 80)
            print()

            if result.success:
                print("✓ All tests passed!")
            else:
                print("✗ Some tests failed")

            print()
            print(f"Total: {result.total}")
            print(f"Passed: {result.passed}")
            print(f"Failed: {result.failed}")
            print(f"Errors: {result.errors}")
            print(f"Skipped: {result.skipped}")
            print(f"Duration: {result.duration_seconds:.2f}s")

            if args.verbose:
                print()
                print("Output:")
                print("-" * 80)
                print(result.output)

            return 0 if result.success else 1

        except Exception as e:
            print(f"Error: {e}")
            return 1

    print("Unknown code subcommand. Use 'rex code --help'")
    return 1


def cmd_msg(args: argparse.Namespace) -> int:
    """Manage messaging."""
    from rex.messaging_service import Message, get_sms_service

    subcommand = args.msg_command

    if subcommand == "send":
        channel = args.channel.lower()

        if channel == "sms":
            sms_service = get_sms_service()
            message = Message(
                channel="sms",
                to=args.to,
                from_=sms_service.from_number,
                body=args.body,
            )
            sent = sms_service.send(message)
            print("Message sent successfully")
            print(f"  ID: {sent.id}")
            print(f"  To: {sent.to}")
            print(f"  Thread: {sent.thread_id}")
            print(f"  Timestamp: {sent.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
            return 0

        print(f"Error: Unsupported channel '{channel}'. Currently only 'sms' is supported.")
        return 1

    if subcommand == "receive":
        channel = args.channel.lower()

        if channel == "sms":
            sms_service = get_sms_service()
            messages = sms_service.receive(limit=args.limit)

            if not messages:
                print("No messages received.")
                return 0

            print("Recent Messages")
            print("=" * 80)
            print()

            for msg in messages:
                preview = (msg.body[:50] + "...") if len(msg.body) > 50 else msg.body
                print(f"{msg.id}: {preview}")
                print(f"  From: {msg.from_}")
                print(f"  To: {msg.to}")
                print(f"  Received: {msg.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"  Thread: {msg.thread_id}")
                print()

            print(f"Total: {len(messages)} messages")
            return 0

        print(f"Error: Unsupported channel '{channel}'. Currently only 'sms' is supported.")
        return 1

    print("Unknown messaging subcommand. Use 'rex msg --help'")
    return 1


def cmd_notify(args: argparse.Namespace) -> int:
    """Manage notifications."""
    from rex.notification import NotificationRequest, get_escalation_manager, get_notifier

    notifier = get_notifier()
    escalation_manager = get_escalation_manager()
    subcommand = args.notify_command

    if subcommand == "send":
        if args.channels:
            channel_list = [ch.strip() for ch in args.channels.split(",")]
        else:
            channel_list = ["dashboard"]

        notification = NotificationRequest(
            priority=args.priority,
            title=args.title,
            body=args.body,
            channel_preferences=channel_list,
        )

        notifier.send(notification)

        if notification.priority == "urgent":
            next_channel = channel_list[1] if len(channel_list) > 1 else "email"
            escalation_manager.track_notification(notification, next_channel)

        print("Notification sent successfully")
        print(f"  ID: {notification.id}")
        print(f"  Priority: {notification.priority}")
        print(f"  Channels: {', '.join(channel_list)}")
        print(f"  Title: {notification.title}")
        return 0

    if subcommand == "list-digests":
        digests = notifier.list_digests()

        if not digests or all(len(q) == 0 for q in digests.values()):
            print("No queued digest notifications.")
            return 0

        print("Queued Digest Notifications")
        print("=" * 80)
        print()

        for channel, notifications in digests.items():
            if not notifications:
                continue

            print(f"Channel: {channel}")
            print(f"  Count: {len(notifications)}")
            print()

            for notif in notifications:
                print(f"  - {notif['id']}: {notif['title']}")
                print(f"    Created: {notif['timestamp']}")
                body_preview = (
                    notif["body"][:60] + "..." if len(notif["body"]) > 60 else notif["body"]
                )
                print(f"    Body: {body_preview}")
                print()

        total_count = sum(len(q) for q in digests.values())
        print(f"Total: {total_count} queued notifications across {len(digests)} channels")
        return 0

    if subcommand == "flush-digests":
        channel = args.channel
        count = notifier.flush_digests(channel=channel)

        if count == 0:
            print("No digest notifications to flush.")
        else:
            if channel:
                print(f"Flushed digest queue for channel: {channel}")
            else:
                print(f"Flushed {count} digest queue(s)")
        return 0

    if subcommand == "ack":
        notification_id = args.notification_id
        if escalation_manager.acknowledge(notification_id):
            print(f"Acknowledged notification: {notification_id}")
            return 0
        print(f"Notification not found or already acknowledged: {notification_id}")
        return 1

    print("Unknown notification subcommand. Use 'rex notify --help'")
    return 1


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser with all subcommands."""
    parser = argparse.ArgumentParser(
        prog="rex",
        description="Rex AI Assistant - Voice-activated AI assistant with speech recognition and synthesis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  rex doctor
  rex doctor -v
  rex chat
  rex version
  rex tools
  rex tools -v
  rex run-workflow workflow.json
  rex run-workflow workflow.json --dry-run
  rex run-workflow workflow.json --resume
  rex approvals
  rex approvals --approve <id>
  rex workflows

Planning and execution commands:
  rex plan "send monthly newsletter"
  rex plan "check weather in Dallas" --execute
  rex plan "turn on living room lights" --save
  rex executor resume <workflow_id>

Memory commands:
  rex memory recent 5
  rex memory add facts '{"key":"value"}'
  rex memory add secrets '{"api":"key"}' --sensitive --ttl=7d
  rex memory search keyword
  rex memory search --category preferences
  rex memory forget <entry_id>
  rex memory stats

Knowledge base commands:
  rex kb ingest /path/to/file.txt --title "My Doc" --tags notes,project
  rex kb search "query"
  rex kb search "query" -v
  rex kb list
  rex kb show <doc_id>
  rex kb delete <doc_id>
  rex kb cite "phrase"
  rex kb tags

Scheduler commands:
  rex scheduler init
  rex scheduler list
  rex scheduler run <job_id>

Email commands:
  rex email unread
  rex email unread --limit 5
  rex email unread -v

Calendar commands:
  rex calendar upcoming
  rex calendar upcoming --days 14
  rex calendar upcoming --conflicts

For more information, visit: https://github.com/Blueibear/rex-ai-assistant
""",
    )

    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--version", action="version", version=f"%(prog)s {_get_version()}")

    subparsers = parser.add_subparsers(
        title="commands",
        description="Available commands",
        dest="command",
        metavar="COMMAND",
    )

    # doctor
    doctor_parser = subparsers.add_parser(
        "doctor",
        help="Run environment diagnostics",
        description="Check Python version, config files, environment variables, and external dependencies.",
    )
    doctor_parser.add_argument(
        "-v", "--verbose", action="store_true", help="Show detailed diagnostic information"
    )
    doctor_parser.set_defaults(func=cmd_doctor)

    # chat
    chat_parser = subparsers.add_parser(
        "chat",
        help="Start interactive chat (default)",
        description="Start an interactive text chat session with Rex.",
    )
    chat_parser.set_defaults(func=cmd_chat)

    # version
    version_parser = subparsers.add_parser(
        "version",
        help="Show version information",
        description="Display Rex version and optionally dependency versions.",
    )
    version_parser.add_argument(
        "-v", "--verbose", action="store_true", help="Show dependency versions"
    )
    version_parser.set_defaults(func=cmd_version)

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
    tools_parser.set_defaults(func=cmd_tools)

    # run-workflow
    workflow_parser = subparsers.add_parser(
        "run-workflow",
        help="Run a workflow from a JSON file",
        description="Load and execute a workflow definition. Supports dry-run and resume modes.",
    )
    workflow_parser.add_argument("workflow", type=str, help="Path to the workflow JSON file")
    workflow_parser.add_argument(
        "--dry-run", action="store_true", help="Preview workflow without executing"
    )
    workflow_parser.add_argument(
        "--resume", action="store_true", help="Resume a blocked workflow after approval"
    )
    workflow_parser.set_defaults(func=cmd_run_workflow)

    # approvals
    approvals_parser = subparsers.add_parser(
        "approvals",
        help="List and manage pending approvals",
        description="View, approve, or deny pending workflow approvals.",
    )
    approvals_parser.add_argument(
        "--approve", type=str, metavar="ID", help="Approve the specified approval request"
    )
    approvals_parser.add_argument(
        "--deny", type=str, metavar="ID", help="Deny the specified approval request"
    )
    approvals_parser.add_argument(
        "--show", type=str, metavar="ID", help="Show details of a specific approval"
    )
    approvals_parser.add_argument(
        "--reason", type=str, help="Reason for approval or denial decision"
    )
    approvals_parser.set_defaults(func=cmd_approvals)

    # workflows
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

    # plan
    plan_parser = subparsers.add_parser(
        "plan",
        help="Generate a workflow plan from a high-level goal",
        description="Use the planner to generate a multi-step workflow from a natural language goal.",
    )
    plan_parser.add_argument(
        "goal", type=str, help="High-level goal (e.g., 'send monthly newsletter')"
    )
    plan_parser.add_argument(
        "--save", action="store_true", help="Save the generated workflow to disk"
    )
    plan_parser.add_argument(
        "--execute", action="store_true", help="Execute the workflow immediately"
    )
    plan_parser.add_argument(
        "--force", action="store_true", help="Force execution even if autonomy mode is OFF"
    )
    plan_parser.add_argument(
        "--max-actions",
        type=int,
        default=0,
        help="Maximum number of actions to execute (0=unlimited)",
    )
    plan_parser.add_argument(
        "--max-messages",
        type=int,
        default=0,
        help="Maximum number of messages to send (0=unlimited)",
    )
    plan_parser.add_argument(
        "--max-time", type=int, default=0, help="Maximum execution time in seconds (0=unlimited)"
    )
    plan_parser.set_defaults(func=cmd_plan)

    # executor resume
    executor_parser = subparsers.add_parser(
        "executor",
        help="Executor commands",
        description="Resume blocked executor workflows.",
    )
    executor_subparsers = executor_parser.add_subparsers(
        title="executor commands",
        dest="executor_command",
        metavar="COMMAND",
    )

    executor_resume = executor_subparsers.add_parser(
        "resume",
        help="Resume a blocked workflow",
        description="Resume execution of a workflow that was blocked pending approval.",
    )
    executor_resume.add_argument("workflow_id", type=str, help="Workflow ID to resume")
    executor_resume.add_argument(
        "--max-actions",
        type=int,
        default=0,
        help="Maximum number of actions to execute (0=unlimited)",
    )
    executor_resume.add_argument(
        "--max-messages",
        type=int,
        default=0,
        help="Maximum number of messages to send (0=unlimited)",
    )
    executor_resume.add_argument(
        "--max-time", type=int, default=0, help="Maximum execution time in seconds (0=unlimited)"
    )
    executor_resume.set_defaults(func=cmd_executor_resume)

    executor_parser.set_defaults(func=cmd_executor_resume, executor_command="resume")

    # memory
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

    memory_recent = memory_subparsers.add_parser(
        "recent",
        help="Show recent working memory entries",
        description="Display the most recent entries from working memory.",
    )
    memory_recent.add_argument(
        "count", type=int, nargs="?", default=10, help="Number of entries to show (default: 10)"
    )
    memory_recent.add_argument(
        "--show-sensitive", action="store_true", help="No-op (kept for compatibility)"
    )
    memory_recent.set_defaults(func=cmd_memory, memory_command="recent")

    memory_add = memory_subparsers.add_parser(
        "add",
        help="Add a long-term memory entry",
        description="Add a new entry to long-term memory with category and JSON content.",
    )
    memory_add.add_argument(
        "category", type=str, help="Category for the entry (e.g., preferences, facts)"
    )
    memory_add.add_argument(
        "content", type=str, help='JSON content for the entry (e.g., \'{"key": "value"}\')'
    )
    memory_add.add_argument("--ttl", type=str, help="Time to live (e.g., 7d, 24h, 30m, 1w, 10s)")
    memory_add.add_argument(
        "--sensitive", action="store_true", help="Mark this entry as containing sensitive data"
    )
    memory_add.set_defaults(func=cmd_memory, memory_command="add")

    memory_search = memory_subparsers.add_parser(
        "search",
        help="Search long-term memory",
        description="Search for entries in long-term memory by keyword or category.",
    )
    memory_search.add_argument(
        "keyword", type=str, nargs="?", help="Keyword to search for in content"
    )
    memory_search.add_argument("--category", type=str, help="Filter by category")
    memory_search.add_argument(
        "--show-sensitive", action="store_true", help="Show content of sensitive entries"
    )
    memory_search.set_defaults(func=cmd_memory, memory_command="search")

    memory_forget = memory_subparsers.add_parser(
        "forget",
        help="Delete a memory entry",
        description="Delete a specific entry from long-term memory.",
    )
    memory_forget.add_argument("entry_id", type=str, help="ID of the entry to delete")
    memory_forget.set_defaults(func=cmd_memory, memory_command="forget")

    memory_clear = memory_subparsers.add_parser(
        "clear",
        help="Clear all working memory",
        description="Remove all entries from working memory.",
    )
    memory_clear.set_defaults(func=cmd_memory, memory_command="clear")

    memory_retention = memory_subparsers.add_parser(
        "retention",
        help="Run retention policy",
        description="Delete all expired entries from long-term memory.",
    )
    memory_retention.set_defaults(func=cmd_memory, memory_command="retention")

    memory_stats = memory_subparsers.add_parser(
        "stats",
        help="Show memory statistics",
        description="Display statistics about working and long-term memory.",
    )
    memory_stats.set_defaults(func=cmd_memory, memory_command="stats")

    memory_parser.set_defaults(func=cmd_memory, memory_command="stats")

    # kb
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

    kb_ingest = kb_subparsers.add_parser(
        "ingest",
        help="Ingest a file into the knowledge base",
        description="Read and index a text file for later search and retrieval.",
    )
    kb_ingest.add_argument("path", type=str, help="Path to the file to ingest")
    kb_ingest.add_argument(
        "--title", type=str, help="Title for the document (defaults to filename)"
    )
    kb_ingest.add_argument("--tags", type=str, help="Comma-separated list of tags")
    kb_ingest.set_defaults(func=cmd_kb, kb_command="ingest")

    kb_search = kb_subparsers.add_parser(
        "search",
        help="Search the knowledge base",
        description="Search for documents matching a query.",
    )
    kb_search.add_argument("query", type=str, help="Search query")
    kb_search.add_argument(
        "--max-results", type=int, default=5, help="Maximum number of results (default: 5)"
    )
    kb_search.add_argument("-v", "--verbose", action="store_true", help="Show content snippets")
    kb_search.set_defaults(func=cmd_kb, kb_command="search")

    kb_list = kb_subparsers.add_parser(
        "list",
        help="List all documents",
        description="List all documents in the knowledge base.",
    )
    kb_list.add_argument(
        "--limit", type=int, default=20, help="Maximum number of documents to list (default: 20)"
    )
    kb_list.set_defaults(func=cmd_kb, kb_command="list")

    kb_show = kb_subparsers.add_parser(
        "show",
        help="Show a document's content",
        description="Display the full content of a document.",
    )
    kb_show.add_argument("doc_id", type=str, help="Document ID")
    kb_show.set_defaults(func=cmd_kb, kb_command="show")

    kb_delete = kb_subparsers.add_parser(
        "delete",
        help="Delete a document",
        description="Remove a document from the knowledge base.",
    )
    kb_delete.add_argument("doc_id", type=str, help="Document ID to delete")
    kb_delete.set_defaults(func=cmd_kb, kb_command="delete")

    kb_cite = kb_subparsers.add_parser(
        "cite",
        help="Get citations for a query",
        description="Find documents that can be cited for a specific phrase or term.",
    )
    kb_cite.add_argument("query", type=str, help="Text to find citations for")
    kb_cite.set_defaults(func=cmd_kb, kb_command="cite")

    kb_tags = kb_subparsers.add_parser(
        "tags",
        help="List all tags",
        description="List all unique tags in the knowledge base.",
    )
    kb_tags.set_defaults(func=cmd_kb, kb_command="tags")

    kb_parser.set_defaults(func=cmd_kb, kb_command="list")

    # scheduler
    scheduler_parser = subparsers.add_parser(
        "scheduler",
        help="Manage scheduled jobs",
        description="Manage Rex's job scheduler for automated tasks.",
    )
    scheduler_subparsers = scheduler_parser.add_subparsers(
        title="scheduler commands",
        dest="scheduler_command",
        metavar="COMMAND",
    )

    scheduler_list = scheduler_subparsers.add_parser(
        "list",
        help="List all scheduled jobs",
        description="Display all registered jobs with their schedules and status.",
    )
    scheduler_list.add_argument(
        "-v", "--verbose", action="store_true", help="Show detailed job information"
    )
    scheduler_list.set_defaults(func=cmd_scheduler, scheduler_command="list")

    scheduler_run = scheduler_subparsers.add_parser(
        "run",
        help="Run a job immediately",
        description="Execute a scheduled job immediately, ignoring its schedule.",
    )
    scheduler_run.add_argument("job_id", help="Job ID to run")
    scheduler_run.set_defaults(func=cmd_scheduler, scheduler_command="run")

    scheduler_init = scheduler_subparsers.add_parser(
        "init",
        help="Initialize scheduler with default jobs",
        description="Set up the scheduler system with default email and calendar jobs.",
    )
    scheduler_init.set_defaults(func=cmd_scheduler, scheduler_command="init")

    scheduler_parser.set_defaults(func=cmd_scheduler, scheduler_command="list")

    # email
    email_parser = subparsers.add_parser(
        "email",
        help="Manage email",
        description="Read and triage emails from your inbox.",
    )
    email_subparsers = email_parser.add_subparsers(
        title="email commands",
        dest="email_command",
        metavar="COMMAND",
    )

    email_unread = email_subparsers.add_parser(
        "unread",
        help="Fetch unread emails",
        description="Display unread emails with categorization.",
    )
    email_unread.add_argument(
        "--limit", type=int, default=10, help="Maximum number of emails to fetch (default: 10)"
    )
    email_unread.add_argument(
        "-v", "--verbose", action="store_true", help="Show detailed email information"
    )
    email_unread.set_defaults(func=cmd_email, email_command="unread")

    email_parser.set_defaults(func=cmd_email, email_command="unread")

    # calendar
    calendar_parser = subparsers.add_parser(
        "calendar",
        help="Manage calendar",
        description="View and manage calendar events.",
    )
    calendar_subparsers = calendar_parser.add_subparsers(
        title="calendar commands",
        dest="calendar_command",
        metavar="COMMAND",
    )

    calendar_upcoming = calendar_subparsers.add_parser(
        "upcoming",
        help="Show upcoming events",
        description="Display upcoming calendar events.",
    )
    calendar_upcoming.add_argument(
        "--days", type=int, default=7, help="Number of days to look ahead (default: 7)"
    )
    calendar_upcoming.add_argument(
        "--conflicts", action="store_true", help="Check for scheduling conflicts"
    )
    calendar_upcoming.add_argument(
        "-v", "--verbose", action="store_true", help="Show detailed event information"
    )
    calendar_upcoming.set_defaults(func=cmd_calendar, calendar_command="upcoming")

    calendar_parser.set_defaults(func=cmd_calendar, calendar_command="upcoming")

    # reminders
    reminders_parser = subparsers.add_parser(
        "reminders",
        help="Manage reminders",
        description="Create and manage one-off reminders with optional follow-up cues.",
    )
    reminders_subparsers = reminders_parser.add_subparsers(
        title="reminders commands",
        dest="reminders_command",
        metavar="COMMAND",
    )

    reminders_add = reminders_subparsers.add_parser(
        "add",
        help="Add a new reminder",
        description="Create a new one-off reminder at a specific date/time.",
    )
    reminders_add.add_argument("title", type=str, help="Reminder title/description")
    reminders_add.add_argument(
        "--at", type=str, required=True, help="When to remind (YYYY-MM-DD HH:MM)"
    )
    reminders_add.add_argument(
        "--follow-up",
        dest="followup",
        action="store_true",
        help="Create a follow-up cue after reminder fires",
    )
    reminders_add.set_defaults(func=cmd_reminders, reminders_command="add")

    reminders_list = reminders_subparsers.add_parser(
        "list",
        help="List reminders",
        description="List all reminders with their status.",
    )
    reminders_list.add_argument(
        "--status",
        type=str,
        choices=["pending", "fired", "done", "canceled"],
        help="Filter by status",
    )
    reminders_list.set_defaults(func=cmd_reminders, reminders_command="list")

    reminders_done = reminders_subparsers.add_parser(
        "done",
        help="Mark a reminder as done",
        description="Mark a reminder as completed.",
    )
    reminders_done.add_argument("id", type=str, help="Reminder ID to mark as done")
    reminders_done.set_defaults(func=cmd_reminders, reminders_command="done")

    reminders_cancel = reminders_subparsers.add_parser(
        "cancel",
        help="Cancel a reminder",
        description="Cancel a pending reminder.",
    )
    reminders_cancel.add_argument("id", type=str, help="Reminder ID to cancel")
    reminders_cancel.set_defaults(func=cmd_reminders, reminders_command="cancel")

    reminders_parser.set_defaults(func=cmd_reminders, reminders_command="list")

    # cues
    cues_parser = subparsers.add_parser(
        "cues",
        help="Manage follow-up cues",
        description="View and manage follow-up cues for conversations.",
    )
    cues_subparsers = cues_parser.add_subparsers(
        title="cues commands",
        dest="cues_command",
        metavar="COMMAND",
    )

    cues_list = cues_subparsers.add_parser(
        "list",
        help="List cues",
        description="List all follow-up cues with their status.",
    )
    cues_list.add_argument(
        "--status", type=str, choices=["pending", "asked", "dismissed"], help="Filter by status"
    )
    cues_list.set_defaults(func=cmd_cues, cues_command="list")

    cues_dismiss = cues_subparsers.add_parser(
        "dismiss",
        help="Dismiss a cue",
        description="Dismiss a follow-up cue so it won't be asked.",
    )
    cues_dismiss.add_argument("cue_id", type=str, help="Cue ID to dismiss")
    cues_dismiss.set_defaults(func=cmd_cues, cues_command="dismiss")

    cues_prune = cues_subparsers.add_parser(
        "prune",
        help="Prune expired cues",
        description="Remove all expired cues from the store.",
    )
    cues_prune.set_defaults(func=cmd_cues, cues_command="prune")

    cues_parser.set_defaults(func=cmd_cues, cues_command="list")

    # msg (messaging)
    msg_parser = subparsers.add_parser(
        "msg",
        help="Send and receive messages via SMS and other channels",
        description="Manage messaging through various channels (SMS, Telegram, etc.).",
    )
    msg_subparsers = msg_parser.add_subparsers(
        title="messaging commands",
        dest="msg_command",
        metavar="COMMAND",
    )

    msg_send = msg_subparsers.add_parser(
        "send",
        help="Send a message",
        description="Send a message via a specific channel.",
    )
    msg_send.add_argument(
        "--channel", type=str, default="sms", help="Channel to send via (default: sms)"
    )
    msg_send.add_argument(
        "--to", type=str, required=True, help="Recipient (phone number, user ID, etc.)"
    )
    msg_send.add_argument("--body", type=str, required=True, help="Message body text")
    msg_send.set_defaults(func=cmd_msg, msg_command="send")

    msg_receive = msg_subparsers.add_parser(
        "receive",
        help="Receive recent messages",
        description="List recent inbound messages from a channel.",
    )
    msg_receive.add_argument(
        "--channel", type=str, default="sms", help="Channel to receive from (default: sms)"
    )
    msg_receive.add_argument(
        "--limit", type=int, default=10, help="Maximum number of messages (default: 10)"
    )
    msg_receive.set_defaults(func=cmd_msg, msg_command="receive")

    msg_parser.set_defaults(func=cmd_msg, msg_command="receive")

    # notify (notifications)
    notify_parser = subparsers.add_parser(
        "notify",
        help="Send and manage notifications",
        description="Multi-channel notification system with priority routing.",
    )
    notify_subparsers = notify_parser.add_subparsers(
        title="notification commands",
        dest="notify_command",
        metavar="COMMAND",
    )

    notify_send = notify_subparsers.add_parser(
        "send",
        help="Send a notification",
        description="Send a notification with priority and channel preferences.",
    )
    notify_send.add_argument(
        "--priority",
        type=str,
        default="normal",
        choices=["urgent", "normal", "digest"],
        help="Priority level (default: normal)",
    )
    notify_send.add_argument("--title", type=str, required=True, help="Notification title")
    notify_send.add_argument("--body", type=str, required=True, help="Notification body")
    notify_send.add_argument(
        "--channels", type=str, help="Comma-separated list of channels (e.g., sms,email,dashboard)"
    )
    notify_send.set_defaults(func=cmd_notify, notify_command="send")

    notify_list_digests = notify_subparsers.add_parser(
        "list-digests",
        help="List queued digest notifications",
        description="Show all notifications queued for digest delivery.",
    )
    notify_list_digests.set_defaults(func=cmd_notify, notify_command="list-digests")

    notify_flush = notify_subparsers.add_parser(
        "flush-digests",
        help="Flush digest queues",
        description="Send all queued digest notifications immediately.",
    )
    notify_flush.add_argument(
        "--channel", type=str, help="Specific channel to flush (default: all)"
    )
    notify_flush.set_defaults(func=cmd_notify, notify_command="flush-digests")

    notify_ack = notify_subparsers.add_parser(
        "ack",
        help="Acknowledge a notification",
        description="Mark a notification as acknowledged (prevents escalation).",
    )
    notify_ack.add_argument("notification_id", type=str, help="Notification ID to acknowledge")
    notify_ack.set_defaults(func=cmd_notify, notify_command="ack")

    notify_parser.set_defaults(func=cmd_notify, notify_command="list-digests")

    # browser
    browser_parser = subparsers.add_parser(
        "browser",
        help="Browser automation with Playwright",
        description="Automate browser interactions with Playwright.",
    )
    browser_subparsers = browser_parser.add_subparsers(
        title="browser commands",
        dest="browser_command",
        metavar="COMMAND",
    )

    browser_run = browser_subparsers.add_parser(
        "run",
        help="Run a browser automation script",
        description="Execute a browser automation script from a JSON file.",
    )
    browser_run.add_argument("script", type=str, help="Path to the script JSON file")
    browser_run.add_argument(
        "--headed", action="store_true", help="Run in headed mode (show browser)"
    )
    browser_run.set_defaults(func=cmd_browser, browser_command="run")

    browser_sessions = browser_subparsers.add_parser(
        "sessions",
        help="List browser sessions",
        description="List available browser sessions.",
    )
    browser_sessions.set_defaults(func=cmd_browser, browser_command="sessions")

    browser_screenshots = browser_subparsers.add_parser(
        "screenshots",
        help="List captured screenshots",
        description="List all captured screenshots.",
    )
    browser_screenshots.set_defaults(func=cmd_browser, browser_command="screenshots")

    browser_parser.set_defaults(func=cmd_browser, browser_command="sessions")

    # os
    os_parser = subparsers.add_parser(
        "os",
        help="OS automation (safe command execution)",
        description="Execute system commands and file operations with safety rails.",
    )
    os_subparsers = os_parser.add_subparsers(
        title="os commands",
        dest="os_command",
        metavar="COMMAND",
    )

    os_run = os_subparsers.add_parser(
        "run",
        help="Run a safe system command",
        description="Execute a whitelisted system command.",
    )
    os_run.add_argument("command", type=str, help="Command to run (e.g., 'ls -la')")
    os_run.set_defaults(func=cmd_os, os_command="run")

    os_copy = os_subparsers.add_parser(
        "copy",
        help="Copy a file",
        description="Copy a file with path sanitization.",
    )
    os_copy.add_argument("src", type=str, help="Source file path")
    os_copy.add_argument("dst", type=str, help="Destination file path")
    os_copy.set_defaults(func=cmd_os, os_command="copy")

    os_move = os_subparsers.add_parser(
        "move",
        help="Move a file",
        description="Move a file with path sanitization.",
    )
    os_move.add_argument("src", type=str, help="Source file path")
    os_move.add_argument("dst", type=str, help="Destination file path")
    os_move.set_defaults(func=cmd_os, os_command="move")

    os_delete = os_subparsers.add_parser(
        "delete",
        help="Delete a file",
        description="Delete a file (moves to trash by default).",
    )
    os_delete.add_argument("path", type=str, help="File path to delete")
    os_delete.add_argument(
        "--permanent", action="store_true", help="Permanently delete (skip trash)"
    )
    os_delete.set_defaults(func=cmd_os, os_command="delete")

    os_trash = os_subparsers.add_parser(
        "trash",
        help="List trash contents",
        description="Show files in the trash folder.",
    )
    os_trash.set_defaults(func=cmd_os, os_command="trash")

    os_parser.set_defaults(func=cmd_os, os_command="trash")

    # gh (GitHub)
    gh_parser = subparsers.add_parser(
        "gh",
        help="GitHub integration",
        description="Manage GitHub repositories, issues, and pull requests.",
    )
    gh_subparsers = gh_parser.add_subparsers(
        title="github commands",
        dest="gh_command",
        metavar="COMMAND",
    )

    gh_repos = gh_subparsers.add_parser(
        "repos",
        help="List repositories",
        description="List repositories accessible to the authenticated user.",
    )
    gh_repos.set_defaults(func=cmd_gh, gh_command="repos")

    gh_prs = gh_subparsers.add_parser(
        "prs",
        help="List pull requests",
        description="List pull requests for a repository.",
    )
    gh_prs.add_argument("repo", type=str, help="Repository in format 'owner/repo'")
    gh_prs.add_argument(
        "--state",
        type=str,
        default="open",
        choices=["open", "closed", "all"],
        help="PR state filter",
    )
    gh_prs.set_defaults(func=cmd_gh, gh_command="prs")

    gh_issue_create = gh_subparsers.add_parser(
        "issue-create",
        help="Create an issue",
        description="Create a new issue in a repository.",
    )
    gh_issue_create.add_argument("repo", type=str, help="Repository in format 'owner/repo'")
    gh_issue_create.add_argument("--title", type=str, required=True, help="Issue title")
    gh_issue_create.add_argument("--body", type=str, required=True, help="Issue body")
    gh_issue_create.add_argument("--labels", type=str, help="Comma-separated list of labels")
    gh_issue_create.set_defaults(func=cmd_gh, gh_command="issue-create")

    gh_pr_create = gh_subparsers.add_parser(
        "pr-create",
        help="Create a pull request",
        description="Create a new pull request.",
    )
    gh_pr_create.add_argument("repo", type=str, help="Repository in format 'owner/repo'")
    gh_pr_create.add_argument("--head", type=str, required=True, help="Head branch")
    gh_pr_create.add_argument("--base", type=str, required=True, help="Base branch")
    gh_pr_create.add_argument("--title", type=str, required=True, help="PR title")
    gh_pr_create.add_argument("--body", type=str, required=True, help="PR body")
    gh_pr_create.set_defaults(func=cmd_gh, gh_command="pr-create")

    gh_parser.set_defaults(func=cmd_gh, gh_command="repos")

    # code (VS Code integration)
    code_parser = subparsers.add_parser(
        "code",
        help="VS Code integration",
        description="File operations, patch application, and test execution.",
    )
    code_subparsers = code_parser.add_subparsers(
        title="code commands",
        dest="code_command",
        metavar="COMMAND",
    )

    code_open = code_subparsers.add_parser(
        "open",
        help="Open and display a file",
        description="Read and display file contents.",
    )
    code_open.add_argument("path", type=str, help="File path to open")
    code_open.set_defaults(func=cmd_code, code_command="open")

    code_patch = code_subparsers.add_parser(
        "patch",
        help="Apply a patch to a file",
        description="Apply a unified diff patch to a file.",
    )
    code_patch.add_argument("path", type=str, help="File path to patch")
    code_patch.add_argument("--patch-file", type=str, required=True, help="Path to patch/diff file")
    code_patch.set_defaults(func=cmd_code, code_command="patch")

    code_test = code_subparsers.add_parser(
        "test",
        help="Run tests",
        description="Execute tests using pytest.",
    )
    code_test.add_argument("--path", type=str, help="Test file or directory (default: tests/)")
    code_test.add_argument("--pattern", type=str, help="Test pattern filter (e.g., 'test_browser')")
    code_test.add_argument("-v", "--verbose", action="store_true", help="Verbose test output")
    code_test.set_defaults(func=cmd_code, code_command="test")

    code_parser.set_defaults(func=cmd_code, code_command="open")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Main entry point for the Rex CLI."""
    parser = create_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        if getattr(args, "verbose", False):
            parser.print_help()
            return 0
        args.func = cmd_chat
        args.verbose = getattr(args, "verbose", False)

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
