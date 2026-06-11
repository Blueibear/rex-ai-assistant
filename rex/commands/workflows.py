"""Workflow, approval, and planner commands for the Rex CLI.

Extracted verbatim from ``rex/cli.py`` (US-REM-027). Handler behavior,
argument definitions, help text, defaults, and exit codes are unchanged.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _cli():
    """Return the ``rex.cli`` module at call time.

    ``rex.cli`` is the single patch point for service getters and command
    handlers (tests monkeypatch ``rex.cli.<name>``). Resolving through the
    module at call time preserves that behavior without creating an import
    cycle at module load time.
    """
    from rex import cli

    return cli


def cmd_run_workflow(args: argparse.Namespace) -> int:
    """Run a workflow from a JSON file."""

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
            result = runner.resume()  # type: ignore[assignment]
        except ValueError as e:
            print(f"Error: {e}")
            return 1
    else:
        print("Running workflow...")
        print("-" * 60)
        result = runner.run()  # type: ignore[assignment]

    print()
    print("-" * 60)
    print("Workflow finished")
    print(f"  Status: {result.status}")  # type: ignore[attr-defined]
    print(f"  Steps executed: {result.steps_executed}/{result.steps_total}")  # type: ignore[attr-defined]

    if result.error:  # type: ignore[attr-defined]
        print(f"  Error: {result.error}")  # type: ignore[attr-defined]
        return 1

    if result.blocking_approval_id:  # type: ignore[attr-defined]
        print(f"  Blocked on approval: {result.blocking_approval_id}")  # type: ignore[attr-defined]
        print()
        print("To approve, run:")
        print(f"  rex approvals --approve {result.blocking_approval_id}")  # type: ignore[attr-defined]
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
    from rex.openclaw.tool_registry import get_tool_registry
    from rex.openclaw.workflow_bridge import WorkflowBridge
    from rex.planner import Planner, UnableToPlanError
    from rex.policy_engine import get_policy_engine

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

        print("-" * 60)

        runner = WorkflowBridge(workflow)
        result = runner.run()

        print()
        print("-" * 60)
        print("Execution complete")
        print(f"Workflow: {result.workflow_id}")
        print(f"Status: {result.status}")
        print(f"Steps: {result.steps_executed}/{result.steps_total}")
        if result.error:
            print(f"Error: {result.error}")

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
    from rex.openclaw.workflow_bridge import WorkflowBridge
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

    print("-" * 60)

    runner = WorkflowBridge(workflow)
    result = runner.run()

    print()
    print("-" * 60)
    print("Execution complete")
    print(f"Workflow: {result.workflow_id}")
    print(f"Status: {result.status}")
    print(f"Steps: {result.steps_executed}/{result.steps_total}")
    if result.error:
        print(f"Error: {result.error}")

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


def register(subparsers: argparse._SubParsersAction) -> None:
    """Register this domain's subcommands on the top-level subparsers."""
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
    workflow_parser.set_defaults(func=_cli().cmd_run_workflow)

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
    approvals_parser.set_defaults(func=_cli().cmd_approvals)

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
    workflows_parser.set_defaults(func=_cli().cmd_workflows)

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
    plan_parser.set_defaults(func=_cli().cmd_plan)

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
    executor_resume.set_defaults(func=_cli().cmd_executor_resume)

    executor_parser.set_defaults(func=_cli().cmd_executor_resume, executor_command="resume")
