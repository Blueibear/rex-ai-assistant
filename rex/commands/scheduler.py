"""Scheduler commands for the Rex CLI.

Extracted verbatim from ``rex/cli.py`` (US-REM-027). Handler behavior,
argument definitions, help text, defaults, and exit codes are unchanged.
"""

from __future__ import annotations

import argparse


def _cli():
    """Return the ``rex.cli`` module at call time.

    ``rex.cli`` is the single patch point for service getters and command
    handlers (tests monkeypatch ``rex.cli.<name>``). Resolving through the
    module at call time preserves that behavior without creating an import
    cycle at module load time.
    """
    from rex import cli

    return cli


def cmd_scheduler(args: argparse.Namespace) -> int:
    """Manage scheduler jobs."""
    scheduler = _cli().get_scheduler()
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
        _cli().initialize_scheduler_system(start_scheduler=False)

        if scheduler.run_job(job_id, force=True):
            print(f"Job {job_id} executed successfully")
            return 0
        print(f"Error: Failed to run job {job_id}")
        return 1

    if subcommand == "init":
        _cli().initialize_scheduler_system(start_scheduler=False)
        print("Scheduler system initialized with default jobs")
        print("Use 'rex scheduler list' to see registered jobs")
        return 0

    print("Unknown scheduler subcommand. Use 'rex scheduler --help'")
    return 1


def register(subparsers: argparse._SubParsersAction) -> None:
    """Register this domain's subcommands on the top-level subparsers."""
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
    scheduler_list.set_defaults(func=_cli().cmd_scheduler, scheduler_command="list")

    scheduler_run = scheduler_subparsers.add_parser(
        "run",
        help="Run a job immediately",
        description="Execute a scheduled job immediately, ignoring its schedule.",
    )
    scheduler_run.add_argument("job_id", help="Job ID to run")
    scheduler_run.set_defaults(func=_cli().cmd_scheduler, scheduler_command="run")

    scheduler_init = scheduler_subparsers.add_parser(
        "init",
        help="Initialize scheduler with default jobs",
        description="Set up the scheduler system with default email and calendar jobs.",
    )
    scheduler_init.set_defaults(func=_cli().cmd_scheduler, scheduler_command="init")

    scheduler_parser.set_defaults(func=_cli().cmd_scheduler, scheduler_command="list")
