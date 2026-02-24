"""CLI tests for scheduler, email, and calendar commands."""

from __future__ import annotations

from argparse import Namespace

from rex.cli import cmd_calendar, cmd_email, cmd_scheduler
from rex.services import initialize_services, reset_services


def test_cli_scheduler_list_and_run(tmp_path, capsys):
    reset_services()
    storage = tmp_path / "jobs.json"
    services = initialize_services(storage_path=storage)
    job_id = services.scheduler.list_jobs()[0].job_id

    args_list = Namespace(storage=str(storage), scheduler_command="list", job_id=None)
    assert cmd_scheduler(args_list) == 0
    output = capsys.readouterr().out
    assert "Scheduled Jobs" in output

    args_run = Namespace(storage=str(storage), scheduler_command="run", job_id=job_id)
    assert cmd_scheduler(args_run) == 0
    output = capsys.readouterr().out
    assert "executed" in output.lower()


def test_cli_scheduler_run_retention_jobs(tmp_path, capsys):
    """Retention jobs are runnable via CLI after scheduler initialization."""
    reset_services()
    storage = tmp_path / "jobs.json"

    raw_config = {
        "notifications": {
            "dashboard": {
                "store": {
                    "cleanup_schedule": "interval:86400",
                    "path": str(tmp_path / "dashboard.db"),
                }
            }
        },
        "messaging": {
            "inbound": {
                "enabled": True,
                "cleanup_schedule": "interval:86400",
                "store_path": str(tmp_path / "inbound.db"),
            }
        },
    }

    from unittest.mock import patch

    with patch("rex.config_manager.load_config", return_value=raw_config):
        initialize_services(storage_path=storage)

        args_dashboard = Namespace(
            storage=str(storage),
            scheduler_command="run",
            job_id="dashboard_retention_cleanup",
        )
        assert cmd_scheduler(args_dashboard) == 0

        args_inbound = Namespace(
            storage=str(storage),
            scheduler_command="run",
            job_id="inbound_sms_retention_cleanup",
        )
        assert cmd_scheduler(args_inbound) == 0

    output = capsys.readouterr().out
    assert "dashboard_retention_cleanup executed successfully" in output
    assert "inbound_sms_retention_cleanup executed successfully" in output


def test_cli_email_unread(capsys):
    reset_services()
    args = Namespace(email_command="unread")
    assert cmd_email(args) == 0
    output = capsys.readouterr().out
    assert "Unread Email Summary" in output


def test_cli_calendar_upcoming(capsys):
    reset_services()
    args = Namespace(calendar_command="upcoming")
    assert cmd_calendar(args) == 0
    output = capsys.readouterr().out
    assert "Upcoming Events" in output
