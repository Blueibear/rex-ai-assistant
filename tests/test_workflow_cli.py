"""Tests for the workflow CLI commands.

This module tests the CLI interface for workflows:
- run-workflow command with --dry-run and --resume
- approvals command for listing and managing approvals
- workflows command for listing workflows
"""

import json
import tempfile
from pathlib import Path
from unittest import mock

import pytest

from rex.cli import (
    create_parser,
    cmd_run_workflow,
    cmd_approvals,
    cmd_workflows,
    main,
)
from rex.workflow import Workflow, WorkflowStep, WorkflowApproval
from rex.contracts import ToolCall


class TestRunWorkflowCommand:
    """Tests for the run-workflow CLI command."""

    def test_parser_run_workflow_subcommand(self):
        """Test that run-workflow subcommand is registered."""
        parser = create_parser()
        args = parser.parse_args(["run-workflow", "test.json"])
        assert args.workflow == "test.json"
        assert hasattr(args, "func")

    def test_parser_run_workflow_dry_run_flag(self):
        """Test that --dry-run flag is recognized."""
        parser = create_parser()
        args = parser.parse_args(["run-workflow", "test.json", "--dry-run"])
        assert args.dry_run is True
        assert args.resume is False

    def test_parser_run_workflow_resume_flag(self):
        """Test that --resume flag is recognized."""
        parser = create_parser()
        args = parser.parse_args(["run-workflow", "test.json", "--resume"])
        assert args.resume is True
        assert args.dry_run is False

    def test_run_workflow_file_not_found(self, capsys):
        """Test run-workflow with non-existent file."""
        parser = create_parser()
        args = parser.parse_args(["run-workflow", "/nonexistent/workflow.json"])
        result = cmd_run_workflow(args)

        assert result == 1
        captured = capsys.readouterr()
        assert "Error" in captured.out
        assert "not found" in captured.out

    def test_run_workflow_invalid_json(self, capsys):
        """Test run-workflow with invalid JSON file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("not valid json")
            f.flush()

            parser = create_parser()
            args = parser.parse_args(["run-workflow", f.name])
            result = cmd_run_workflow(args)

            assert result == 1
            captured = capsys.readouterr()
            assert "Error" in captured.out

    def test_run_workflow_dry_run_mode(self, capsys):
        """Test run-workflow in dry-run mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a workflow file
            wf = Workflow(
                workflow_id="wf_test",
                title="Test Workflow",
                steps=[
                    WorkflowStep(
                        step_id="s1",
                        description="Get time",
                        tool_call=ToolCall(tool="time_now", args={}),
                    )
                ],
            )
            wf_path = Path(tmpdir) / "workflow.json"
            with open(wf_path, "w") as f:
                f.write(wf.model_dump_json())

            parser = create_parser()
            args = parser.parse_args(["run-workflow", str(wf_path), "--dry-run"])

            # Mock the workflow directories
            with mock.patch("rex.workflow.DEFAULT_WORKFLOW_DIR", Path(tmpdir) / "workflows"):
                with mock.patch("rex.workflow.DEFAULT_APPROVAL_DIR", Path(tmpdir) / "approvals"):
                    result = cmd_run_workflow(args)

            assert result == 0
            captured = capsys.readouterr()
            assert "dry-run" in captured.out.lower()
            assert "s1" in captured.out
            assert "Get time" in captured.out

    def test_run_workflow_executes_steps(self, capsys):
        """Test run-workflow actually executes workflow steps."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a workflow file with a low-risk step
            wf = Workflow(
                workflow_id="wf_exec_test",
                title="Execution Test",
                steps=[
                    WorkflowStep(
                        step_id="s1",
                        description="Get time in Dallas",
                        tool_call=ToolCall(tool="time_now", args={"location": "Dallas, TX"}),
                    )
                ],
            )
            wf_path = Path(tmpdir) / "workflow.json"
            with open(wf_path, "w") as f:
                f.write(wf.model_dump_json())

            parser = create_parser()
            args = parser.parse_args(["run-workflow", str(wf_path)])

            with mock.patch("rex.workflow.DEFAULT_WORKFLOW_DIR", Path(tmpdir) / "workflows"):
                with mock.patch("rex.workflow.DEFAULT_APPROVAL_DIR", Path(tmpdir) / "approvals"):
                    with mock.patch("rex.workflow_runner.DEFAULT_WORKFLOW_DIR", Path(tmpdir) / "workflows"):
                        with mock.patch("rex.workflow_runner.DEFAULT_APPROVAL_DIR", Path(tmpdir) / "approvals"):
                            result = cmd_run_workflow(args)

            assert result == 0
            captured = capsys.readouterr()
            assert "completed" in captured.out.lower()

    def test_run_workflow_resume_not_blocked(self, capsys):
        """Test resume on non-blocked workflow fails."""
        with tempfile.TemporaryDirectory() as tmpdir:
            wf = Workflow(
                workflow_id="wf_resume_test",
                title="Resume Test",
                status="queued",
                steps=[],
            )
            wf_path = Path(tmpdir) / "workflow.json"
            with open(wf_path, "w") as f:
                f.write(wf.model_dump_json())

            parser = create_parser()
            args = parser.parse_args(["run-workflow", str(wf_path), "--resume"])

            result = cmd_run_workflow(args)

            assert result == 1
            captured = capsys.readouterr()
            assert "Cannot resume" in captured.out

    def test_run_workflow_resume_loads_persisted(self, capsys):
        """Test resume loads persisted blocked workflow state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workflow_id = "wf_resume_persisted"
            workflow_file = Path(tmpdir) / "workflow.json"
            queued = Workflow(
                workflow_id=workflow_id,
                title="Queued Workflow",
                status="queued",
                steps=[],
            )
            workflow_file.write_text(queued.model_dump_json())

            blocked = Workflow(
                workflow_id=workflow_id,
                title="Blocked Workflow",
                status="blocked",
                steps=[],
                blocking_approval_id="apr_resume_pending",
            )
            workflows_dir = Path(tmpdir) / "workflows"
            approvals_dir = Path(tmpdir) / "approvals"
            blocked.save(workflows_dir)

            approval = WorkflowApproval(
                approval_id="apr_resume_pending",
                workflow_id=workflow_id,
                step_id="step_001",
                status="pending",
            )
            approval.save(approvals_dir)

            parser = create_parser()
            args = parser.parse_args(["run-workflow", str(workflow_file), "--resume"])

            with mock.patch("rex.workflow.DEFAULT_WORKFLOW_DIR", workflows_dir):
                with mock.patch("rex.workflow.DEFAULT_APPROVAL_DIR", approvals_dir):
                    with mock.patch("rex.workflow_runner.DEFAULT_WORKFLOW_DIR", workflows_dir):
                        with mock.patch("rex.workflow_runner.DEFAULT_APPROVAL_DIR", approvals_dir):
                            result = cmd_run_workflow(args)

            assert result == 0
            captured = capsys.readouterr()
            assert "blocked on approval" in captured.out.lower()


class TestApprovalsCommand:
    """Tests for the approvals CLI command."""

    def test_parser_approvals_subcommand(self):
        """Test that approvals subcommand is registered."""
        parser = create_parser()
        args = parser.parse_args(["approvals"])
        assert hasattr(args, "func")

    def test_parser_approvals_approve_flag(self):
        """Test that --approve flag is recognized."""
        parser = create_parser()
        args = parser.parse_args(["approvals", "--approve", "apr_123"])
        assert args.approve == "apr_123"

    def test_parser_approvals_deny_flag(self):
        """Test that --deny flag is recognized."""
        parser = create_parser()
        args = parser.parse_args(["approvals", "--deny", "apr_123", "--reason", "Not allowed"])
        assert args.deny == "apr_123"
        assert args.reason == "Not allowed"

    def test_parser_approvals_show_flag(self):
        """Test that --show flag is recognized."""
        parser = create_parser()
        args = parser.parse_args(["approvals", "--show", "apr_123"])
        assert args.show == "apr_123"

    def test_approvals_list_empty(self, capsys):
        """Test listing approvals when none exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            parser = create_parser()
            args = parser.parse_args(["approvals"])

            with mock.patch("rex.workflow_runner.DEFAULT_APPROVAL_DIR", Path(tmpdir)):
                result = cmd_approvals(args)

            assert result == 0
            captured = capsys.readouterr()
            assert "No pending approvals" in captured.out

    def test_approvals_list_pending(self, capsys):
        """Test listing pending approvals."""
        with tempfile.TemporaryDirectory() as tmpdir:
            approval = WorkflowApproval(
                approval_id="apr_test",
                workflow_id="wf_001",
                step_id="step_001",
                step_description="Test step",
            )
            approval.save(tmpdir)

            parser = create_parser()
            args = parser.parse_args(["approvals"])

            with mock.patch("rex.workflow_runner.DEFAULT_APPROVAL_DIR", Path(tmpdir)):
                result = cmd_approvals(args)

            assert result == 0
            captured = capsys.readouterr()
            assert "apr_test" in captured.out
            assert "wf_001" in captured.out

    def test_approvals_approve(self, capsys):
        """Test approving an approval via CLI."""
        with tempfile.TemporaryDirectory() as tmpdir:
            approval = WorkflowApproval(
                approval_id="apr_approve_test",
                workflow_id="wf_001",
                step_id="step_001",
            )
            approval.save(tmpdir)

            parser = create_parser()
            args = parser.parse_args(["approvals", "--approve", "apr_approve_test"])

            with mock.patch("rex.workflow_runner.DEFAULT_APPROVAL_DIR", Path(tmpdir)):
                with mock.patch("rex.workflow.DEFAULT_APPROVAL_DIR", Path(tmpdir)):
                    result = cmd_approvals(args)

            assert result == 0
            captured = capsys.readouterr()
            assert "Approved" in captured.out

            # Verify approval was updated
            loaded = WorkflowApproval.load("apr_approve_test", tmpdir)
            assert loaded.status == "approved"

    def test_approvals_deny(self, capsys):
        """Test denying an approval via CLI."""
        with tempfile.TemporaryDirectory() as tmpdir:
            approval = WorkflowApproval(
                approval_id="apr_deny_test",
                workflow_id="wf_001",
                step_id="step_001",
            )
            approval.save(tmpdir)

            parser = create_parser()
            args = parser.parse_args([
                "approvals",
                "--deny", "apr_deny_test",
                "--reason", "Not authorized",
            ])

            with mock.patch("rex.workflow_runner.DEFAULT_APPROVAL_DIR", Path(tmpdir)):
                with mock.patch("rex.workflow.DEFAULT_APPROVAL_DIR", Path(tmpdir)):
                    result = cmd_approvals(args)

            assert result == 0
            captured = capsys.readouterr()
            assert "Denied" in captured.out

            loaded = WorkflowApproval.load("apr_deny_test", tmpdir)
            assert loaded.status == "denied"
            assert loaded.reason == "Not authorized"

    def test_approvals_show(self, capsys):
        """Test showing approval details."""
        with tempfile.TemporaryDirectory() as tmpdir:
            approval = WorkflowApproval(
                approval_id="apr_show_test",
                workflow_id="wf_001",
                step_id="step_001",
                step_description="Send email",
                tool_call_summary="send_email({'to': 'user@example.com'})",
            )
            approval.save(tmpdir)

            parser = create_parser()
            args = parser.parse_args(["approvals", "--show", "apr_show_test"])

            with mock.patch("rex.workflow.DEFAULT_APPROVAL_DIR", Path(tmpdir)):
                result = cmd_approvals(args)

            assert result == 0
            captured = capsys.readouterr()
            assert "apr_show_test" in captured.out
            assert "wf_001" in captured.out
            assert "Send email" in captured.out

    def test_approvals_approve_nonexistent(self, capsys):
        """Test approving a non-existent approval."""
        with tempfile.TemporaryDirectory() as tmpdir:
            parser = create_parser()
            args = parser.parse_args(["approvals", "--approve", "nonexistent"])

            with mock.patch("rex.workflow_runner.DEFAULT_APPROVAL_DIR", Path(tmpdir)):
                result = cmd_approvals(args)

            assert result == 1
            captured = capsys.readouterr()
            assert "Error" in captured.out
            assert "not found" in captured.out


class TestWorkflowsCommand:
    """Tests for the workflows CLI command."""

    def test_parser_workflows_subcommand(self):
        """Test that workflows subcommand is registered."""
        parser = create_parser()
        args = parser.parse_args(["workflows"])
        assert hasattr(args, "func")

    def test_parser_workflows_status_filter(self):
        """Test that --status filter is recognized."""
        parser = create_parser()
        args = parser.parse_args(["workflows", "--status", "blocked"])
        assert args.status == "blocked"

    def test_workflows_list_empty(self, capsys):
        """Test listing workflows when none exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            parser = create_parser()
            args = parser.parse_args(["workflows"])

            with mock.patch("rex.workflow.DEFAULT_WORKFLOW_DIR", Path(tmpdir)):
                result = cmd_workflows(args)

            assert result == 0
            captured = capsys.readouterr()
            assert "No workflows found" in captured.out

    def test_workflows_list_all(self, capsys):
        """Test listing all workflows."""
        with tempfile.TemporaryDirectory() as tmpdir:
            wf1 = Workflow(workflow_id="wf_1", title="First", status="completed")
            wf2 = Workflow(workflow_id="wf_2", title="Second", status="blocked")
            wf1.save(tmpdir)
            wf2.save(tmpdir)

            parser = create_parser()
            args = parser.parse_args(["workflows"])

            with mock.patch("rex.workflow.DEFAULT_WORKFLOW_DIR", Path(tmpdir)):
                result = cmd_workflows(args)

            assert result == 0
            captured = capsys.readouterr()
            assert "wf_1" in captured.out
            assert "wf_2" in captured.out
            assert "First" in captured.out
            assert "Second" in captured.out
            assert "Total: 2" in captured.out

    def test_workflows_list_filtered(self, capsys):
        """Test listing workflows with status filter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            wf1 = Workflow(workflow_id="wf_1", title="First", status="completed")
            wf2 = Workflow(workflow_id="wf_2", title="Second", status="blocked")
            wf1.save(tmpdir)
            wf2.save(tmpdir)

            parser = create_parser()
            args = parser.parse_args(["workflows", "--status", "blocked"])

            with mock.patch("rex.workflow.DEFAULT_WORKFLOW_DIR", Path(tmpdir)):
                result = cmd_workflows(args)

            assert result == 0
            captured = capsys.readouterr()
            assert "wf_2" in captured.out
            assert "wf_1" not in captured.out
            assert "Total: 1" in captured.out


class TestMainFunction:
    """Tests for the main CLI entry point."""

    def test_main_run_workflow_integration(self):
        """Test main function with run-workflow command."""
        with tempfile.TemporaryDirectory() as tmpdir:
            wf = Workflow(
                workflow_id="wf_main_test",
                title="Main Test",
                steps=[
                    WorkflowStep(
                        step_id="s1",
                        description="Get time",
                        tool_call=ToolCall(tool="time_now", args={"location": "Dallas, TX"}),
                    )
                ],
            )
            wf_path = Path(tmpdir) / "workflow.json"
            with open(wf_path, "w") as f:
                f.write(wf.model_dump_json())

            with mock.patch("rex.workflow.DEFAULT_WORKFLOW_DIR", Path(tmpdir) / "workflows"):
                with mock.patch("rex.workflow.DEFAULT_APPROVAL_DIR", Path(tmpdir) / "approvals"):
                    with mock.patch("rex.workflow_runner.DEFAULT_WORKFLOW_DIR", Path(tmpdir) / "workflows"):
                        with mock.patch("rex.workflow_runner.DEFAULT_APPROVAL_DIR", Path(tmpdir) / "approvals"):
                            result = main(["run-workflow", str(wf_path), "--dry-run"])

            assert result == 0

    def test_main_approvals_integration(self):
        """Test main function with approvals command."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch("rex.workflow_runner.DEFAULT_APPROVAL_DIR", Path(tmpdir)):
                result = main(["approvals"])

            assert result == 0

    def test_main_workflows_integration(self):
        """Test main function with workflows command."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch("rex.workflow.DEFAULT_WORKFLOW_DIR", Path(tmpdir)):
                result = main(["workflows"])

            assert result == 0


class TestCLIHelp:
    """Tests for CLI help output."""

    def test_main_help_includes_workflow_commands(self, capsys):
        """Test that --help includes workflow commands."""
        parser = create_parser()

        # Get help text
        with pytest.raises(SystemExit):
            parser.parse_args(["--help"])

        captured = capsys.readouterr()
        assert "run-workflow" in captured.out
        assert "approvals" in captured.out
        assert "workflows" in captured.out

    def test_run_workflow_help(self, capsys):
        """Test run-workflow --help output."""
        parser = create_parser()

        with pytest.raises(SystemExit):
            parser.parse_args(["run-workflow", "--help"])

        captured = capsys.readouterr()
        assert "--dry-run" in captured.out
        assert "--resume" in captured.out

    def test_approvals_help(self, capsys):
        """Test approvals --help output."""
        parser = create_parser()

        with pytest.raises(SystemExit):
            parser.parse_args(["approvals", "--help"])

        captured = capsys.readouterr()
        assert "--approve" in captured.out
        assert "--deny" in captured.out
        assert "--show" in captured.out
