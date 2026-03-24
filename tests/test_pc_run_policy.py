"""Offline tests for rex.computers.pc_run_policy (Cycle 5.2b).

All tests run without any network calls.  Approvals are stored in a
pytest ``tmp_path`` directory so repo tracked files are never modified.

Coverage
--------
- check_pc_run_policy: creates pending approval when none exists
- check_pc_run_policy: returns existing pending approval (idempotent)
- check_pc_run_policy: returns approved approval when one exists
- check_pc_run_policy: denied decision — no approval record created
- check_pc_run_policy: auto-execute decision (custom low-risk policy)
- find_pending_or_approved_approval: returns None when dir is absent
- find_pending_or_approved_approval: ignores unrelated approvals
- Approval payload contains required fields; no auth tokens stored
- CLI integration: policy creates approval → user sees approval ID
- CLI integration: denied policy → refuses with clear message
- CLI integration: approved approval + --yes → executes
- CLI integration: approved approval without --yes → refuses (--yes guard)
- CLI integration: non-allowlisted command → refused before approval
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from rex.computers.pc_run_policy import (
    PC_RUN_WORKFLOW_ID,
    _command_step_id,
    check_pc_run_policy,
    find_pending_or_approved_approval,
)
from rex.contracts import RiskLevel
from rex.policy import ActionPolicy, PolicyDecision
from rex.policy_engine import PolicyEngine
from rex.workflow import WorkflowApproval

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_denied_engine() -> PolicyEngine:
    """Return a PolicyEngine whose pc_run policy always denies."""
    # Use allowed_recipients restriction with an empty list to force denial.
    engine = PolicyEngine(
        policies=[
            ActionPolicy(
                tool_name="pc_run",
                risk=RiskLevel.HIGH,
                allow_auto=False,
                allowed_recipients=["nobody"],  # desktop will never match
            )
        ]
    )
    return engine


def _make_auto_engine() -> PolicyEngine:
    """Return a PolicyEngine whose pc_run policy allows auto-execute."""
    engine = PolicyEngine(
        policies=[
            ActionPolicy(
                tool_name="pc_run",
                risk=RiskLevel.LOW,
                allow_auto=True,
            )
        ]
    )
    return engine


def _make_approval_engine() -> PolicyEngine:
    """Return the default PolicyEngine (HIGH risk, requires approval)."""
    return PolicyEngine()


def _save_approved_approval(
    computer_id: str,
    command: str,
    args: list[str],
    approval_dir: Path,
) -> WorkflowApproval:
    """Helper: save an already-approved approval record to disk."""
    step_id = _command_step_id(computer_id, command, args)
    approval = WorkflowApproval(
        workflow_id=PC_RUN_WORKFLOW_ID,
        step_id=step_id,
        status="approved",
        requested_by="cli",
        step_description=f"Remote execution on {computer_id!r}: {command}",
        tool_call_summary=json.dumps({"computer_id": computer_id, "command": command}),
    )
    approval.save(approval_dir)
    return approval


# ===========================================================================
# Unit tests for check_pc_run_policy
# ===========================================================================


class TestCheckPcRunPolicyApprovalRequired:
    def test_creates_pending_approval_when_none_exists(self, tmp_path: Path) -> None:
        engine = _make_approval_engine()
        decision, approval = check_pc_run_policy(
            "desktop",
            "whoami",
            [],
            allowlist_matched=True,
            allowlist_rule="whoami",
            initiated_by="alice",
            policy_engine=engine,
            approval_dir=tmp_path,
        )

        assert decision.requires_approval is True
        assert decision.denied is False
        assert approval is not None
        assert approval.status == "pending"
        assert approval.workflow_id == PC_RUN_WORKFLOW_ID

        # Verify approval was persisted
        approval_files = list(tmp_path.glob("*.json"))
        assert len(approval_files) == 1

    def test_pending_approval_contains_required_metadata(self, tmp_path: Path) -> None:
        engine = _make_approval_engine()
        _, approval = check_pc_run_policy(
            "desktop",
            "ipconfig",
            ["/all"],
            allowlist_matched=True,
            allowlist_rule="ipconfig",
            initiated_by="bob",
            policy_engine=engine,
            approval_dir=tmp_path,
        )

        assert approval is not None
        payload = json.loads(approval.tool_call_summary)
        assert payload["computer_id"] == "desktop"
        assert payload["command"] == "ipconfig"
        assert payload["args"] == ["/all"]
        assert payload["allowlist_decision"] == "allowed"
        assert payload["allowlist_rule_matched"] == "ipconfig"
        assert payload["initiated_by"] == "bob"
        # No auth tokens
        assert "token" not in approval.tool_call_summary.lower()
        assert "password" not in approval.tool_call_summary.lower()

    def test_pending_approval_step_description_is_human_readable(self, tmp_path: Path) -> None:
        engine = _make_approval_engine()
        _, approval = check_pc_run_policy(
            "server01",
            "systeminfo",
            [],
            policy_engine=engine,
            approval_dir=tmp_path,
        )
        assert approval is not None
        assert "server01" in approval.step_description
        assert "systeminfo" in approval.step_description

    def test_idempotent_returns_existing_pending_approval(self, tmp_path: Path) -> None:
        """Re-running check_pc_run_policy with no existing approval creates one;
        a second call finds it and returns it without creating another."""
        engine = _make_approval_engine()
        _, first = check_pc_run_policy(
            "desktop",
            "whoami",
            [],
            policy_engine=engine,
            approval_dir=tmp_path,
        )
        assert first is not None

        _, second = check_pc_run_policy(
            "desktop",
            "whoami",
            [],
            policy_engine=engine,
            approval_dir=tmp_path,
        )
        assert second is not None
        assert second.approval_id == first.approval_id

        # Only one file on disk
        assert len(list(tmp_path.glob("*.json"))) == 1

    def test_returns_approved_approval_when_present(self, tmp_path: Path) -> None:
        """After the user approves, re-running returns the approved record."""
        _save_approved_approval("desktop", "whoami", [], tmp_path)

        engine = _make_approval_engine()
        decision, approval = check_pc_run_policy(
            "desktop",
            "whoami",
            [],
            policy_engine=engine,
            approval_dir=tmp_path,
        )

        assert decision.requires_approval is True
        assert approval is not None
        assert approval.status == "approved"

    def test_different_args_different_approvals(self, tmp_path: Path) -> None:
        """Commands with different arguments get separate approval records."""
        engine = _make_approval_engine()
        _, a1 = check_pc_run_policy(
            "desktop", "dir", ["C:\\"], policy_engine=engine, approval_dir=tmp_path
        )
        _, a2 = check_pc_run_policy(
            "desktop", "dir", ["D:\\"], policy_engine=engine, approval_dir=tmp_path
        )

        assert a1 is not None
        assert a2 is not None
        assert a1.approval_id != a2.approval_id


class TestCheckPcRunPolicyDenied:
    def test_denied_decision_returns_none_approval(self, tmp_path: Path) -> None:
        engine = _make_denied_engine()
        decision, approval = check_pc_run_policy(
            "desktop",
            "whoami",
            [],
            policy_engine=engine,
            approval_dir=tmp_path,
        )

        assert decision.denied is True
        assert decision.allowed is False
        assert approval is None
        # No approval file created
        assert list(tmp_path.glob("*.json")) == []

    def test_denied_decision_has_clear_reason(self, tmp_path: Path) -> None:
        engine = _make_denied_engine()
        decision, _ = check_pc_run_policy(
            "desktop",
            "whoami",
            [],
            policy_engine=engine,
            approval_dir=tmp_path,
        )
        assert decision.reason  # non-empty


class TestCheckPcRunPolicyAutoExecute:
    def test_auto_execute_returns_none_approval(self, tmp_path: Path) -> None:
        engine = _make_auto_engine()
        decision, approval = check_pc_run_policy(
            "desktop",
            "whoami",
            [],
            policy_engine=engine,
            approval_dir=tmp_path,
        )

        assert decision.denied is False
        assert decision.requires_approval is False
        assert approval is None
        # No approval file created for auto-execute
        assert list(tmp_path.glob("*.json")) == []


# ===========================================================================
# Unit tests for find_pending_or_approved_approval
# ===========================================================================


class TestFindPendingOrApprovedApproval:
    def test_returns_none_when_dir_absent(self, tmp_path: Path) -> None:
        missing = tmp_path / "no_such_dir"
        result = find_pending_or_approved_approval("desktop", "whoami", [], approval_dir=missing)
        assert result is None

    def test_returns_none_when_no_matching_approval(self, tmp_path: Path) -> None:
        # Save an approval for a different command
        step_id = _command_step_id("desktop", "dir", [])
        approval = WorkflowApproval(
            workflow_id=PC_RUN_WORKFLOW_ID,
            step_id=step_id,
            status="pending",
            requested_by="cli",
            step_description="dir",
            tool_call_summary="{}",
        )
        approval.save(tmp_path)

        result = find_pending_or_approved_approval("desktop", "whoami", [], approval_dir=tmp_path)
        assert result is None

    def test_ignores_denied_approvals(self, tmp_path: Path) -> None:
        step_id = _command_step_id("desktop", "whoami", [])
        approval = WorkflowApproval(
            workflow_id=PC_RUN_WORKFLOW_ID,
            step_id=step_id,
            status="denied",
            requested_by="cli",
            step_description="whoami",
            tool_call_summary="{}",
        )
        approval.save(tmp_path)

        result = find_pending_or_approved_approval("desktop", "whoami", [], approval_dir=tmp_path)
        assert result is None

    def test_ignores_unrelated_workflow_id(self, tmp_path: Path) -> None:
        step_id = _command_step_id("desktop", "whoami", [])
        approval = WorkflowApproval(
            workflow_id="some_other_workflow",
            step_id=step_id,
            status="approved",
            requested_by="cli",
            step_description="whoami",
            tool_call_summary="{}",
        )
        approval.save(tmp_path)

        result = find_pending_or_approved_approval("desktop", "whoami", [], approval_dir=tmp_path)
        assert result is None

    def test_finds_pending_approval(self, tmp_path: Path) -> None:
        step_id = _command_step_id("desktop", "whoami", [])
        approval = WorkflowApproval(
            workflow_id=PC_RUN_WORKFLOW_ID,
            step_id=step_id,
            status="pending",
            requested_by="cli",
            step_description="whoami",
            tool_call_summary="{}",
        )
        approval.save(tmp_path)

        result = find_pending_or_approved_approval("desktop", "whoami", [], approval_dir=tmp_path)
        assert result is not None
        assert result.approval_id == approval.approval_id

    def test_finds_approved_approval(self, tmp_path: Path) -> None:
        approved = _save_approved_approval("desktop", "whoami", [], tmp_path)
        result = find_pending_or_approved_approval("desktop", "whoami", [], approval_dir=tmp_path)
        assert result is not None
        assert result.approval_id == approved.approval_id


# ===========================================================================
# CLI integration tests
# ===========================================================================


class TestPcRunCliPolicyIntegration:
    """Test cmd_pc with policy+approval flow using mocked dependencies."""

    def _make_service_mock(
        self,
        *,
        allowlist_matched: bool = True,
        allowed_cmds: list[str] | None = None,
    ) -> MagicMock:
        """Build a mock ComputerService whose allowlist check is controllable."""
        if allowed_cmds is None:
            allowed_cmds = ["whoami", "dir", "ipconfig"]
        service = MagicMock()
        service.get_command_allowed.return_value = (allowlist_matched, allowed_cmds)
        service.run.return_value = type(
            "RunResult",
            (),
            {
                "stdout": "alice\n",
                "stderr": "",
                "ok": True,
                "error": None,
                "exit_code": 0,
            },
        )()
        return service

    # ------------------------------------------------------------------
    # Policy requires approval: command is allowlisted, no prior approval
    # ------------------------------------------------------------------

    def test_approval_required_prints_approval_id(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        tmp_path: Path,
    ) -> None:
        from rex.cli import cmd_pc

        service = self._make_service_mock()
        monkeypatch.setattr("rex.computers.service.get_computer_service", lambda: service)
        monkeypatch.setattr("rex.computers.pc_run_policy.DEFAULT_APPROVAL_DIR", tmp_path)

        args = argparse.Namespace(
            pc_command="run", id="desktop", cmd=["whoami"], yes=True, user=None
        )
        code = cmd_pc(args)

        assert code == 1
        service.run.assert_not_called()
        out = capsys.readouterr().out
        assert "Approval required" in out
        assert "rex approvals --approve" in out
        assert "desktop" in out
        assert "whoami" in out

    def test_approval_required_command_not_executed(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        from rex.cli import cmd_pc

        service = self._make_service_mock()
        monkeypatch.setattr("rex.computers.service.get_computer_service", lambda: service)
        monkeypatch.setattr("rex.computers.pc_run_policy.DEFAULT_APPROVAL_DIR", tmp_path)

        args = argparse.Namespace(
            pc_command="run", id="desktop", cmd=["whoami"], yes=True, user=None
        )
        cmd_pc(args)

        service.run.assert_not_called()

    # ------------------------------------------------------------------
    # Denied policy path
    # ------------------------------------------------------------------

    def test_denied_policy_refuses_before_network(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        tmp_path: Path,
    ) -> None:
        from rex.cli import cmd_pc

        service = self._make_service_mock()
        monkeypatch.setattr("rex.computers.service.get_computer_service", lambda: service)
        monkeypatch.setattr("rex.computers.pc_run_policy.DEFAULT_APPROVAL_DIR", tmp_path)

        denied_decision = PolicyDecision(
            allowed=False,
            reason="pc_run is explicitly denied",
            requires_approval=False,
            denied=True,
        )

        def _mock_check(*args, **kwargs):
            return denied_decision, None

        monkeypatch.setattr("rex.computers.pc_run_policy.check_pc_run_policy", _mock_check)

        args = argparse.Namespace(
            pc_command="run", id="desktop", cmd=["whoami"], yes=True, user=None
        )
        code = cmd_pc(args)

        assert code == 1
        service.run.assert_not_called()
        out = capsys.readouterr().out
        assert "denied by policy" in out
        assert "pc_run is explicitly denied" in out

    # ------------------------------------------------------------------
    # Approved approval + --yes → executes
    # ------------------------------------------------------------------

    def test_approved_approval_with_yes_executes(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        tmp_path: Path,
    ) -> None:
        from rex.cli import cmd_pc

        _save_approved_approval("desktop", "whoami", [], tmp_path)
        service = self._make_service_mock()
        monkeypatch.setattr("rex.computers.service.get_computer_service", lambda: service)
        monkeypatch.setattr("rex.computers.pc_run_policy.DEFAULT_APPROVAL_DIR", tmp_path)

        args = argparse.Namespace(
            pc_command="run", id="desktop", cmd=["whoami"], yes=True, user=None
        )
        code = cmd_pc(args)

        assert code == 0
        service.run.assert_called_once_with("desktop", "whoami", args=[])
        out = capsys.readouterr().out
        assert "alice" in out

    # ------------------------------------------------------------------
    # Approved approval without --yes → --yes guard fires
    # ------------------------------------------------------------------

    def test_approved_approval_without_yes_refuses(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        tmp_path: Path,
    ) -> None:
        from rex.cli import cmd_pc

        _save_approved_approval("desktop", "whoami", [], tmp_path)
        service = self._make_service_mock()
        monkeypatch.setattr("rex.computers.service.get_computer_service", lambda: service)
        monkeypatch.setattr("rex.computers.pc_run_policy.DEFAULT_APPROVAL_DIR", tmp_path)

        args = argparse.Namespace(
            pc_command="run", id="desktop", cmd=["whoami"], yes=False, user=None
        )
        code = cmd_pc(args)

        assert code == 1
        service.run.assert_not_called()
        out = capsys.readouterr().out
        assert "without explicit confirmation" in out
        assert "--yes" in out

    # ------------------------------------------------------------------
    # Non-allowlisted command → refused before any approval or network call
    # ------------------------------------------------------------------

    def test_non_allowlisted_command_refused(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        tmp_path: Path,
    ) -> None:
        from rex.cli import cmd_pc

        service = self._make_service_mock(
            allowlist_matched=False,
            allowed_cmds=["whoami", "dir"],
        )
        monkeypatch.setattr("rex.computers.service.get_computer_service", lambda: service)
        monkeypatch.setattr("rex.computers.pc_run_policy.DEFAULT_APPROVAL_DIR", tmp_path)

        args = argparse.Namespace(
            pc_command="run", id="desktop", cmd=["rm", "-rf", "/"], yes=True, user=None
        )
        code = cmd_pc(args)

        assert code == 1
        service.run.assert_not_called()
        out = capsys.readouterr().out
        assert "not on the allowlist" in out
        # No approval file created
        assert list(tmp_path.glob("*.json")) == []

    # ------------------------------------------------------------------
    # Auto-execute policy (custom low-risk) + --yes → executes
    # ------------------------------------------------------------------

    def test_auto_execute_policy_with_yes_executes(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        tmp_path: Path,
    ) -> None:
        from rex.cli import cmd_pc

        auto_decision = PolicyDecision(
            allowed=True,
            reason="Low-risk tool with auto-execute enabled",
            requires_approval=False,
            denied=False,
        )

        def _mock_check(*args, **kwargs):
            return auto_decision, None

        service = self._make_service_mock()
        monkeypatch.setattr("rex.computers.service.get_computer_service", lambda: service)
        monkeypatch.setattr("rex.computers.pc_run_policy.check_pc_run_policy", _mock_check)

        args = argparse.Namespace(
            pc_command="run", id="desktop", cmd=["whoami"], yes=True, user=None
        )
        code = cmd_pc(args)

        assert code == 0
        service.run.assert_called_once_with("desktop", "whoami", args=[])
