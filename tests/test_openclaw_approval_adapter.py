"""Tests for ApprovalAdapter — US-P3-019: Test approval adapter end-to-end.

End-to-end flow:
  create() → list_pending() sees it → approve()/deny() → reload shows updated status
"""

from __future__ import annotations

from rex.openclaw.approval_adapter import ApprovalAdapter
from rex.workflow import WorkflowApproval

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _adapter(tmp_path) -> ApprovalAdapter:
    """Return an ApprovalAdapter isolated to a temporary directory."""
    return ApprovalAdapter(approval_dir=tmp_path)


# ---------------------------------------------------------------------------
# TestApprovalAdapterInstantiation
# ---------------------------------------------------------------------------


class TestApprovalAdapterInstantiation:
    def test_import(self):
        from rex.openclaw import approval_adapter  # noqa: F401

    def test_no_args(self, tmp_path):
        adapter = _adapter(tmp_path)
        assert adapter.approval_dir == tmp_path

    def test_approval_dir_property(self, tmp_path):
        adapter = _adapter(tmp_path)
        assert adapter.approval_dir == tmp_path


# ---------------------------------------------------------------------------
# TestApprovalAdapterCreate
# ---------------------------------------------------------------------------


class TestApprovalAdapterCreate:
    def test_create_returns_workflow_approval(self, tmp_path):
        adapter = _adapter(tmp_path)
        result = adapter.create("wf_001", "step_001")
        assert isinstance(result, WorkflowApproval)

    def test_create_status_is_pending(self, tmp_path):
        adapter = _adapter(tmp_path)
        approval = adapter.create("wf_001", "step_001")
        assert approval.status == "pending"

    def test_create_workflow_id_stored(self, tmp_path):
        adapter = _adapter(tmp_path)
        approval = adapter.create("wf_abc", "step_xyz")
        assert approval.workflow_id == "wf_abc"

    def test_create_step_id_stored(self, tmp_path):
        adapter = _adapter(tmp_path)
        approval = adapter.create("wf_abc", "step_xyz")
        assert approval.step_id == "step_xyz"

    def test_create_persists_to_disk(self, tmp_path):
        adapter = _adapter(tmp_path)
        approval = adapter.create("wf_001", "step_001")
        json_files = list(tmp_path.glob("*.json"))
        assert len(json_files) == 1
        assert approval.approval_id in json_files[0].name

    def test_create_with_optional_fields(self, tmp_path):
        adapter = _adapter(tmp_path)
        approval = adapter.create(
            "wf_002",
            "step_002",
            step_description="Send email",
            tool_call_summary="email tool",
            requested_by="test_runner",
        )
        assert approval.step_description == "Send email"
        assert approval.tool_call_summary == "email tool"
        assert approval.requested_by == "test_runner"

    def test_create_generates_unique_ids(self, tmp_path):
        adapter = _adapter(tmp_path)
        a1 = adapter.create("wf_001", "step_001")
        a2 = adapter.create("wf_001", "step_002")
        assert a1.approval_id != a2.approval_id


# ---------------------------------------------------------------------------
# TestApprovalAdapterLoad
# ---------------------------------------------------------------------------


class TestApprovalAdapterLoad:
    def test_load_returns_approval_after_create(self, tmp_path):
        adapter = _adapter(tmp_path)
        approval = adapter.create("wf_load", "step_load")
        loaded = adapter.load(approval.approval_id)
        assert loaded is not None
        assert loaded.approval_id == approval.approval_id

    def test_load_returns_none_for_unknown_id(self, tmp_path):
        adapter = _adapter(tmp_path)
        result = adapter.load("nonexistent_id")
        assert result is None

    def test_load_preserves_workflow_id(self, tmp_path):
        adapter = _adapter(tmp_path)
        approval = adapter.create("wf_preserve", "step_1")
        loaded = adapter.load(approval.approval_id)
        assert loaded.workflow_id == "wf_preserve"


# ---------------------------------------------------------------------------
# TestApprovalAdapterListPending
# ---------------------------------------------------------------------------


class TestApprovalAdapterListPending:
    def test_list_pending_empty_dir(self, tmp_path):
        adapter = _adapter(tmp_path)
        assert adapter.list_pending() == []

    def test_list_pending_contains_new_approval(self, tmp_path):
        adapter = _adapter(tmp_path)
        approval = adapter.create("wf_list", "step_list")
        pending = adapter.list_pending()
        ids = [a.approval_id for a in pending]
        assert approval.approval_id in ids

    def test_list_pending_multiple(self, tmp_path):
        adapter = _adapter(tmp_path)
        a1 = adapter.create("wf_m", "step_1")
        a2 = adapter.create("wf_m", "step_2")
        pending = adapter.list_pending()
        ids = {a.approval_id for a in pending}
        assert a1.approval_id in ids
        assert a2.approval_id in ids

    def test_list_pending_only_shows_pending(self, tmp_path):
        adapter = _adapter(tmp_path)
        a1 = adapter.create("wf_filter", "step_1")
        a2 = adapter.create("wf_filter", "step_2")
        adapter.approve(a1.approval_id)
        pending = adapter.list_pending()
        ids = {a.approval_id for a in pending}
        assert a1.approval_id not in ids
        assert a2.approval_id in ids


# ---------------------------------------------------------------------------
# TestApprovalAdapterApprove — end-to-end flow
# ---------------------------------------------------------------------------


class TestApprovalAdapterApprove:
    def test_approve_returns_true_for_valid_id(self, tmp_path):
        adapter = _adapter(tmp_path)
        approval = adapter.create("wf_approve", "step_1")
        result = adapter.approve(approval.approval_id)
        assert result is True

    def test_approve_updates_status_to_approved(self, tmp_path):
        adapter = _adapter(tmp_path)
        approval = adapter.create("wf_approve", "step_1")
        adapter.approve(approval.approval_id, decided_by="alice")
        reloaded = adapter.load(approval.approval_id)
        assert reloaded is not None
        assert reloaded.status == "approved"

    def test_approve_returns_false_for_unknown_id(self, tmp_path):
        adapter = _adapter(tmp_path)
        result = adapter.approve("nonexistent_id")
        assert result is False

    def test_approve_removes_from_pending_list(self, tmp_path):
        adapter = _adapter(tmp_path)
        approval = adapter.create("wf_approve", "step_1")
        adapter.approve(approval.approval_id)
        pending = adapter.list_pending()
        ids = {a.approval_id for a in pending}
        assert approval.approval_id not in ids


# ---------------------------------------------------------------------------
# TestApprovalAdapterDeny — end-to-end flow
# ---------------------------------------------------------------------------


class TestApprovalAdapterDeny:
    def test_deny_returns_true_for_valid_id(self, tmp_path):
        adapter = _adapter(tmp_path)
        approval = adapter.create("wf_deny", "step_1")
        result = adapter.deny(approval.approval_id)
        assert result is True

    def test_deny_updates_status_to_denied(self, tmp_path):
        adapter = _adapter(tmp_path)
        approval = adapter.create("wf_deny", "step_1")
        adapter.deny(approval.approval_id, decided_by="bob", reason="Not allowed")
        reloaded = adapter.load(approval.approval_id)
        assert reloaded is not None
        assert reloaded.status == "denied"

    def test_deny_returns_false_for_unknown_id(self, tmp_path):
        adapter = _adapter(tmp_path)
        result = adapter.deny("nonexistent_id")
        assert result is False

    def test_deny_removes_from_pending_list(self, tmp_path):
        adapter = _adapter(tmp_path)
        approval = adapter.create("wf_deny", "step_1")
        adapter.deny(approval.approval_id, reason="Rejected")
        pending = adapter.list_pending()
        ids = {a.approval_id for a in pending}
        assert approval.approval_id not in ids


# ---------------------------------------------------------------------------
# TestApprovalAdapterRegister
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# TestApprovalAdapterEndToEnd
# ---------------------------------------------------------------------------


class TestApprovalAdapterEndToEnd:
    def test_create_list_approve_reload(self, tmp_path):
        """Full happy path: create → list_pending sees it → approve → reload shows approved."""
        adapter = _adapter(tmp_path)

        # Step 1: create
        approval = adapter.create(
            "wf_e2e",
            "step_e2e",
            step_description="Execute risky action",
            requested_by="workflow_runner",
        )
        approval_id = approval.approval_id

        # Step 2: list_pending shows it
        pending = adapter.list_pending()
        assert any(a.approval_id == approval_id for a in pending)

        # Step 3: approve
        ok = adapter.approve(approval_id, decided_by="admin")
        assert ok is True

        # Step 4: reload shows approved
        reloaded = adapter.load(approval_id)
        assert reloaded is not None
        assert reloaded.status == "approved"

        # Step 5: no longer in pending
        pending_after = adapter.list_pending()
        assert all(a.approval_id != approval_id for a in pending_after)

    def test_create_list_deny_reload(self, tmp_path):
        """Full deny path: create → list_pending sees it → deny → reload shows denied."""
        adapter = _adapter(tmp_path)

        approval = adapter.create("wf_e2e_deny", "step_deny")
        approval_id = approval.approval_id

        pending = adapter.list_pending()
        assert any(a.approval_id == approval_id for a in pending)

        ok = adapter.deny(approval_id, decided_by="user", reason="Not authorised")
        assert ok is True

        reloaded = adapter.load(approval_id)
        assert reloaded is not None
        assert reloaded.status == "denied"

        pending_after = adapter.list_pending()
        assert all(a.approval_id != approval_id for a in pending_after)

    def test_multiple_approvals_independent(self, tmp_path):
        """Approving one approval does not affect another."""
        adapter = _adapter(tmp_path)

        a1 = adapter.create("wf_multi", "step_1")
        a2 = adapter.create("wf_multi", "step_2")

        adapter.approve(a1.approval_id)

        r1 = adapter.load(a1.approval_id)
        r2 = adapter.load(a2.approval_id)

        assert r1.status == "approved"
        assert r2.status == "pending"
