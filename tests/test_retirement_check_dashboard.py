"""Permanent retirement guard for retired dashboard modules (US-P7-014).

These modules were retired in iter 93 after all callers were migrated:
  - rex/health.py — check_dashboard_db removed
  - rex/retention.py — setup_dashboard_cleanup_job converted to no-op
  - rex/notification.py — _send_to_dashboard converted to logging-only stub
  - rex/digest_job.py — TYPE_CHECKING import removed; store param typed Any
  - rex/messaging_backends/inbound_store.py — retired with messaging_service (iter 91)
  - rex/gui_app.py — _create_flask_app converted to stub routes (iter 92)
"""

from __future__ import annotations

import pathlib

REPO_ROOT = pathlib.Path(__file__).parent.parent
REX_PKG = REPO_ROOT / "rex"


class TestDashboardRetired:
    def test_dashboard_store_deleted(self):
        assert not (
            REX_PKG / "dashboard_store.py"
        ).exists(), "rex/dashboard_store.py was re-introduced — it is a retired module"

    def test_legacy_dashboard_routes_deleted(self):
        for module_name in ("auth.py", "routes.py"):
            assert not (
                REX_PKG / "dashboard" / module_name
            ).exists(), f"rex/dashboard/{module_name} was re-introduced — it is retired"

    def test_dashboard_contract_deleted(self):
        assert not (
            REX_PKG / "contracts" / "dashboard.py"
        ).exists(), "rex/contracts/dashboard.py was re-introduced — it is a retired contract"
