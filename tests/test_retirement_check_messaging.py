"""Permanent retirement guard for rex/messaging_backends/ and rex/messaging_service.py.

These modules were retired in iter 91 (Phase 7 messaging retirement).
Migration path:
  notification.py (iter 88) → __init__.py + services.py (iter 89) → cli.py cmd_msg (iter 90)
  sms_tool.py converted to stub (iter 91) → files deleted (iter 91)

This test permanently asserts that the retired files do NOT exist.
If they reappear, something has gone wrong with the migration.
"""

from __future__ import annotations

import pathlib

REPO_ROOT = pathlib.Path(__file__).parent.parent

RETIRED_FILES = [
    "rex/messaging_service.py",
    "rex/messaging_backends/__init__.py",
    "rex/messaging_backends/base.py",
    "rex/messaging_backends/factory.py",
    "rex/messaging_backends/stub.py",
    "rex/messaging_backends/twilio_backend.py",
    "rex/messaging_backends/twilio_adapter.py",
    "rex/messaging_backends/twilio_signature.py",
    "rex/messaging_backends/webhook_wiring.py",
    "rex/messaging_backends/inbound_webhook.py",
    "rex/messaging_backends/inbound_store.py",
    "rex/messaging_backends/account_config.py",
    "rex/messaging_backends/message_router.py",
    "rex/messaging_backends/sms_receiver_stub.py",
    "rex/messaging_backends/sms_sender_stub.py",
]

RETIRED_DIRS = [
    "rex/messaging_backends",
]


class TestMessagingRetirementGuard:
    """Assert that retired messaging modules are gone for good."""

    def test_messaging_service_deleted(self):
        path = REPO_ROOT / "rex" / "messaging_service.py"
        assert not path.exists(), (
            f"{path} still exists — messaging_service.py was retired in iter 91 "
            "and must not be reintroduced"
        )

    def test_messaging_backends_dir_deleted(self):
        path = REPO_ROOT / "rex" / "messaging_backends"
        assert not path.exists(), (
            f"{path} still exists — rex/messaging_backends/ was retired in iter 91 "
            "and must not be reintroduced"
        )

    def test_no_imports_of_messaging_service_in_rex(self):
        """No rex/ module (outside openclaw/) should import from rex.messaging_service."""
        import ast

        rex_pkg = REPO_ROOT / "rex"
        violators = []
        for py_file in rex_pkg.rglob("*.py"):
            rel = py_file.relative_to(REPO_ROOT).as_posix()
            if "__pycache__" in rel or "openclaw" in rel:
                continue
            try:
                source = py_file.read_text(encoding="utf-8")
            except OSError:
                continue
            if "messaging_service" not in source:
                continue
            try:
                tree = ast.parse(source, filename=str(py_file))
            except SyntaxError:
                violators.append(rel)
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    if "messaging_service" in (node.module or ""):
                        violators.append(rel)
                        break
                if isinstance(node, ast.Import):
                    if any("messaging_service" in a.name for a in node.names):
                        violators.append(rel)
                        break

        assert (
            not violators
        ), f"These modules import from retired rex.messaging_service: {violators}"
