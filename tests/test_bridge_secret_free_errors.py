"""S4: credential-bearing bridges never leak secret-derived exception content.

Each of these bridges resolves credentials (vault entries, tokens, passwords)
somewhere on its execution path. A raw ``str(exc))``/traceback response can
embed provider responses, tokens, or submitted request values. These tests
force a failure whose message carries a unique marker and assert the marker
never reaches the JSON the bridge prints - only a fixed, pre-written category
message may appear.
"""

from __future__ import annotations

import importlib
import json
from io import StringIO
from unittest.mock import patch

MARKER = "s3cr3t-marker-should-never-leak"  # pragma: allowlist secret


def _run(module_name: str, payload: dict, *, patches: list[tuple[str, object]]) -> dict:
    module = importlib.import_module(module_name)
    importlib.reload(module)
    captured = StringIO()
    with (
        patch.object(module.sys, "stdin", StringIO(json.dumps(payload))),
        patch.object(module.sys, "stdout", captured),
    ):
        ctxs = [patch(target, side_effect=RuntimeError(MARKER)) for target, _ in patches]
        for ctx in ctxs:
            ctx.__enter__()
        try:
            try:
                module.main()
            except SystemExit:
                pass
        finally:
            for ctx in reversed(ctxs):
                ctx.__exit__(None, None, None)
    output = captured.getvalue().strip()
    return json.loads(output)


def test_credential_vault_bridge_never_leaks_exception_marker():
    from rex.credential_vault import InMemoryCredentialVault

    vault = InMemoryCredentialVault(scope="user", user_id="alice")
    with patch("rex.credential_vault.get_credential_vault", return_value=vault):
        with patch.object(vault, "get_secret", side_effect=RuntimeError(MARKER)):
            module = importlib.import_module("rex_credential_vault_bridge")
            importlib.reload(module)
            captured = StringIO()
            payload = {
                "action": "get",
                "key": "cred_" + "B" * 32,
                "scope": "user",
                "request_user_id": "alice",
                "integration": "email",
                "account": "primary",
                "slot": "password",
            }
            with (
                patch.object(module.sys, "stdin", StringIO(json.dumps(payload))),
                patch.object(module.sys, "stdout", captured),
            ):
                try:
                    module.main()
                except SystemExit:
                    pass
    result = json.loads(captured.getvalue().strip())
    serialized = json.dumps(result)
    assert MARKER not in serialized
    assert "traceback" not in result
    assert result["ok"] is False


def test_sms_bridge_never_leaks_exception_marker():
    result = _run(
        "rex_sms_bridge",
        {"command": "list_threads", "user": "alice", "data_scope": "private"},
        patches=[("rex_sms_bridge._handle_list_threads", None)],
    )
    serialized = json.dumps(result)
    assert MARKER not in serialized
    assert "traceback" not in result
    assert result["ok"] is False


def test_email_bridge_never_leaks_exception_marker():
    result = _run(
        "rex_email_bridge",
        {"command": "list", "user": "default", "data_scope": "private"},
        patches=[("rex_email_bridge._resolve_user", None)],
    )
    serialized = json.dumps(result)
    assert MARKER not in serialized
    assert "traceback" not in result
    assert result["ok"] is False


def test_calendar_bridge_never_leaks_exception_marker():
    result = _run(
        "rex_calendar_bridge",
        {"command": "list", "user": "default", "data_scope": "private"},
        patches=[("rex_calendar_bridge._resolve_user", None)],
    )
    serialized = json.dumps(result)
    assert MARKER not in serialized
    assert "traceback" not in result
    assert result["ok"] is False


def test_ha_mutation_bridge_never_leaks_exception_marker():
    result = _run(
        "rex_ha_mutation_bridge",
        {
            "user": "alice",
            "data_scope": "private",
            "entity_id": "light.kitchen",
            "domain": "light",
            "service": "turn_on",
            "request_id": "req-1",
        },
        patches=[("rex_ha_mutation_bridge.validate_user_id", None)],
    )
    serialized = json.dumps(result)
    assert MARKER not in serialized
    assert "traceback" not in result
    assert result["ok"] is False
