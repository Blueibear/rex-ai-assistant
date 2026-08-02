"""Electron credential-vault bridge authorization tests."""

from __future__ import annotations

import importlib
import json
from io import StringIO
from unittest.mock import patch

from rex.credential_vault import InMemoryCredentialVault

MODULE_NAME = "rex_credential_vault_bridge"
REF = "cred_" + "B" * 32
CONTEXT = {
    "scope": "user",
    "request_user_id": "alice",
    "integration": "email",
    "account": "primary",
    "slot": "password",
}


def _run(payload) -> dict:
    module = importlib.import_module(MODULE_NAME)
    importlib.reload(module)
    captured = StringIO()
    raw = payload if isinstance(payload, str) else json.dumps(payload)
    with patch("sys.stdin", StringIO(raw)), patch("sys.stdout", captured):
        try:
            module.main()
        except SystemExit:
            pass
    return json.loads(captured.getvalue().strip())


def test_set_generates_opaque_ref_and_never_echoes_secret():
    vault = InMemoryCredentialVault(scope="user", user_id="alice")
    with patch("rex.credential_vault.get_credential_vault", return_value=vault):
        result = _run({"action": "set", "value": "secret-marker", **CONTEXT})
    assert result["ok"] is True
    assert result["ref"].startswith("cred_")
    assert "secret-marker" not in json.dumps(result)
    assert (
        vault.get_secret(result["ref"], integration="email", account="primary", slot="password")
        == "secret-marker"
    )


def test_set_readback_failure_removes_staged_secret_and_reports_no_value():
    class LyingVault(InMemoryCredentialVault):
        def get_secret(self, *args, **kwargs):
            return "different"

    vault = LyingVault(scope="user", user_id="alice")
    with patch("rex.credential_vault.get_credential_vault", return_value=vault):
        result = _run({"action": "set", "value": "secret-marker", **CONTEXT})
    assert result == {"ok": False, "error": "Credential vault operation failed"}
    assert vault.list_entries() == []
    assert "secret-marker" not in json.dumps(result)


def test_get_has_delete_require_exact_context():
    vault = InMemoryCredentialVault(scope="user", user_id="alice")
    vault.set_secret(REF, "secret", integration="email", account="primary", slot="password")
    with patch("rex.credential_vault.get_credential_vault", return_value=vault):
        assert _run({"action": "has", "key": REF, **CONTEXT}) == {"ok": True, "has": True}
        assert _run({"action": "get", "key": REF, **CONTEXT}) == {
            "ok": True,
            "value": "secret",
        }
        assert _run({"action": "delete", "key": REF, **CONTEXT}) == {
            "ok": True,
            "deleted": True,
        }


def test_missing_identity_scope_or_context_fails_closed():
    vault = InMemoryCredentialVault()
    incomplete = [
        {
            "action": "get",
            "key": REF,
            **{k: v for k, v in CONTEXT.items() if k != "request_user_id"},
        },
        {"action": "get", "key": REF, **{k: v for k, v in CONTEXT.items() if k != "scope"}},
        {"action": "get", "key": REF, **{k: v for k, v in CONTEXT.items() if k != "slot"}},
    ]
    with patch("rex.credential_vault.get_credential_vault", return_value=vault):
        assert all(_run(payload)["ok"] is False for payload in incomplete)


def test_user_owner_is_derived_from_validated_requester_not_payload_authority():
    calls = []

    def capture(**kwargs):
        calls.append(kwargs)
        return InMemoryCredentialVault(scope="user", user_id=kwargs["user_id"])

    with patch("rex.credential_vault.get_credential_vault", capture):
        result = _run({"action": "list", "user_id": "bob", **CONTEXT})
    assert result == {"ok": True, "entries": []}
    assert calls == [{"scope": "user", "user_id": "alice"}]


def test_reference_swapping_and_errors_return_no_traceback_or_secret():
    vault = InMemoryCredentialVault(scope="user", user_id="alice")
    vault.set_secret(REF, "secret-marker", integration="email", account="primary", slot="password")
    swapped = {**CONTEXT, "action": "get", "key": REF, "account": "other"}
    with patch("rex.credential_vault.get_credential_vault", return_value=vault):
        result = _run(swapped)
    serialized = json.dumps(result)
    assert result["ok"] is False
    assert "traceback" not in result
    assert "secret-marker" not in serialized


def test_invalid_json_and_unknown_action_are_secret_free_errors():
    assert _run("not json")["ok"] is False
    result = _run({"action": "bogus"})
    assert result["ok"] is False
    assert "traceback" not in result
