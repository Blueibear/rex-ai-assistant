"""Per-user email credential isolation through contextual opaque references."""

from __future__ import annotations

import sys

import pytest

from rex.credential_vault import InMemoryCredentialVault, VaultCorruptedError, get_credential_vault
from rex.credentials import CredentialManager, set_credential_manager

REF = "cred_" + "E" * 32
CONTEXT = {"integration": "email", "account": "personal", "slot": "password"}
REFS = {
    "email:personal": {
        "ref": REF,
        "integration": "email",
        "account": "personal",
        "slot": "password",
    }
}


def _manager(user_id: str, vault) -> CredentialManager:
    return CredentialManager(
        vault=vault,
        vault_refs=REFS,
        scope="user",
        user_id=user_id,
    )


def test_injected_user_vaults_keep_account_secrets_separate():
    alice_vault = InMemoryCredentialVault(scope="user", user_id="alice")
    bob_vault = InMemoryCredentialVault(scope="user", user_id="bob")
    alice_vault.set_secret(REF, "alice-secret", **CONTEXT)
    bob_vault.set_secret(REF, "bob-secret", **CONTEXT)

    assert _manager("alice", alice_vault).get_token("email:personal", **CONTEXT) == "alice-secret"
    assert _manager("bob", bob_vault).get_token("email:personal", **CONTEXT) == "bob-secret"


def test_shared_store_reference_swap_across_user_fails_closed():
    shared: dict = {}
    alice = InMemoryCredentialVault(scope="user", user_id="alice", store=shared)
    bob = InMemoryCredentialVault(scope="user", user_id="bob", store=shared)
    alice.set_secret(REF, "alice-secret", **CONTEXT)
    with pytest.raises(VaultCorruptedError):
        bob.get_secret(REF, **CONTEXT)


def test_reference_swap_across_account_or_slot_fails_closed():
    vault = InMemoryCredentialVault(scope="user", user_id="alice")
    vault.set_secret(REF, "alice-secret", **CONTEXT)
    with pytest.raises(VaultCorruptedError):
        vault.get_secret(REF, integration="email", account="work", slot="password")
    with pytest.raises(VaultCorruptedError):
        vault.get_secret(REF, integration="email", account="personal", slot="client_secret")


def test_user_scoped_manager_cannot_be_installed_globally():
    manager = _manager("alice", InMemoryCredentialVault(scope="user", user_id="alice"))
    with pytest.raises(ValueError):
        set_credential_manager(manager)


@pytest.mark.skipif(sys.platform != "win32", reason="Real DPAPI isolation is Windows-only")
def test_real_dpapi_user_reference_swap_fails_closed(tmp_path):
    shared_path = tmp_path / "shared-vault.json"
    alice = get_credential_vault(scope="user", user_id="alice", vault_path_override=shared_path)
    alice.set_secret(REF, "alice-secret", **CONTEXT)
    bob = get_credential_vault(scope="user", user_id="bob", vault_path_override=shared_path)
    with pytest.raises(VaultCorruptedError):
        bob.get_secret(REF, **CONTEXT)
