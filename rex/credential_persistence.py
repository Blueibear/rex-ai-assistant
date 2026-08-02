"""Transactional persistence for household secrets and opaque references."""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from rex.credential_vault import generate_credential_ref, get_credential_vault
from rex.credentials import credential_context_for_name


def _strict_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("Credential reference registry cannot be read") from exc
    if not isinstance(value, dict):
        raise RuntimeError("Credential reference registry is invalid")
    return value


def _record(
    config: dict[str, Any], logical_name: str, expected: tuple[str, str | None, str]
) -> dict[str, Any] | None:
    root = config.get("credential_refs")
    if root is None:
        return None
    if not isinstance(root, dict):
        raise RuntimeError("Credential reference registry is invalid")
    household = root.get("household")
    if household is None:
        return None
    if not isinstance(household, dict):
        raise RuntimeError("Household credential reference registry is invalid")
    value = household.get(logical_name)
    if value is None:
        return None
    integration, account, slot = expected
    from rex.credential_vault import validate_credential_ref

    allowed_fields = {"ref", "integration", "account", "slot"}
    if isinstance(value, dict) and "migrated_from" in value:
        allowed_fields.add("migrated_from")
    if (
        not isinstance(value, dict)
        or set(value) != allowed_fields
        or not isinstance(value.get("ref"), str)
        or value.get("integration") != integration
        or value.get("account") != account
        or value.get("slot") != slot
        or (
            "migrated_from" in value
            and value.get("migrated_from") not in {"env", "credentials.json"}
        )
    ):
        raise RuntimeError("Credential reference context is invalid")
    try:
        validate_credential_ref(value["ref"])
    except ValueError as exc:
        raise RuntimeError("Credential reference context is invalid") from exc
    return value


def persist_household_secrets(
    values: dict[str, str],
    *,
    config_path: Path | None = None,
    update_config: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, str]:
    """Write nonblank secrets and atomically replace their config references.

    Blank values mean unchanged. New vault entries are read back before the
    reference registry is written and read back. Old entries are deleted only
    after the new registry is durable. On failure, the original registry is
    restored before staged entries are removed.
    """
    from rex.config_manager import resolve_config_path, save_config

    path = resolve_config_path(config_path)
    original_exists = path.exists()
    original = _strict_config(path)
    updated = copy.deepcopy(original)
    vault = None
    staged: list[tuple[str, str, str | None, str]] = []
    replaced: list[tuple[str, str, str | None, str]] = []
    result: dict[str, str] = {}
    config_written = False
    try:
        if update_config is not None:
            update_config(updated)
        for logical_name, raw_secret in values.items():
            secret = raw_secret.strip()
            if not secret:
                continue
            refs = updated.setdefault("credential_refs", {})
            if not isinstance(refs, dict):
                raise RuntimeError("Credential reference registry is invalid")
            household = refs.setdefault("household", {})
            if not isinstance(household, dict):
                raise RuntimeError("Household credential reference registry is invalid")
            expected = credential_context_for_name(logical_name)
            old = _record(original, logical_name, expected)
            integration, account, slot = expected
            if vault is None:
                vault = get_credential_vault(scope="household")
            ref = generate_credential_ref()
            vault.set_secret(ref, secret, integration=integration, account=account, slot=slot)
            staged.append((ref, integration, account, slot))
            if vault.get_secret(ref, integration=integration, account=account, slot=slot) != secret:
                raise RuntimeError("Credential vault readback failed")
            household[logical_name] = {
                "ref": ref,
                "integration": integration,
                "account": account,
                "slot": slot,
            }
            result[logical_name] = ref
            if old is not None:
                replaced.append((old["ref"], integration, account, slot))

        if not staged and update_config is None:
            return {}
        save_config(updated, path)
        config_written = True
        readback = _strict_config(path)
        for logical_name, ref in result.items():
            record = _record(readback, logical_name, credential_context_for_name(logical_name))
            if record is None or record["ref"] != ref:
                raise RuntimeError("Credential reference readback failed")
    except Exception:
        restored = not config_written
        if config_written:
            try:
                if original_exists:
                    save_config(original, path)
                else:
                    path.unlink(missing_ok=True)
                restored = True
            except Exception:
                restored = False
        if restored:
            for ref, integration, account, slot in staged:
                try:
                    assert vault is not None
                    vault.delete_secret(ref, integration=integration, account=account, slot=slot)
                except (
                    Exception
                ):  # noqa: B110 - best-effort rollback cleanup must not hide the primary error
                    pass
        raise

    for ref, integration, account, slot in replaced:
        try:
            assert vault is not None
            vault.delete_secret(ref, integration=integration, account=account, slot=slot)
        except (
            Exception
        ):  # noqa: B110 - best-effort rollback cleanup must not hide the primary error
            pass
    return result


__all__ = ["persist_household_secrets"]
