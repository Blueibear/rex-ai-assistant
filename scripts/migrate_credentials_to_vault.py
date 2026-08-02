"""Migrate legacy plaintext desktop credentials into the OS-backed vault.

The command is dry-run by default. Apply mode is transactional per source:
all destination entries are verified, the secret-free opaque-reference
registry is atomically persisted and read back, and only then is the source
atomically sanitized. No plaintext backup or secret-derived output is made.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

KNOWN_SECRET_ENV_VARS: tuple[str, ...] = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "OLLAMA_API_KEY",
    "HA_TOKEN",
    "HA_SECRET",
    "BRAVE_API_KEY",
    "SERPAPI_KEY",
    "SERPAPI_API_KEY",
    "GOOGLE_API_KEY",
    "GOOGLE_CSE_ID",
    "OPENWEATHERMAP_API_KEY",
    "REX_SPEAK_API_KEY",
    "REX_TOOL_API_KEY",
    "REX_JWT_SECRET",
    "REX_PROXY_TOKEN",
    "REX_AGENT_TOKEN",
    "OPENCLAW_GATEWAY_TOKEN",
    "TWILIO_ACCOUNT_SID",
    "TWILIO_AUTH_TOKEN",
    "TWILIO_FROM_NUMBER",
    "TWILIO_PHONE_NUMBER",
    "TWILIO_PHONE_ACCOUNT_SID",
    "TWILIO_PHONE_AUTH_TOKEN",
    "TWILIO_TRANSFER_NUMBER",
    "TELEGRAM_BOT_TOKEN",
    "GITHUB_TOKEN",
    "ELEVENLABS_API_KEY",
    "PUSH_TOKEN",
)
_RECOVERY_JOURNAL_VERSION = 1
_RECOVERY_JOURNAL_NAME = "credential_migration_recovery.json"


class MigrationError(RuntimeError):
    """A source or reference registry could not be safely processed."""


@dataclass(frozen=True)
class MigrationCandidate:
    logical_name: str
    value: str
    source: str
    integration: str
    account: str | None
    slot: str


@dataclass(frozen=True)
class MigrationResult:
    logical_name: str
    source: str
    status: str
    detail: str | None = None


def _context_for_name(name: str) -> tuple[str, str | None, str]:
    from rex.credentials import credential_context_for_name

    return credential_context_for_name(name)


def _read_env_candidates(path: Path) -> list[MigrationCandidate]:
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise MigrationError("The environment source could not be read") from exc
    from dotenv import dotenv_values

    wanted = set(KNOWN_SECRET_ENV_VARS)
    parsed = dotenv_values(path)
    candidates: list[MigrationCandidate] = []
    seen: set[str] = set()
    for line in lines:
        key = _env_key(line)
        if key not in wanted:
            continue
        if key in seen:
            raise MigrationError("The environment source contains duplicate credential keys")
        seen.add(key)
        value = parsed.get(key)
        if not value:
            continue
        integration, account, slot = _context_for_name(key)
        candidates.append(MigrationCandidate(key, value, "env", integration, account, slot))
    return candidates


def _read_json_candidates(path: Path) -> list[MigrationCandidate]:
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise MigrationError("The credential JSON source is invalid") from exc
    if not isinstance(raw, dict):
        raise MigrationError("The credential JSON source is invalid")
    section = raw.get("credentials", raw)
    if not isinstance(section, dict):
        raise MigrationError("The credential JSON source is invalid")
    from rex.credentials import DEFAULT_CREDENTIAL_MAPPING

    candidates: list[MigrationCandidate] = []
    for service_value, credential in section.items():
        service = str(service_value)
        if isinstance(credential, str):
            value = credential
        elif isinstance(credential, dict) and isinstance(credential.get("token"), str):
            value = credential["token"]
        else:
            continue
        if not value:
            continue
        logical_name = DEFAULT_CREDENTIAL_MAPPING.get(service, service)
        integration, account, slot = _context_for_name(logical_name)
        candidates.append(
            MigrationCandidate(
                logical_name,
                value,
                "credentials.json",
                integration,
                account,
                slot,
            )
        )
    return candidates


def discover_candidates(*, env_path: Path, credentials_json_path: Path) -> list[MigrationCandidate]:
    return _read_env_candidates(env_path) + _read_json_candidates(credentials_json_path)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with open(temp, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        try:
            temp.unlink()
        except FileNotFoundError:
            pass


def _env_key(line: str) -> str:
    match = re.match(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=", line)
    return match.group(1) if match else ""


def _sanitize_env(path: Path, logical_names: set[str]) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    kept: list[str] = []
    for line in lines:
        key = _env_key(line)
        if key in logical_names:
            continue
        kept.append(line)
    _atomic_write_text(path, ("\n".join(kept) + "\n") if kept else "")


def _sanitize_credentials_json(path: Path, logical_names: set[str]) -> None:
    raw = json.loads(path.read_text(encoding="utf-8"))
    section = raw.get("credentials", raw)
    from rex.credentials import DEFAULT_CREDENTIAL_MAPPING

    for service in list(section):
        if DEFAULT_CREDENTIAL_MAPPING.get(str(service), str(service)) in logical_names:
            del section[service]
    _atomic_write_text(path, json.dumps(raw, indent=2) + "\n")


def _load_registry(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        return {}
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise MigrationError("The reference registry config is invalid") from exc
    if not isinstance(raw, dict):
        raise MigrationError("The reference registry config is invalid")
    return raw


def _recovery_journal_path(config_path: Path) -> Path:
    return config_path.with_name(_RECOVERY_JOURNAL_NAME)


def _load_recovery_journal(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise MigrationError("The encrypted-rollback recovery journal is invalid") from exc
    if (
        not isinstance(raw, dict)
        or set(raw) != {"version", "entries"}
        or raw.get("version") != _RECOVERY_JOURNAL_VERSION
        or not isinstance(raw.get("entries"), list)
    ):
        raise MigrationError("The encrypted-rollback recovery journal is invalid")
    entries: list[dict[str, Any]] = []
    required = {
        "ref",
        "logical_name",
        "source",
        "scope",
        "owner",
        "integration",
        "account",
        "slot",
    }
    from rex.credential_vault import validate_credential_ref

    for entry in raw["entries"]:
        if not isinstance(entry, dict) or set(entry) != required:
            raise MigrationError("The encrypted-rollback recovery journal is invalid")
        try:
            validate_credential_ref(entry["ref"])
        except (TypeError, ValueError) as exc:
            raise MigrationError("The encrypted-rollback recovery journal is invalid") from exc
        if (
            not isinstance(entry["logical_name"], str)
            or entry["source"] not in {"env", "credentials.json"}
            or entry["scope"] not in {"household", "user"}
            or not isinstance(entry["owner"], str)
        ):
            raise MigrationError("The encrypted-rollback recovery journal is invalid")
        expected = _context_for_name(entry["logical_name"])
        if (entry["integration"], entry["account"], entry["slot"]) != expected:
            raise MigrationError("The encrypted-rollback recovery journal is invalid")
        entries.append(entry)
    return entries


def _write_recovery_journal(path: Path, entries: list[dict[str, Any]]) -> None:
    if not entries:
        path.unlink(missing_ok=True)
        return
    payload = {"version": _RECOVERY_JOURNAL_VERSION, "entries": entries}
    _atomic_write_text(path, json.dumps(payload, indent=2) + "\n")
    if _load_recovery_journal(path) != entries:
        raise MigrationError("The encrypted-rollback recovery journal could not be verified")


def _journal_retained_entries(
    config_path: Path,
    *,
    source_name: str,
    scope: str,
    owner: str,
    retained: list[tuple[MigrationCandidate, str]],
) -> None:
    if not retained:
        return
    path = _recovery_journal_path(config_path)
    entries = _load_recovery_journal(path)
    known_refs = {entry["ref"] for entry in entries}
    for candidate, ref in retained:
        if ref in known_refs:
            continue
        entries.append(
            {
                "ref": ref,
                "logical_name": candidate.logical_name,
                "source": source_name,
                "scope": scope,
                "owner": owner,
                "integration": candidate.integration,
                "account": candidate.account,
                "slot": candidate.slot,
            }
        )
    _write_recovery_journal(path, entries)


def _cleanup_recovery_journal(config_path: Path, *, vault: Any, scope: str, owner: str) -> None:
    path = _recovery_journal_path(config_path)
    entries = _load_recovery_journal(path)
    if not entries:
        return
    active_config = _load_registry(config_path)
    active_refs: set[str] = set()
    refs_root = active_config.get("credential_refs", {})
    if not isinstance(refs_root, dict):
        raise MigrationError("The reference registry config is invalid")
    for scoped_section in refs_root.values():
        if not isinstance(scoped_section, dict):
            raise MigrationError("The reference registry config is invalid")
        for possible_user_section in scoped_section.values():
            if isinstance(possible_user_section, dict) and isinstance(
                possible_user_section.get("ref"), str
            ):
                active_refs.add(possible_user_section["ref"])
            elif isinstance(possible_user_section, dict):
                for record in possible_user_section.values():
                    if isinstance(record, dict) and isinstance(record.get("ref"), str):
                        active_refs.add(record["ref"])

    remaining: list[dict[str, Any]] = []
    for entry in entries:
        if entry["scope"] != scope or entry["owner"] != owner:
            remaining.append(entry)
            continue
        if entry["ref"] in active_refs:
            raise MigrationError("A recovery reference is unexpectedly active")
        try:
            deleted = vault.delete_secret(
                entry["ref"],
                integration=entry["integration"],
                account=entry["account"],
                slot=entry["slot"],
            )
        except Exception:
            remaining.append(entry)
        else:
            if not deleted:
                continue
    _write_recovery_journal(path, remaining)
    if any(entry["scope"] == scope and entry["owner"] == owner for entry in remaining):
        raise MigrationError("Encrypted rollback cleanup is still pending")


def _registry_section(
    config: dict[str, Any], *, scope: str, owner: str, create: bool
) -> dict[str, Any]:
    refs = config.setdefault("credential_refs", {}) if create else config.get("credential_refs", {})
    if not isinstance(refs, dict):
        raise MigrationError("The credential reference registry is invalid")
    if scope == "household":
        section = refs.setdefault("household", {}) if create else refs.get("household", {})
    else:
        users = refs.setdefault("users", {}) if create else refs.get("users", {})
        if not isinstance(users, dict):
            raise MigrationError("The credential reference registry is invalid")
        section = users.setdefault(owner, {}) if create else users.get(owner, {})
    if not isinstance(section, dict):
        raise MigrationError("The credential reference registry is invalid")
    return section


def _validate_scope_owner(scope: str, owner: str) -> str:
    if scope == "household":
        if owner != "household":
            raise MigrationError("Household migration owner must be 'household'")
        return owner
    if scope != "user":
        raise MigrationError("Migration scope must be 'household' or 'user'")
    if owner.casefold() == "household":
        raise MigrationError("User migration owner must identify a Rex user")
    from rex.identity import validate_user_id

    try:
        return validate_user_id(owner)
    except ValueError as exc:
        raise MigrationError("Migration owner is invalid") from exc


def _migrate_source(
    *,
    source_path: Path,
    source_name: str,
    candidates: list[MigrationCandidate],
    vault: Any,
    config_path: Path,
    scope: str,
    owner: str,
) -> list[MigrationResult]:
    from rex.credential_vault import (
        VaultCorruptedError,
        generate_credential_ref,
        validate_credential_ref,
    )

    config = _load_registry(config_path)
    original_config_text = config_path.read_text(encoding="utf-8") if config_path.exists() else None
    section = _registry_section(config, scope=scope, owner=owner, create=True)
    prepared: list[tuple[MigrationCandidate, str, bool, str]] = []
    conflict_results: list[MigrationResult] = []

    logical_names = [candidate.logical_name for candidate in candidates]
    if len(logical_names) != len(set(logical_names)):
        return [
            MigrationResult(candidate.logical_name, source_name, "conflict")
            for candidate in candidates
        ]

    for candidate in candidates:
        record = section.get(candidate.logical_name)
        if record is not None:
            if (
                not isinstance(record, dict)
                or set(record)
                not in (
                    {"ref", "integration", "account", "slot"},
                    {"ref", "integration", "account", "slot", "migrated_from"},
                )
                or any(
                    (
                        record.get("integration") != candidate.integration,
                        record.get("account") != candidate.account,
                        record.get("slot") != candidate.slot,
                        not isinstance(record.get("ref"), str),
                        "migrated_from" in record and record.get("migrated_from") != source_name,
                    )
                )
            ):
                conflict_results.append(
                    MigrationResult(candidate.logical_name, source_name, "conflict")
                )
                continue
            ref = record["ref"]
            try:
                validate_credential_ref(ref)
                existing = vault.get_secret(
                    ref,
                    integration=candidate.integration,
                    account=candidate.account,
                    slot=candidate.slot,
                )
            except (VaultCorruptedError, ValueError):
                conflict_results.append(
                    MigrationResult(candidate.logical_name, source_name, "conflict")
                )
                continue
            if existing != candidate.value:
                conflict_results.append(
                    MigrationResult(candidate.logical_name, source_name, "conflict")
                )
                continue
            prepared.append((candidate, ref, False, "already_migrated"))
        else:
            prepared.append((candidate, generate_credential_ref(), True, "migrated"))

    if conflict_results:
        conflicts = {result.logical_name for result in conflict_results}
        return [
            result
            for candidate in candidates
            for result in [
                MigrationResult(
                    candidate.logical_name,
                    source_name,
                    "conflict" if candidate.logical_name in conflicts else "blocked",
                )
            ]
        ]

    new_entries: list[tuple[MigrationCandidate, str]] = []
    config_written = False
    try:
        for candidate, ref, is_new, _status in prepared:
            if is_new:
                vault.set_secret(
                    ref,
                    candidate.value,
                    integration=candidate.integration,
                    account=candidate.account,
                    slot=candidate.slot,
                )
                new_entries.append((candidate, ref))
                if (
                    vault.get_secret(
                        ref,
                        integration=candidate.integration,
                        account=candidate.account,
                        slot=candidate.slot,
                    )
                    != candidate.value
                ):
                    raise MigrationError("Vault verification failed")
            section[candidate.logical_name] = {
                "ref": ref,
                "integration": candidate.integration,
                "account": candidate.account,
                "slot": candidate.slot,
                "migrated_from": source_name,
            }

        _atomic_write_text(config_path, json.dumps(config, indent=2) + "\n")
        config_written = True
        readback = _load_registry(config_path)
        readback_section = _registry_section(readback, scope=scope, owner=owner, create=False)
        for candidate, ref, _is_new, _status in prepared:
            record = readback_section.get(candidate.logical_name)
            if not isinstance(record, dict) or (
                record.get("ref") != ref
                or record.get("integration") != candidate.integration
                or record.get("account") != candidate.account
                or record.get("slot") != candidate.slot
                or record.get("migrated_from") != source_name
            ):
                raise MigrationError("Reference registry verification failed")

        names = {candidate.logical_name for candidate in candidates}
        if source_name == "env":
            _sanitize_env(source_path, names)
        else:
            _sanitize_credentials_json(source_path, names)
    except Exception as exc:
        registry_restored = not config_written
        if config_written:
            try:
                if original_config_text is None:
                    config_path.unlink(missing_ok=True)
                else:
                    _atomic_write_text(config_path, original_config_text)
                registry_restored = True
            except Exception:
                registry_restored = False
        if registry_restored:
            retained: list[tuple[MigrationCandidate, str]] = []
            for candidate, ref in new_entries:
                try:
                    deleted = vault.delete_secret(
                        ref,
                        integration=candidate.integration,
                        account=candidate.account,
                        slot=candidate.slot,
                    )
                except Exception:
                    retained.append((candidate, ref))
                else:
                    if not deleted:
                        continue
            _journal_retained_entries(
                config_path,
                source_name=source_name,
                scope=scope,
                owner=owner,
                retained=retained,
            )
        if isinstance(exc, MigrationError):
            raise
        raise MigrationError("Credential migration apply failed") from exc

    return [
        MigrationResult(candidate.logical_name, source_name, status)
        for candidate, _ref, _is_new, status in prepared
    ]


def migrate(
    *,
    env_path: Path,
    credentials_json_path: Path,
    config_path: Path,
    scope: str,
    owner: str,
    apply: bool = False,
) -> list[MigrationResult]:
    owner = _validate_scope_owner(scope, owner)
    candidates = discover_candidates(env_path=env_path, credentials_json_path=credentials_json_path)
    if not apply:
        return [
            MigrationResult(candidate.logical_name, candidate.source, "planned")
            for candidate in candidates
        ]
    if not candidates:
        return []

    from rex.credential_vault import get_credential_vault

    vault = get_credential_vault(scope=scope, user_id=owner if scope == "user" else None)
    _cleanup_recovery_journal(config_path, vault=vault, scope=scope, owner=owner)
    results: list[MigrationResult] = []
    for source_name, source_path in (
        ("env", env_path),
        ("credentials.json", credentials_json_path),
    ):
        source_candidates = [c for c in candidates if c.source == source_name]
        if source_candidates:
            results.extend(
                _migrate_source(
                    source_path=source_path,
                    source_name=source_name,
                    candidates=source_candidates,
                    vault=vault,
                    config_path=config_path,
                    scope=scope,
                    owner=owner,
                )
            )
    return results


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-path", type=Path)
    parser.add_argument("--credentials-json-path", type=Path)
    parser.add_argument("--config-path", type=Path)
    parser.add_argument("--scope", required=True, choices=("household", "user"))
    parser.add_argument("--owner", required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    from rex.credentials import DEFAULT_CONFIG_PATH
    from rex.runtime_paths import config_path as runtime_config_path
    from rex.runtime_paths import env_path as resolve_env_path

    try:
        results = migrate(
            env_path=args.env_path or resolve_env_path(),
            credentials_json_path=args.credentials_json_path or DEFAULT_CONFIG_PATH,
            config_path=args.config_path or runtime_config_path(),
            scope=args.scope,
            owner=args.owner,
            apply=args.apply,
        )
    except Exception:
        results = [MigrationResult("migration", "operation", "failed")]

    if args.json:
        print(json.dumps([asdict(result) for result in results], indent=2))
    else:
        print(f"AskRex credential vault migration - {'APPLY' if args.apply else 'DRY RUN'}")
        if not results:
            print("No known plaintext secrets found to migrate.")
        for result in results:
            print(f"[{result.status}] {result.logical_name} (from {result.source})")
    return 1 if any(r.status in {"failed", "conflict", "blocked"} for r in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
