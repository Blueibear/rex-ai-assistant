"""Private metadata storage for user-owned media provider accounts."""

from __future__ import annotations

import json
import os
import re
import tempfile
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rex.credential_vault import validate_credential_ref
from rex.identity import validate_user_id
from rex.runtime_paths import user_data_path

_SCHEMA_VERSION = 1
_PROVIDER_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_ACCOUNT_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:@-]{0,127}$")
_MAX_DISPLAY_NAME_LENGTH = 200
_ACCOUNT_KEYS = frozenset({"provider", "account_id", "credential_ref", "display_name"})

_ACCOUNT_LOCKS_GUARD = threading.Lock()
_ACCOUNT_LOCKS: dict[Path, threading.RLock] = {}


def _account_path_lock(path: Path) -> threading.RLock:
    """Return one process-wide lock shared by every store using this path.

    ``os.replace`` only makes a single file replacement atomic; it does not
    make the read-modify-write ``put`` transaction atomic across separate
    ``MediaAccountStore`` instances pointed at the same account file. Keying
    the lock by resolved path lets unrelated users' files proceed
    independently while serializing same-file mutations.
    """
    resolved = path.resolve()
    with _ACCOUNT_LOCKS_GUARD:
        lock = _ACCOUNT_LOCKS.get(resolved)
        if lock is None:
            lock = threading.RLock()
            _ACCOUNT_LOCKS[resolved] = lock
        return lock


def _validate_provider(provider: str) -> str:
    if not isinstance(provider, str) or not _PROVIDER_PATTERN.fullmatch(provider):
        raise ValueError("Media account provider is invalid")
    return provider


def _validate_account_id(account_id: str) -> str:
    if not isinstance(account_id, str) or not _ACCOUNT_ID_PATTERN.fullmatch(account_id):
        raise ValueError("Media account ID is invalid")
    return account_id


def _validate_display_name(display_name: str) -> str:
    if (
        not isinstance(display_name, str)
        or not display_name
        or display_name != display_name.strip()
        or len(display_name) > _MAX_DISPLAY_NAME_LENGTH
        or any(ord(character) < 32 or ord(character) == 127 for character in display_name)
    ):
        raise ValueError("Media account display name is invalid")
    return display_name


@dataclass(frozen=True, slots=True)
class MediaAccountRef:
    """Non-secret reference to one user-owned provider account."""

    user_id: str
    provider: str
    account_id: str
    credential_ref: str
    display_name: str

    def __post_init__(self) -> None:
        validate_user_id(self.user_id)
        _validate_provider(self.provider)
        _validate_account_id(self.account_id)
        validate_credential_ref(self.credential_ref)
        _validate_display_name(self.display_name)


class MediaAccountStore:
    """Persist metadata in physically separate, user-owned JSON files."""

    def __init__(self, root: Path | str | None = None) -> None:
        self._root = Path(root) if root is not None else None
        self._lock = threading.RLock()

    def put(
        self,
        user_id: str,
        provider: str,
        account_id: str,
        credential_ref: str,
        display_name: str,
    ) -> MediaAccountRef:
        """Create or replace non-secret account metadata for one user."""
        account = MediaAccountRef(
            user_id=user_id,
            provider=provider,
            account_id=account_id,
            credential_ref=credential_ref,
            display_name=display_name,
        )
        with _account_path_lock(self._path(account.user_id)):
            accounts = list(self._read_accounts(account.user_id))
            accounts = [
                existing
                for existing in accounts
                if (existing.provider, existing.account_id)
                != (account.provider, account.account_id)
            ]
            accounts.append(account)
            accounts.sort(key=lambda item: (item.provider, item.account_id))
            self._write_accounts(account.user_id, accounts)
        return account

    def get(self, user_id: str, provider: str, account_id: str) -> MediaAccountRef | None:
        """Return metadata only from the requested user's private partition."""
        user_id = validate_user_id(user_id)
        provider = _validate_provider(provider)
        account_id = _validate_account_id(account_id)
        with self._lock:
            return next(
                (
                    account
                    for account in self._read_accounts(user_id)
                    if account.provider == provider and account.account_id == account_id
                ),
                None,
            )

    def list(self, user_id: str) -> tuple[MediaAccountRef, ...]:
        """List only metadata owned by the requested user."""
        user_id = validate_user_id(user_id)
        with self._lock:
            return self._read_accounts(user_id)

    def _path(self, user_id: str) -> Path:
        user_id = validate_user_id(user_id)
        if self._root is None:
            return user_data_path(user_id, "media", "accounts.json")
        return self._root / user_id / "media" / "accounts.json"

    def _read_accounts(self, user_id: str) -> tuple[MediaAccountRef, ...]:
        path = self._path(user_id)
        if not path.exists():
            return ()
        try:
            payload: Any = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Media account store is unreadable: {exc}") from exc

        if (
            not isinstance(payload, dict)
            or set(payload) != {"version", "user_id", "accounts"}
            or payload.get("version") != _SCHEMA_VERSION
            or payload.get("user_id") != user_id
            or not isinstance(payload.get("accounts"), list)
        ):
            raise ValueError("Media account store has invalid ownership or schema")

        accounts: list[MediaAccountRef] = []
        keys: set[tuple[str, str]] = set()
        for raw_account in payload["accounts"]:
            if not isinstance(raw_account, dict) or set(raw_account) != _ACCOUNT_KEYS:
                raise ValueError("Media account entry is malformed")
            if not all(isinstance(raw_account[key], str) for key in _ACCOUNT_KEYS):
                raise ValueError("Media account entry is malformed")
            account = MediaAccountRef(user_id=user_id, **raw_account)
            key = (account.provider, account.account_id)
            if key in keys:
                raise ValueError("Media account entries must be unique")
            keys.add(key)
            accounts.append(account)
        return tuple(sorted(accounts, key=lambda item: (item.provider, item.account_id)))

    def _write_accounts(self, user_id: str, accounts: Sequence[MediaAccountRef]) -> None:
        path = self._path(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": _SCHEMA_VERSION,
            "user_id": user_id,
            "accounts": [
                {
                    "provider": account.provider,
                    "account_id": account.account_id,
                    "credential_ref": account.credential_ref,
                    "display_name": account.display_name,
                }
                for account in accounts
            ],
        }
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                json.dump(payload, handle, indent=2, ensure_ascii=False)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            temporary = None
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


__all__ = ["MediaAccountRef", "MediaAccountStore"]
