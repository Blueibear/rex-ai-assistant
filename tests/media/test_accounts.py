from __future__ import annotations

import json
import threading
import time
from dataclasses import fields
from pathlib import Path

import pytest

from rex.credential_vault import generate_credential_ref
from rex.media.accounts import MediaAccountRef, MediaAccountStore


def _account_path(root: Path, user_id: str) -> Path:
    return root / user_id / "media" / "accounts.json"


def test_same_user_can_put_get_and_list_accounts(tmp_path: Path) -> None:
    store = MediaAccountStore(tmp_path)
    apple_ref = generate_credential_ref()
    sonos_ref = generate_credential_ref()

    apple = store.put("james", "apple_music", "main", apple_ref, "James Apple Music")
    sonos = store.put("james", "sonos", "home", sonos_ref, "James Sonos")

    assert apple == MediaAccountRef(
        user_id="james",
        provider="apple_music",
        account_id="main",
        credential_ref=apple_ref,
        display_name="James Apple Music",
    )
    assert store.get("james", "apple_music", "main") == apple
    assert store.list("james") == (apple, sonos)


def test_account_lookup_and_list_cannot_cross_user(tmp_path: Path) -> None:
    store = MediaAccountStore(tmp_path)
    james = store.put(
        "james", "apple_music", "main", generate_credential_ref(), "James Apple Music"
    )
    cole = store.put("cole", "sonos", "home", generate_credential_ref(), "Cole Sonos")

    assert store.get("cole", "apple_music", "main") is None
    assert store.get("james", "sonos", "home") is None
    assert store.list("james") == (james,)
    assert store.list("cole") == (cole,)
    assert _account_path(tmp_path, "james").is_file()
    assert _account_path(tmp_path, "cole").is_file()


@pytest.mark.parametrize("user_id", ["", "../cole", "two/users", "CON"])
@pytest.mark.parametrize("method", ["put", "get", "list"])
def test_account_operations_reject_invalid_user_ids(
    tmp_path: Path, user_id: str, method: str
) -> None:
    store = MediaAccountStore(tmp_path)

    with pytest.raises(ValueError, match="Invalid user_id"):
        if method == "put":
            store.put(
                user_id,
                "sonos",
                "main",
                generate_credential_ref(),
                "Home Sonos",
            )
        elif method == "get":
            store.get(user_id, "sonos", "main")
        else:
            store.list(user_id)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("provider", ""),
        ("provider", "Apple Music"),
        ("provider", "APPLE_MUSIC"),
        ("account_id", ""),
        ("account_id", "../main"),
        ("display_name", ""),
        ("display_name", "   "),
        ("display_name", "Bad\nName"),
    ],
)
def test_put_rejects_invalid_account_metadata(tmp_path: Path, field: str, value: str) -> None:
    values = {
        "provider": "apple_music",
        "account_id": "main",
        "display_name": "James Apple Music",
    }
    values[field] = value
    store = MediaAccountStore(tmp_path)

    with pytest.raises(ValueError):
        store.put(
            "james",
            values["provider"],
            values["account_id"],
            generate_credential_ref(),
            values["display_name"],
        )


def test_put_reuses_existing_credential_ref_validator(tmp_path: Path) -> None:
    store = MediaAccountStore(tmp_path)

    with pytest.raises(ValueError, match="valid opaque reference"):
        store.put("james", "apple_music", "main", "cred_j", "James Apple Music")


@pytest.mark.parametrize(
    "payload",
    [
        {"version": 1, "user_id": "cole", "accounts": []},
        {
            "version": 1,
            "user_id": "james",
            "accounts": [
                {
                    "provider": "apple_music",
                    "account_id": "main",
                    "credential_ref": "cred_j",
                    "display_name": "James Apple Music",
                }
            ],
        },
        {
            "version": 1,
            "user_id": "james",
            "accounts": [
                {
                    "provider": "apple_music",
                    "account_id": "main",
                    "credential_ref": "cred_00000000000000000000000000000000",
                    "display_name": "James Apple Music",
                    "access_token": "must-not-be-here",
                }
            ],
        },
    ],
)
def test_malformed_or_cross_user_persisted_metadata_fails_closed(
    tmp_path: Path, payload: dict[str, object]
) -> None:
    path = _account_path(tmp_path, "james")
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    store = MediaAccountStore(tmp_path)

    with pytest.raises(ValueError, match="account store|account entry|Credential reference"):
        store.list("james")


def test_apple_music_is_metadata_only_without_connection_status(tmp_path: Path) -> None:
    store = MediaAccountStore(tmp_path)
    account = store.put(
        "james",
        "apple_music",
        "main",
        generate_credential_ref(),
        "James Apple Music",
    )

    assert account.provider == "apple_music"
    assert {field.name for field in fields(MediaAccountRef)} == {
        "user_id",
        "provider",
        "account_id",
        "credential_ref",
        "display_name",
    }
    persisted = json.loads(_account_path(tmp_path, "james").read_text(encoding="utf-8"))
    serialized = json.dumps(persisted)
    assert "connected" not in serialized
    assert "authenticated" not in serialized
    assert "access_token" not in serialized


def test_concurrent_puts_across_same_root_stores_do_not_lose_updates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store_a = MediaAccountStore(tmp_path)
    store_b = MediaAccountStore(tmp_path)

    read_done = threading.Event()
    proceed = threading.Event()
    original_read = store_a._read_accounts

    def paused_read(user_id: str) -> tuple[MediaAccountRef, ...]:
        result = original_read(user_id)
        read_done.set()
        assert proceed.wait(timeout=5), "store_b never got a chance to race store_a"
        return result

    monkeypatch.setattr(store_a, "_read_accounts", paused_read)

    thread_a = threading.Thread(
        target=store_a.put,
        args=("james", "apple_music", "main", generate_credential_ref(), "James Apple Music"),
    )
    thread_a.start()
    assert read_done.wait(timeout=5), "store_a never reached its read pause point"

    thread_b = threading.Thread(
        target=store_b.put,
        args=("james", "sonos", "home", generate_credential_ref(), "James Sonos"),
    )
    thread_b.start()
    # Give store_b a real chance to race store_a's paused read-modify-write
    # transaction before letting store_a resume and write.
    time.sleep(0.2)
    proceed.set()

    thread_a.join(timeout=5)
    thread_b.join(timeout=5)
    assert not thread_a.is_alive()
    assert not thread_b.is_alive()

    fresh = MediaAccountStore(tmp_path)
    accounts = fresh.list("james")
    assert {(account.provider, account.account_id) for account in accounts} == {
        ("apple_music", "main"),
        ("sonos", "home"),
    }


def test_failed_atomic_replace_preserves_existing_account_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = MediaAccountStore(tmp_path)
    store.put("james", "sonos", "home", generate_credential_ref(), "James Sonos")
    path = _account_path(tmp_path, "james")
    original_payload = path.read_text(encoding="utf-8")

    def fail_replace(source: Path, destination: Path) -> None:
        raise OSError("simulated replace failure")

    monkeypatch.setattr("rex.media.accounts.os.replace", fail_replace)

    with pytest.raises(OSError, match="simulated replace failure"):
        store.put(
            "james",
            "apple_music",
            "main",
            generate_credential_ref(),
            "James Apple Music",
        )

    assert path.read_text(encoding="utf-8") == original_payload
    assert list(path.parent.glob("*.tmp")) == []
