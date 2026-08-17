from __future__ import annotations

import json

import pytest

from rex.media.groups import SpeakerGroupStore
from rex.media.models import MediaCapability


def test_group_rejects_unknown_member(tmp_path) -> None:
    store = SpeakerGroupStore(
        tmp_path / "groups.json",
        target_exists=lambda target_id: False,
    )

    with pytest.raises(ValueError, match="Unknown audio target: missing:target"):
        store.create("Downstairs", ["missing:target"])


def test_group_rejects_empty_members(tmp_path) -> None:
    store = SpeakerGroupStore(
        tmp_path / "groups.json",
        target_exists=lambda target_id: True,
    )

    with pytest.raises(ValueError, match="at least one audio target"):
        store.create("Downstairs", [])


def test_group_names_are_unique_after_casefold_and_whitespace_normalization(tmp_path) -> None:
    store = SpeakerGroupStore(
        tmp_path / "groups.json",
        target_exists=lambda target_id: True,
    )
    store.create("Downstairs Speakers", ["sonos:office"])

    with pytest.raises(ValueError, match="Speaker group name already exists"):
        store.create("  downstairs   SPEAKERS  ", ["bose:kitchen"])


def test_groups_persist_stable_ids_and_member_target_ids(tmp_path) -> None:
    path = tmp_path / "groups.json"
    target_ids = {"sonos:office", "bose:kitchen"}
    first_store = SpeakerGroupStore(path, target_exists=target_ids.__contains__)

    created = first_store.create("Downstairs", ["sonos:office", "bose:kitchen"])
    reopened = SpeakerGroupStore(path, target_exists=target_ids.__contains__)

    assert created.id.startswith("group:")
    assert reopened.get(created.id) == created
    assert reopened.list() == (created,)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload == {
        "version": 1,
        "groups": [
            {
                "id": created.id,
                "name": "Downstairs",
                "member_ids": ["sonos:office", "bose:kitchen"],
            }
        ],
    }


def test_mixed_provider_group_capabilities_are_member_intersection(tmp_path) -> None:
    capabilities = {
        "sonos:office": frozenset(
            {
                MediaCapability.PLAY,
                MediaCapability.PAUSE,
                MediaCapability.SET_VOLUME,
            }
        ),
        "ha:media_player.kitchen": frozenset(
            {
                MediaCapability.PLAY,
                MediaCapability.PAUSE,
                MediaCapability.NEXT,
            }
        ),
    }
    store = SpeakerGroupStore(
        tmp_path / "groups.json",
        target_exists=capabilities.__contains__,
        target_capabilities=capabilities.__getitem__,
    )

    group = store.create(
        "Downstairs",
        ["sonos:office", "ha:media_player.kitchen"],
    )

    assert group.member_ids == ("sonos:office", "ha:media_player.kitchen")
    assert group.capabilities == frozenset({MediaCapability.PLAY, MediaCapability.PAUSE})


def test_rename_and_set_members_validate_before_persisting(tmp_path) -> None:
    known_targets = {"sonos:office", "bose:kitchen"}
    path = tmp_path / "groups.json"
    store = SpeakerGroupStore(path, target_exists=known_targets.__contains__)
    group = store.create("Downstairs", ["sonos:office"])
    original_payload = path.read_text(encoding="utf-8")

    known_targets.remove("sonos:office")
    with pytest.raises(ValueError, match="Unknown audio target: sonos:office"):
        store.rename(group.id, "First Floor")
    with pytest.raises(ValueError, match="Unknown audio target: missing:target"):
        store.set_members(group.id, ["bose:kitchen", "missing:target"])

    assert path.read_text(encoding="utf-8") == original_payload


def test_any_group_mutation_rejects_members_that_became_unresolved(tmp_path) -> None:
    known_targets = {"sonos:office", "bose:kitchen"}
    path = tmp_path / "groups.json"
    store = SpeakerGroupStore(path, target_exists=known_targets.__contains__)
    store.create("Downstairs", ["sonos:office"])
    original_payload = path.read_text(encoding="utf-8")
    known_targets.remove("sonos:office")

    with pytest.raises(ValueError, match="Unknown audio target: sonos:office"):
        store.create("Upstairs", ["bose:kitchen"])

    assert path.read_text(encoding="utf-8") == original_payload


@pytest.mark.parametrize("read_method", ["get", "list"])
def test_group_reads_reject_members_that_became_unresolved(tmp_path, read_method: str) -> None:
    known_targets = {"sonos:office"}
    store = SpeakerGroupStore(
        tmp_path / "groups.json",
        target_exists=known_targets.__contains__,
    )
    group = store.create("Downstairs", ["sonos:office"])
    known_targets.clear()

    with pytest.raises(ValueError, match="Unknown audio target: sonos:office"):
        if read_method == "get":
            store.get(group.id)
        else:
            store.list()


@pytest.mark.parametrize(
    ("groups", "message"),
    [
        (
            [{"id": "group:one", "name": "Downstairs", "member_ids": []}],
            "at least one audio target",
        ),
        (
            [
                {
                    "id": "group:one",
                    "name": "Downstairs",
                    "member_ids": ["sonos:office", "sonos:office"],
                }
            ],
            "members must be unique",
        ),
        (
            [{"id": "", "name": "Downstairs", "member_ids": ["sonos:office"]}],
            "ID must be a stable group ID",
        ),
        (
            [
                {"id": "group:one", "name": "Downstairs", "member_ids": ["sonos:office"]},
                {"id": "group:one", "name": "Upstairs", "member_ids": ["bose:bedroom"]},
            ],
            "IDs must be unique",
        ),
        (
            [{"id": "group:one", "name": "  ", "member_ids": ["sonos:office"]}],
            "name is required",
        ),
        (
            [
                {"id": "group:one", "name": "Downstairs", "member_ids": ["sonos:office"]},
                {"id": "group:two", "name": "downstairs", "member_ids": ["bose:bedroom"]},
            ],
            "names must be unique",
        ),
    ],
)
def test_group_reads_reject_malformed_persisted_records(
    tmp_path,
    groups: list[dict[str, object]],
    message: str,
) -> None:
    path = tmp_path / "groups.json"
    path.write_text(json.dumps({"version": 1, "groups": groups}), encoding="utf-8")
    store = SpeakerGroupStore(path, target_exists=lambda target_id: True)

    with pytest.raises(ValueError, match=message):
        store.list()


def test_group_mutations_persist_and_delete(tmp_path) -> None:
    known_targets = {"sonos:office", "bose:kitchen"}
    path = tmp_path / "groups.json"
    store = SpeakerGroupStore(path, target_exists=known_targets.__contains__)
    group = store.create("Downstairs", ["sonos:office"])

    renamed = store.rename(group.id, "First Floor")
    updated = store.set_members(group.id, ["bose:kitchen"])
    deleted = store.delete(group.id)

    assert renamed.name == "First Floor"
    assert updated.member_ids == ("bose:kitchen",)
    assert deleted is True
    assert store.get(group.id) is None
    assert store.list() == ()


def test_failed_atomic_replace_preserves_previous_group_file(tmp_path, monkeypatch) -> None:
    path = tmp_path / "groups.json"
    store = SpeakerGroupStore(path, target_exists=lambda target_id: True)
    store.create("Downstairs", ["sonos:office"])
    original_payload = path.read_text(encoding="utf-8")

    def fail_replace(source, destination) -> None:
        raise OSError("simulated replace failure")

    monkeypatch.setattr("rex.media.groups.os.replace", fail_replace)

    with pytest.raises(OSError, match="simulated replace failure"):
        store.create("Upstairs", ["bose:bedroom"])

    assert path.read_text(encoding="utf-8") == original_payload
    assert list(tmp_path.glob("*.tmp")) == []
