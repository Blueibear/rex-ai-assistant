"""Atomic household persistence for canonical speaker groups."""

from __future__ import annotations

import json
import os
import tempfile
import threading
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from rex.runtime_paths import household_data_path

from .models import MediaCapability

_SCHEMA_VERSION = 1
GroupRecord = dict[str, Any]
GroupRecords = list[GroupRecord]


@dataclass(frozen=True, slots=True)
class SpeakerGroup:
    """Persistent group identity and capabilities shared by every member."""

    id: str
    name: str
    member_ids: tuple[str, ...]
    capabilities: frozenset[MediaCapability]


class SpeakerGroupStore:
    """Persist stable-ID groups by atomically replacing one JSON file."""

    def __init__(
        self,
        path: Path | str | None = None,
        *,
        target_exists: Callable[[str], bool],
        target_capabilities: Callable[[str], Collection[MediaCapability]] | None = None,
    ) -> None:
        self._path = (
            Path(path) if path is not None else household_data_path("media", "speaker_groups.json")
        )
        self._target_exists = target_exists
        self._target_capabilities = target_capabilities
        self._lock = threading.RLock()

    def create(self, name: str, member_ids: Sequence[str]) -> SpeakerGroup:
        normalized_name = self._normalized_name(name)
        members = self._validated_members(member_ids)
        with self._lock:
            records = self._read_records()
            self._ensure_unique_name(records, normalized_name)
            record = {
                "id": f"group:{uuid4()}",
                "name": normalized_name,
                "member_ids": list(members),
            }
            records.append(record)
            self._validate_all_members(records)
            self._write_records(records)
            return self._to_group(record)

    def get(self, group_id: str) -> SpeakerGroup | None:
        with self._lock:
            record = self._find(self._read_records(), group_id)
            return self._to_group(record) if record is not None else None

    def list(self) -> tuple[SpeakerGroup, ...]:
        with self._lock:
            groups = (self._to_group(record) for record in self._read_records())
            return tuple(sorted(groups, key=lambda group: (group.name.casefold(), group.id)))

    def rename(self, group_id: str, name: str) -> SpeakerGroup:
        normalized_name = self._normalized_name(name)
        with self._lock:
            records = self._read_records()
            record = self._required(records, group_id)
            self._validated_members(record["member_ids"])
            self._ensure_unique_name(records, normalized_name, excluding_id=group_id)
            record["name"] = normalized_name
            self._validate_all_members(records)
            self._write_records(records)
            return self._to_group(record)

    def set_members(self, group_id: str, member_ids: Sequence[str]) -> SpeakerGroup:
        members = self._validated_members(member_ids)
        with self._lock:
            records = self._read_records()
            record = self._required(records, group_id)
            record["member_ids"] = list(members)
            self._validate_all_members(records)
            self._write_records(records)
            return self._to_group(record)

    def delete(self, group_id: str) -> bool:
        with self._lock:
            records = self._read_records()
            remaining = [record for record in records if record["id"] != group_id]
            if len(remaining) == len(records):
                return False
            self._validate_all_members(remaining)
            self._write_records(remaining)
            return True

    @staticmethod
    def _normalized_name(name: str) -> str:
        normalized = " ".join(name.split())
        if not normalized:
            raise ValueError("Speaker group name is required")
        return normalized

    def _validated_members(self, member_ids: Sequence[str]) -> tuple[str, ...]:
        members = tuple(member_ids)
        if not members:
            raise ValueError("Speaker group requires at least one audio target")
        if any(not isinstance(member_id, str) or not member_id for member_id in members):
            raise ValueError("Speaker group member IDs must be non-empty strings")
        if len(set(members)) != len(members):
            raise ValueError("Speaker group members must be unique audio targets")
        for member_id in members:
            if not self._target_exists(member_id):
                raise ValueError(f"Unknown audio target: {member_id}")
        return members

    def _validate_all_members(self, records: GroupRecords) -> None:
        for record in records:
            self._validated_members(record["member_ids"])

    @staticmethod
    def _ensure_unique_name(
        records: GroupRecords,
        name: str,
        *,
        excluding_id: str | None = None,
    ) -> None:
        normalized = name.casefold()
        for record in records:
            if (
                record["id"] != excluding_id
                and " ".join(str(record["name"]).split()).casefold() == normalized
            ):
                raise ValueError(f"Speaker group name already exists: {name}")

    def _to_group(self, record: GroupRecord) -> SpeakerGroup:
        member_ids = tuple(record["member_ids"])
        return SpeakerGroup(
            id=record["id"],
            name=record["name"],
            member_ids=member_ids,
            capabilities=self._capability_intersection(member_ids),
        )

    def _capability_intersection(
        self,
        member_ids: tuple[str, ...],
    ) -> frozenset[MediaCapability]:
        if self._target_capabilities is None:
            return frozenset()
        capabilities = [frozenset(self._target_capabilities(member_id)) for member_id in member_ids]
        return frozenset.intersection(*capabilities)

    def _read_records(self) -> GroupRecords:
        if not self._path.exists():
            return []
        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Speaker group store is unreadable: {exc}") from exc
        if not isinstance(payload, dict) or payload.get("version") != _SCHEMA_VERSION:
            raise ValueError("Speaker group store has an unsupported schema")
        raw_groups = payload.get("groups")
        if not isinstance(raw_groups, list):
            raise ValueError("Speaker group store groups must be a list")
        records = []
        group_ids: set[str] = set()
        group_names: set[str] = set()
        for raw_group in raw_groups:
            if not isinstance(raw_group, dict):
                raise ValueError("Speaker group entry must be an object")
            group_id = raw_group.get("id")
            name = raw_group.get("name")
            member_ids = raw_group.get("member_ids")
            if (
                not isinstance(group_id, str)
                or not isinstance(name, str)
                or not isinstance(member_ids, list)
                or not all(isinstance(member_id, str) for member_id in member_ids)
            ):
                raise ValueError("Speaker group entry is malformed")
            if not group_id.startswith("group:") or not group_id.removeprefix("group:"):
                raise ValueError("Speaker group ID must be a stable group ID")
            if group_id in group_ids:
                raise ValueError("Speaker group IDs must be unique")
            group_ids.add(group_id)

            normalized_name = self._normalized_name(name)
            if name != normalized_name:
                raise ValueError("Speaker group names must use normalized whitespace")
            casefolded_name = normalized_name.casefold()
            if casefolded_name in group_names:
                raise ValueError("Speaker group names must be unique")
            group_names.add(casefolded_name)

            records.append(
                {"id": group_id, "name": normalized_name, "member_ids": list(member_ids)}
            )
        self._validate_all_members(records)
        return records

    def _write_records(self, records: GroupRecords) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self._path.parent,
                prefix=f".{self._path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                json.dump({"version": _SCHEMA_VERSION, "groups": records}, handle, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self._path)
            temporary = None
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    @staticmethod
    def _find(records: GroupRecords, group_id: str) -> GroupRecord | None:
        return next((record for record in records if record["id"] == group_id), None)

    @classmethod
    def _required(cls, records: GroupRecords, group_id: str) -> GroupRecord:
        record = cls._find(records, group_id)
        if record is None:
            raise KeyError(f"Unknown speaker group: {group_id}")
        return record


__all__ = ["SpeakerGroup", "SpeakerGroupStore"]
