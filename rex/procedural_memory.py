"""Guarded procedural experience memory built only from verified outcomes.

Procedures are declarative capability sequences, not executable Python/code blobs.  This
module deliberately has no hook from normal conversation or long-term memory writes:
creation requires an :class:`ActionLifecycleSnapshot` in the VERIFIED state.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from collections.abc import Callable, Iterable
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from rex.actions.lifecycle import ActionLifecycleSnapshot, ActionState
from rex.identity import validate_user_id
from rex.runtime_paths import household_data_path, user_data_path
from rex.tools.execution import ToolOperation, ToolRisk

_SCHEMA_VERSION = 1
_STORE_LOCK = threading.RLock()


class ProcedureScope(StrEnum):
    """Authority/storage scope for a learned procedure."""

    USER = "user"
    HOUSEHOLD = "household"


class ProcedureStatus(StrEnum):
    """Lifecycle state for a learned procedure."""

    PENDING_APPROVAL = "pending_approval"
    ACTIVE = "active"
    DISABLED = "disabled"
    REVOKED = "revoked"


class ProcedurePromotionError(ValueError):
    """Raised when unverified evidence is offered for procedural learning."""


class ProcedureStoreError(RuntimeError):
    """Raised when persisted procedure state fails closed validation."""


class ProcedureRevalidationPolicy(BaseModel):
    """Bounded rules controlling expiry, revalidation, and failure quarantine."""

    revalidate_after_seconds: int = Field(default=7 * 24 * 3600, ge=1)
    expires_after_seconds: int | None = Field(default=30 * 24 * 3600, ge=1)
    failure_threshold: int = Field(default=3, ge=1, le=100)

    model_config = ConfigDict(extra="forbid", frozen=True)


class ProcedureDefinition(BaseModel):
    """Declarative reusable procedure supplied to the verified-outcome promoter.

    ``steps`` are capability identifiers only.  Arguments, secrets, prompts, arbitrary
    code, and serialized callables are intentionally outside this contract.
    """

    name: str = Field(min_length=1, max_length=160)
    description: str = Field(default="", max_length=2000)
    capabilities: tuple[str, ...] = Field(min_length=1)
    required_permissions: tuple[str, ...] = ()
    operation: ToolOperation = ToolOperation.READ
    risk: ToolRisk = ToolRisk.SAFE
    version: str = Field(min_length=1, max_length=80)
    dependency_fingerprint: str = Field(min_length=1, max_length=256)
    steps: tuple[str, ...] = Field(min_length=1)
    revalidation: ProcedureRevalidationPolicy = Field(default_factory=ProcedureRevalidationPolicy)

    model_config = ConfigDict(extra="forbid", frozen=True)

    @model_validator(mode="after")
    def _validate_identifiers(self) -> ProcedureDefinition:
        capabilities = tuple(item.strip() for item in self.capabilities)
        permissions = tuple(item.strip() for item in self.required_permissions)
        steps = tuple(item.strip() for item in self.steps)
        if any(not item for item in (*capabilities, *steps)):
            raise ValueError("capability and step identifiers must not be empty")
        if any(not item for item in permissions):
            raise ValueError("permission identifiers must not be empty")
        if any(step not in capabilities for step in steps):
            raise ValueError("procedure steps must reference declared capabilities")
        if len(set(capabilities)) != len(capabilities):
            raise ValueError("procedure capabilities must be unique")
        if len(set(permissions)) != len(permissions):
            raise ValueError("procedure permissions must be unique")
        object.__setattr__(self, "capabilities", capabilities)
        object.__setattr__(self, "required_permissions", permissions)
        object.__setattr__(self, "steps", steps)
        object.__setattr__(self, "name", self.name.strip())
        object.__setattr__(self, "version", self.version.strip())
        object.__setattr__(self, "dependency_fingerprint", self.dependency_fingerprint.strip())
        return self


class ProcedureProvenance(BaseModel):
    """Immutable correlation references proving where a procedure came from."""

    action_id: str
    plan_id: str | None = None
    attempt_id: str
    verification_id: str
    audit_id: str
    user_result_id: str
    verified_at: datetime

    model_config = ConfigDict(extra="forbid", frozen=True)


class ProcedureAuditEvent(BaseModel):
    """Content-minimal lifecycle evidence retained with the procedure."""

    timestamp: datetime
    event: str = Field(min_length=1, max_length=96)
    actor_user_id: str
    reason: str | None = Field(default=None, max_length=160)
    evidence_ref: str | None = Field(default=None, max_length=256)

    model_config = ConfigDict(extra="forbid", frozen=True)


class ProcedureRecord(BaseModel):
    """Persisted learned procedure and its bounded trust/revalidation metadata."""

    procedure_id: str
    name: str
    description: str
    owner_id: str
    scope: ProcedureScope
    capabilities: tuple[str, ...]
    required_permissions: tuple[str, ...]
    operation: ToolOperation
    risk: ToolRisk
    version: str
    dependency_fingerprint: str
    steps: tuple[str, ...]
    provenance: ProcedureProvenance
    revalidation: ProcedureRevalidationPolicy
    status: ProcedureStatus
    approval_required: bool
    approved_by: str | None = None
    approved_at: datetime | None = None
    success_count: int = Field(default=1, ge=0)
    failure_count: int = Field(default=0, ge=0)
    consecutive_failures: int = Field(default=0, ge=0)
    created_at: datetime
    last_validated_at: datetime
    expires_at: datetime | None = None
    disabled_reason: str | None = None
    audit_history: tuple[ProcedureAuditEvent, ...] = ()

    model_config = ConfigDict(extra="forbid", frozen=True)


class ProceduralMemory:
    """Identity-scoped store/promoter for verified reusable procedures."""

    def __init__(
        self,
        *,
        base_dir: str | os.PathLike[str] | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._base_dir = Path(base_dir).resolve(strict=False) if base_dir is not None else None
        self._clock = clock or (lambda: datetime.now(UTC))

    def _now(self) -> datetime:
        now = self._clock()
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("procedural-memory clock must return a timezone-aware datetime")
        return now.astimezone(UTC)

    def _private_path(self, user_id: str) -> Path:
        owner = validate_user_id(user_id)
        if self._base_dir is not None:
            return self._base_dir / "users" / owner / "procedures.json"
        return user_data_path(owner, "procedures.json")

    def _household_path(self) -> Path:
        if self._base_dir is not None:
            return self._base_dir / "household" / "procedures.json"
        return household_data_path("procedures.json")

    def _path_for_record(self, record: ProcedureRecord) -> Path:
        return (
            self._private_path(record.owner_id)
            if record.scope is ProcedureScope.USER
            else self._household_path()
        )

    def _load(
        self,
        path: Path,
        *,
        expected_scope: ProcedureScope,
        expected_owner: str | None = None,
    ) -> list[ProcedureRecord]:
        if not path.exists():
            return []
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("schema_version") != _SCHEMA_VERSION:
                raise ProcedureStoreError("unsupported procedural-memory schema version")
            raw_records = payload.get("procedures")
            if not isinstance(raw_records, list):
                raise ProcedureStoreError("procedural-memory procedures must be a list")
            records = [ProcedureRecord.model_validate(item) for item in raw_records]
        except ProcedureStoreError:
            raise
        except Exception as exc:
            raise ProcedureStoreError(f"invalid procedural-memory store: {path.name}") from exc
        for record in records:
            if record.scope is not expected_scope:
                raise ProcedureStoreError("procedure scope does not match its storage boundary")
            if expected_owner is not None and record.owner_id != expected_owner:
                raise ProcedureStoreError("private procedure owner does not match storage boundary")
        return records

    def _write(self, path: Path, records: list[ProcedureRecord]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": _SCHEMA_VERSION,
            "procedures": [record.model_dump(mode="json") for record in records],
        }
        encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        temp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        try:
            temp.write_text(encoded, encoding="utf-8")
            os.replace(temp, path)
        finally:
            temp.unlink(missing_ok=True)

    @staticmethod
    def _definition_fingerprint(definition: ProcedureDefinition) -> str:
        encoded = json.dumps(
            definition.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @classmethod
    def _procedure_id(
        cls,
        *,
        owner_id: str,
        scope: ProcedureScope,
        verification_id: str,
        definition: ProcedureDefinition,
    ) -> str:
        material = "|".join(
            (
                owner_id,
                scope.value,
                verification_id,
                cls._definition_fingerprint(definition),
            )
        ).encode("utf-8")
        return f"proc_{hashlib.sha256(material).hexdigest()[:24]}"

    @staticmethod
    def _expires_at(now: datetime, policy: ProcedureRevalidationPolicy) -> datetime | None:
        if policy.expires_after_seconds is None:
            return None
        return now + timedelta(seconds=policy.expires_after_seconds)

    @staticmethod
    def _event(
        *,
        timestamp: datetime,
        event: str,
        actor_user_id: str,
        reason: str | None = None,
        evidence_ref: str | None = None,
    ) -> ProcedureAuditEvent:
        return ProcedureAuditEvent(
            timestamp=timestamp,
            event=event,
            actor_user_id=validate_user_id(actor_user_id),
            reason=reason,
            evidence_ref=evidence_ref,
        )

    def _insert(self, record: ProcedureRecord) -> ProcedureRecord:
        path = self._path_for_record(record)
        expected_owner = record.owner_id if record.scope is ProcedureScope.USER else None
        records = self._load(
            path,
            expected_scope=record.scope,
            expected_owner=expected_owner,
        )
        for existing in records:
            if existing.procedure_id == record.procedure_id:
                return existing
        records.append(record)
        records.sort(key=lambda item: (item.created_at, item.procedure_id))
        self._write(path, records)
        return record

    def _replace(self, record: ProcedureRecord) -> ProcedureRecord:
        path = self._path_for_record(record)
        expected_owner = record.owner_id if record.scope is ProcedureScope.USER else None
        records = self._load(
            path,
            expected_scope=record.scope,
            expected_owner=expected_owner,
        )
        replaced = False
        updated: list[ProcedureRecord] = []
        for existing in records:
            if existing.procedure_id == record.procedure_id:
                updated.append(record)
                replaced = True
            else:
                updated.append(existing)
        if not replaced:
            raise ProcedureStoreError("procedure disappeared from its storage boundary")
        self._write(path, updated)
        return record

    def _visible_record(self, procedure_id: str, *, requester_user_id: str) -> ProcedureRecord:
        requester = validate_user_id(requester_user_id)
        private = self._load(
            self._private_path(requester),
            expected_scope=ProcedureScope.USER,
            expected_owner=requester,
        )
        for record in private:
            if record.procedure_id == procedure_id:
                return record
        household = self._load(
            self._household_path(),
            expected_scope=ProcedureScope.HOUSEHOLD,
        )
        for record in household:
            if record.procedure_id == procedure_id:
                return record
        # Mask both non-existence and cross-user existence with the same fail-closed error.
        raise PermissionError("procedure is not accessible to this user")

    @staticmethod
    def _require_owner(record: ProcedureRecord, requester_user_id: str) -> str:
        requester = validate_user_id(requester_user_id)
        if requester != record.owner_id:
            raise PermissionError("procedure changes require the procedure owner")
        return requester

    def learn_from_verified_outcome(
        self,
        outcome: ActionLifecycleSnapshot,
        definition: ProcedureDefinition,
        *,
        owner_id: str,
        scope: ProcedureScope | str,
    ) -> ProcedureRecord:
        """Promote one independently verified action/workflow outcome.

        This is intentionally the only creation API.  Normal conversation and memory writes
        have no path to `_insert`, and an unverified/completed-only lifecycle is rejected.
        """
        owner = validate_user_id(owner_id)
        procedure_scope = ProcedureScope(scope)
        if outcome.state is not ActionState.VERIFIED:
            raise ProcedurePromotionError(
                "only verified action/workflow outcomes can become procedures"
            )
        now = self._now()
        correlation = outcome.correlation
        provenance = ProcedureProvenance(
            action_id=correlation.action_id,
            plan_id=correlation.plan_id,
            attempt_id=correlation.attempt_id,
            verification_id=correlation.verification_id,
            audit_id=correlation.audit_id,
            user_result_id=correlation.user_result_id,
            verified_at=now,
        )
        approval_required = (
            definition.operation is ToolOperation.MUTATION or definition.risk is not ToolRisk.SAFE
        )
        disabled_reason: str | None = None
        if definition.risk is ToolRisk.PROHIBITED:
            status = ProcedureStatus.REVOKED
            disabled_reason = "prohibited_risk"
        elif approval_required:
            status = ProcedureStatus.PENDING_APPROVAL
        else:
            status = ProcedureStatus.ACTIVE
        event_name = (
            "revoked_prohibited_risk"
            if status is ProcedureStatus.REVOKED
            else "promoted_from_verified_outcome"
        )
        record = ProcedureRecord(
            procedure_id=self._procedure_id(
                owner_id=owner,
                scope=procedure_scope,
                verification_id=correlation.verification_id,
                definition=definition,
            ),
            name=definition.name,
            description=definition.description,
            owner_id=owner,
            scope=procedure_scope,
            capabilities=definition.capabilities,
            required_permissions=definition.required_permissions,
            operation=definition.operation,
            risk=definition.risk,
            version=definition.version,
            dependency_fingerprint=definition.dependency_fingerprint,
            steps=definition.steps,
            provenance=provenance,
            revalidation=definition.revalidation,
            status=status,
            approval_required=approval_required,
            success_count=1,
            failure_count=0,
            consecutive_failures=0,
            created_at=now,
            last_validated_at=now,
            expires_at=self._expires_at(now, definition.revalidation),
            disabled_reason=disabled_reason,
            audit_history=(
                self._event(
                    timestamp=now,
                    event=event_name,
                    actor_user_id=owner,
                    reason=disabled_reason,
                    evidence_ref=correlation.verification_id,
                ),
            ),
        )
        with _STORE_LOCK:
            return self._insert(record)

    def get(self, procedure_id: str, *, requester_user_id: str) -> ProcedureRecord:
        with _STORE_LOCK:
            return self._visible_record(procedure_id, requester_user_id=requester_user_id)

    def list(
        self,
        *,
        requester_user_id: str,
        include_household: bool = False,
    ) -> list[ProcedureRecord]:
        requester = validate_user_id(requester_user_id)
        with _STORE_LOCK:
            records = self._load(
                self._private_path(requester),
                expected_scope=ProcedureScope.USER,
                expected_owner=requester,
            )
            if include_household:
                records.extend(
                    self._load(
                        self._household_path(),
                        expected_scope=ProcedureScope.HOUSEHOLD,
                    )
                )
            return sorted(records, key=lambda item: (item.created_at, item.procedure_id))

    def approve(
        self,
        procedure_id: str,
        *,
        requester_user_id: str,
        approver_user_id: str,
        confirmed: bool,
    ) -> ProcedureRecord:
        """Explicitly activate a risky procedure after owner-bound human approval."""
        with _STORE_LOCK:
            record = self._visible_record(procedure_id, requester_user_id=requester_user_id)
            requester = self._require_owner(record, requester_user_id)
            approver = validate_user_id(approver_user_id)
            if approver != requester:
                raise PermissionError("human approval identity must match the approving requester")
            if record.risk is ToolRisk.PROHIBITED:
                raise PermissionError("prohibited procedures cannot be activated")
            if record.status is ProcedureStatus.REVOKED:
                raise PermissionError("revoked procedures cannot be activated")
            if not record.approval_required and record.status is ProcedureStatus.ACTIVE:
                return record
            if record.status is not ProcedureStatus.PENDING_APPROVAL:
                raise PermissionError(
                    "disabled procedures require verified revalidation before approval"
                )
            if not confirmed:
                raise PermissionError("explicit human approval is required")
            now = self._now()
            updated = record.model_copy(
                update={
                    "status": ProcedureStatus.ACTIVE,
                    "approved_by": approver,
                    "approved_at": now,
                    "disabled_reason": None,
                    "audit_history": record.audit_history
                    + (
                        self._event(
                            timestamp=now,
                            event="human_approval_granted",
                            actor_user_id=approver,
                        ),
                    ),
                }
            )
            return self._replace(updated)

    def _disable_record(
        self,
        record: ProcedureRecord,
        *,
        actor_user_id: str,
        reason: str,
        event: str,
        evidence_ref: str | None = None,
    ) -> ProcedureRecord:
        now = self._now()
        updated = record.model_copy(
            update={
                "status": ProcedureStatus.DISABLED,
                "disabled_reason": reason,
                "audit_history": record.audit_history
                + (
                    self._event(
                        timestamp=now,
                        event=event,
                        actor_user_id=actor_user_id,
                        reason=reason,
                        evidence_ref=evidence_ref,
                    ),
                ),
            }
        )
        return self._replace(updated)

    def validate_for_execution(
        self,
        procedure_id: str,
        *,
        requester_user_id: str,
        dependency_fingerprint: str,
        version: str,
    ) -> ProcedureRecord:
        """Fail closed when trust evidence is stale, expired, or dependency-drifted."""
        with _STORE_LOCK:
            record = self._visible_record(procedure_id, requester_user_id=requester_user_id)
            requester = self._require_owner(record, requester_user_id)
            if record.status is not ProcedureStatus.ACTIVE:
                return record
            if version != record.version:
                return self._disable_record(
                    record,
                    actor_user_id=requester,
                    reason="version_drift",
                    event="disabled_version_drift",
                )
            if dependency_fingerprint != record.dependency_fingerprint:
                return self._disable_record(
                    record,
                    actor_user_id=requester,
                    reason="dependency_drift",
                    event="disabled_dependency_drift",
                )
            now = self._now()
            if record.expires_at is not None and now >= record.expires_at:
                return self._disable_record(
                    record,
                    actor_user_id=requester,
                    reason="expired",
                    event="disabled_expired",
                )
            due_at = record.last_validated_at + timedelta(
                seconds=record.revalidation.revalidate_after_seconds
            )
            if now >= due_at:
                return self._disable_record(
                    record,
                    actor_user_id=requester,
                    reason="revalidation_due",
                    event="disabled_revalidation_due",
                )
            return record

    def can_execute(
        self,
        procedure_id: str,
        *,
        requester_user_id: str,
        dependency_fingerprint: str,
        version: str,
        granted_permissions: Iterable[str],
        available_capabilities: Iterable[str],
    ) -> bool:
        """Return whether current authority and dependencies still permit execution.

        Stored permissions are requirements only; they never become grants.  Current
        permissions and capability availability must be supplied for every execution check.
        """
        with _STORE_LOCK:
            record = self._visible_record(procedure_id, requester_user_id=requester_user_id)
            requester = validate_user_id(requester_user_id)
            if record.owner_id != requester:
                # Household sharing permits inspection, not inherited execution authority.
                return False
            if record.status is not ProcedureStatus.ACTIVE:
                return False
            granted = set(granted_permissions)
            required = set(record.required_permissions)
            if "admin" not in granted and not required.issubset(granted):
                return False
            available = set(available_capabilities)
            if not set(record.capabilities).issubset(available):
                return False
            checked = self.validate_for_execution(
                procedure_id,
                requester_user_id=requester,
                dependency_fingerprint=dependency_fingerprint,
                version=version,
            )
            return checked.status is ProcedureStatus.ACTIVE

    def record_execution_outcome(
        self,
        procedure_id: str,
        *,
        requester_user_id: str,
        outcome: ActionLifecycleSnapshot,
        dependency_fingerprint: str,
        version: str,
    ) -> ProcedureRecord:
        """Update trust only from canonical terminal execution evidence."""
        with _STORE_LOCK:
            record = self._visible_record(procedure_id, requester_user_id=requester_user_id)
            requester = self._require_owner(record, requester_user_id)
            checked = self.validate_for_execution(
                procedure_id,
                requester_user_id=requester,
                dependency_fingerprint=dependency_fingerprint,
                version=version,
            )
            if checked.status is not ProcedureStatus.ACTIVE:
                return checked
            terminal_evidence = {
                ActionState.VERIFIED,
                ActionState.COMPLETED,
                ActionState.UNVERIFIED,
                ActionState.FAILED,
                ActionState.CANCELLED,
            }
            if outcome.state not in terminal_evidence:
                raise ValueError("procedure execution outcome must be terminal evidence")
            now = self._now()
            if outcome.state is ActionState.VERIFIED:
                updated = checked.model_copy(
                    update={
                        "success_count": checked.success_count + 1,
                        "consecutive_failures": 0,
                        "last_validated_at": now,
                        "expires_at": self._expires_at(now, checked.revalidation),
                        "audit_history": checked.audit_history
                        + (
                            self._event(
                                timestamp=now,
                                event="verified_execution",
                                actor_user_id=requester,
                                evidence_ref=outcome.correlation.verification_id,
                            ),
                        ),
                    }
                )
                return self._replace(updated)

            failure_count = checked.failure_count + 1
            consecutive = checked.consecutive_failures + 1
            audit = checked.audit_history + (
                self._event(
                    timestamp=now,
                    event="execution_not_verified",
                    actor_user_id=requester,
                    reason=outcome.state.value,
                    evidence_ref=outcome.correlation.audit_id,
                ),
            )
            updated = checked.model_copy(
                update={
                    "failure_count": failure_count,
                    "consecutive_failures": consecutive,
                    "audit_history": audit,
                }
            )
            if consecutive >= checked.revalidation.failure_threshold:
                updated = updated.model_copy(
                    update={
                        "status": ProcedureStatus.DISABLED,
                        "disabled_reason": "repeated_failure",
                        "audit_history": updated.audit_history
                        + (
                            self._event(
                                timestamp=now,
                                event="disabled_repeated_failure",
                                actor_user_id=requester,
                                reason="repeated_failure",
                                evidence_ref=outcome.correlation.audit_id,
                            ),
                        ),
                    }
                )
            return self._replace(updated)

    def revalidate(
        self,
        procedure_id: str,
        *,
        requester_user_id: str,
        outcome: ActionLifecycleSnapshot,
        dependency_fingerprint: str,
        version: str | None = None,
    ) -> ProcedureRecord:
        """Reactivate a disabled procedure only after new verified evidence."""
        if outcome.state is not ActionState.VERIFIED:
            raise ProcedurePromotionError("procedure revalidation requires a verified outcome")
        with _STORE_LOCK:
            record = self._visible_record(procedure_id, requester_user_id=requester_user_id)
            requester = self._require_owner(record, requester_user_id)
            if record.status is ProcedureStatus.REVOKED:
                raise PermissionError("revoked procedures cannot be revalidated")
            now = self._now()
            new_version = version or record.version
            trust_boundary_changed = (
                dependency_fingerprint != record.dependency_fingerprint
                or new_version != record.version
            )
            needs_approval = record.approval_required and (
                record.approved_by is None or trust_boundary_changed
            )
            new_status = (
                ProcedureStatus.PENDING_APPROVAL if needs_approval else ProcedureStatus.ACTIVE
            )
            updated = record.model_copy(
                update={
                    "dependency_fingerprint": dependency_fingerprint,
                    "version": new_version,
                    "status": new_status,
                    "approved_by": None if needs_approval else record.approved_by,
                    "approved_at": None if needs_approval else record.approved_at,
                    "consecutive_failures": 0,
                    "last_validated_at": now,
                    "expires_at": self._expires_at(now, record.revalidation),
                    "disabled_reason": None,
                    "success_count": record.success_count + 1,
                    "audit_history": record.audit_history
                    + (
                        self._event(
                            timestamp=now,
                            event=(
                                "revalidated_pending_approval" if needs_approval else "revalidated"
                            ),
                            actor_user_id=requester,
                            evidence_ref=outcome.correlation.verification_id,
                        ),
                    ),
                }
            )
            return self._replace(updated)

    def disable(
        self,
        procedure_id: str,
        *,
        requester_user_id: str,
        reason: str = "user_disabled",
    ) -> ProcedureRecord:
        with _STORE_LOCK:
            record = self._visible_record(procedure_id, requester_user_id=requester_user_id)
            requester = self._require_owner(record, requester_user_id)
            if record.status is ProcedureStatus.REVOKED:
                return record
            return self._disable_record(
                record,
                actor_user_id=requester,
                reason=reason,
                event="disabled_by_owner",
            )

    def revoke(
        self,
        procedure_id: str,
        *,
        requester_user_id: str,
        reason: str = "user_revoked",
    ) -> ProcedureRecord:
        with _STORE_LOCK:
            record = self._visible_record(procedure_id, requester_user_id=requester_user_id)
            requester = self._require_owner(record, requester_user_id)
            now = self._now()
            updated = record.model_copy(
                update={
                    "status": ProcedureStatus.REVOKED,
                    "disabled_reason": reason,
                    "audit_history": record.audit_history
                    + (
                        self._event(
                            timestamp=now,
                            event="revoked_by_owner",
                            actor_user_id=requester,
                            reason=reason,
                        ),
                    ),
                }
            )
            return self._replace(updated)

    def delete(self, procedure_id: str, *, requester_user_id: str) -> None:
        with _STORE_LOCK:
            record = self._visible_record(procedure_id, requester_user_id=requester_user_id)
            self._require_owner(record, requester_user_id)
            path = self._path_for_record(record)
            expected_owner = record.owner_id if record.scope is ProcedureScope.USER else None
            records = self._load(
                path,
                expected_scope=record.scope,
                expected_owner=expected_owner,
            )
            remaining = [item for item in records if item.procedure_id != procedure_id]
            if len(remaining) == len(records):
                raise ProcedureStoreError("procedure disappeared before deletion")
            self._write(path, remaining)


__all__ = [
    "ProcedureAuditEvent",
    "ProcedureDefinition",
    "ProcedurePromotionError",
    "ProcedureProvenance",
    "ProcedureRecord",
    "ProcedureRevalidationPolicy",
    "ProcedureScope",
    "ProcedureStatus",
    "ProcedureStoreError",
    "ProceduralMemory",
]
