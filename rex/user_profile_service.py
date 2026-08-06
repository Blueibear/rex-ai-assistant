"""Canonical user profile service composing identity, permissions, and voice data.

Provides a single typed view of user profile information from multiple authorities:
- Identity metadata from Memory/<user_id>/core.json
- Live permissions from rex.permissions
- Voice enrollment status from rex.voice_identity
- Avatar data from data/users/<user_id>/profile/avatar.jpg

All data is JSON-safe and never returns raw filesystem paths.
"""

from __future__ import annotations

import base64
import copy
import io
import json
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from rex.identity import (
    create_user_profile,
    get_user_profile,
    update_user_preferences,
    validate_user_id,
)
from rex.permissions import get_permissions
from rex.runtime_paths import memory_dir as get_memory_dir
from rex.runtime_paths import users_data_dir as get_users_data_dir
from rex.voice_identity.embeddings_store import EmbeddingsStore

logger = logging.getLogger(__name__)

_MAX_AVATAR_SIZE = 2 * 1024 * 1024  # 2 MiB
_MAX_AVATAR_PIXELS = 16_000_000
_MAX_PREFERENCE_SIZE = 32 * 1024  # 32 KiB
_MAX_NESTING_DEPTH = 4

_RESERVED_PROFILE_KEYS = {"name", "role", "user", "created_at", "last_updated", "preferences"}


@dataclass(frozen=True)
class UserProfileView:
    """Frozen top-level profile snapshot, JSON-safe for transport.

    Collection fields are defensive copies and do not alias persisted state.
    All fields are JSON-serializable. Avatar data is base64-encoded if present.
    Scope labels indicate privacy boundaries: user-private fields are keyed
    by user_id, while shared household settings are household-scoped.
    """

    user_id: str
    name: str
    initials: str
    role: str
    permissions: list[str]
    preferences: dict
    voice_enrolled: bool
    voice_model_id: str | None
    voice_sample_count: int
    voice_updated_at: str | None
    avatar_present: bool
    avatar_mime_type: str | None
    avatar_data: str | None  # base64-encoded if present
    scope_labels: dict[str, str]


class UserProfileService:
    """Compose and manage user profile views from multiple authorities."""

    def __init__(self, memory_dir: Path | None = None, users_data_dir: Path | None = None):
        """Initialize with optional test directory overrides.

        Args:
            memory_dir: Override Memory/ directory for profiles (default: canonical).
            users_data_dir: Override data/users/ directory for avatars (default: canonical).
        """
        self._memory_dir = memory_dir or get_memory_dir()
        self._users_data_dir = users_data_dir or get_users_data_dir()
        self._embeddings_store: EmbeddingsStore | None = None

    def _get_embeddings_store(self) -> EmbeddingsStore:
        """Lazy-load embeddings store (deferred for testability)."""
        if self._embeddings_store is None:
            self._embeddings_store = EmbeddingsStore(self._memory_dir)
        return self._embeddings_store

    def get_profile(self, user_id: str) -> UserProfileView:
        """Compose a safe profile snapshot from all authoritative stores."""
        user_id = validate_user_id(user_id)
        profile_data = self._load_profile(user_id)
        name = self._sanitize_name(profile_data.get("name", user_id)) or user_id
        profile_role = self._sanitize_role(profile_data.get("role", ""))
        raw_preferences = profile_data.get("preferences", {})
        preferences = copy.deepcopy(raw_preferences) if isinstance(raw_preferences, dict) else {}

        try:
            permissions = sorted(set(get_permissions(user_id)))
        except Exception as exc:
            logger.warning("Failed to load permissions for %s: %s", user_id, exc)
            permissions = []

        role = self._derive_role(permissions, profile_role)
        voice_enrolled, voice_model_id, voice_sample_count, voice_updated_at = (
            self._load_voice_summary(user_id)
        )
        avatar_present, avatar_mime_type, avatar_data = self._load_avatar(user_id)
        scope_labels = {
            "profile": "user-private",
            "preferences": "user-private",
            "memory": "user-private",
            "private_settings": "user-private",
            "avatar": "user-private",
            "voice_identity": "user-private",
            "household_settings": "shared",
        }
        return UserProfileView(
            user_id=user_id,
            name=name,
            initials=self._derive_initials(name),
            role=role,
            permissions=list(permissions),
            preferences=preferences,
            voice_enrolled=voice_enrolled,
            voice_model_id=voice_model_id,
            voice_sample_count=voice_sample_count,
            voice_updated_at=voice_updated_at,
            avatar_present=avatar_present,
            avatar_mime_type=avatar_mime_type,
            avatar_data=avatar_data,
            scope_labels=dict(scope_labels),
        )

    def update_preferences(self, user_id: str, preferences: object) -> None:
        """Validate and merge user-private preferences through identity APIs."""
        user_id = validate_user_id(user_id)
        if not isinstance(preferences, dict):
            raise ValueError("preferences must be a JSON object")
        self._validate_preferences(preferences)

        profile_path = self._memory_dir / user_id / "core.json"
        profile_data = get_user_profile(user_id, memory_dir=self._memory_dir)
        if profile_data is None:
            if profile_path.exists():
                raise ValueError("existing profile data is invalid")
            create_user_profile(user_id, name=user_id, memory_dir=self._memory_dir)
            profile_data = get_user_profile(user_id, memory_dir=self._memory_dir) or {}

        existing = profile_data.get("preferences", {})
        existing_preferences = copy.deepcopy(existing) if isinstance(existing, dict) else {}
        merged = {**existing_preferences, **copy.deepcopy(preferences)}
        self._validate_preferences(merged, label="Merged preferences")
        if not update_user_preferences(user_id, preferences, memory_dir=self._memory_dir):
            raise RuntimeError("failed to persist user preferences")

    def set_avatar(self, user_id: str, image_data: bytes, mime_type: str) -> None:
        """Validate, normalize, and atomically persist a private avatar."""
        user_id = validate_user_id(user_id)
        if not isinstance(image_data, bytes):
            raise ValueError("image_data must be bytes")
        if mime_type not in {"image/jpeg", "image/png"}:
            raise ValueError("Only image/jpeg and image/png are supported")
        if not image_data:
            raise ValueError("Image data is empty")
        if len(image_data) > _MAX_AVATAR_SIZE:
            raise ValueError("Image is larger than 2 MiB")

        try:
            from PIL import Image, ImageOps
        except ImportError as exc:
            raise ValueError("Pillow is required for avatar processing") from exc

        try:
            with Image.open(io.BytesIO(image_data)) as opened:
                actual_mime = {"JPEG": "image/jpeg", "PNG": "image/png"}.get(opened.format or "")
                if actual_mime != mime_type:
                    raise ValueError(
                        f"Image format {opened.format} does not match declared MIME type {mime_type}"
                    )
                width, height = opened.size
                if width <= 0 or height <= 0 or width * height > _MAX_AVATAR_PIXELS:
                    raise ValueError("Image dimensions are not allowed")
                opened.load()
                normalized = ImageOps.fit(
                    opened.convert("RGB"),
                    (256, 256),
                    method=Image.Resampling.LANCZOS,
                    centering=(0.5, 0.5),
                )
                output = io.BytesIO()
                normalized.save(
                    output,
                    format="JPEG",
                    quality=85,
                    optimize=False,
                    progressive=False,
                )
                avatar_bytes = output.getvalue()
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError("Image data is not a valid image") from exc

        avatar_path = self._avatar_path(user_id)
        self._atomic_write(avatar_path, avatar_bytes)
        logger.info("Set avatar for user %s", user_id)

    def remove_avatar(self, user_id: str) -> None:
        """Remove a private avatar idempotently."""
        user_id = validate_user_id(user_id)
        self._avatar_path(user_id).unlink(missing_ok=True)
        logger.info("Removed avatar for user %s", user_id)

    def _load_profile(self, user_id: str) -> dict:
        """Load a profile object, degrading missing or malformed data safely."""
        profile = get_user_profile(user_id, memory_dir=self._memory_dir)
        return copy.deepcopy(profile) if isinstance(profile, dict) else {}

    def _sanitize_name(self, name: object) -> str:
        """Return a trimmed display name capped for UI and IPC transport."""
        return name.strip()[:128] if isinstance(name, str) and name.strip() else ""

    def _sanitize_role(self, role: object) -> str:
        """Return a trimmed presentation role capped for UI transport."""
        return role.strip()[:128] if isinstance(role, str) and role.strip() else ""

    def _derive_initials(self, name: str) -> str:
        """Derive initials from name (up to 2 uppercase letters).

        Splits on whitespace and takes first letter of each word.
        Falls back to first letter of name if only one word.
        """
        if not name:
            return ""
        words = name.split()
        if not words:
            return ""
        if len(words) == 1:
            return words[0][0].upper()
        # Multiple words - take first letter of first two words
        initials = "".join(word[0].upper() for word in words[:2] if word)
        return initials

    def _derive_role(self, permissions: list[str], profile_role: str) -> str:
        """Derive presentation role from permissions + profile role.

        Priority:
        1. Administrator if "admin" in permissions
        2. profile_role if non-empty
        3. Member
        """
        if "admin" in permissions:
            return "Administrator"
        if profile_role:
            return profile_role
        return "Member"

    def _load_voice_summary(self, user_id: str) -> tuple[bool, str | None, int, str | None]:
        """Load voice enrollment metadata without exposing embedding vectors."""
        try:
            embedding = self._get_embeddings_store().load(user_id)
        except Exception as exc:
            logger.warning("Failed to load voice enrollment for %s: %s", user_id, exc)
            embedding = None
        if embedding is None:
            return False, None, 0, None
        return True, embedding.model_id, embedding.sample_count, embedding.updated_at or None

    def _load_avatar(self, user_id: str) -> tuple[bool, str | None, str | None]:
        """Read at most 2 MiB of normalized JPEG avatar data."""
        avatar_path = self._avatar_path(user_id)
        try:
            with avatar_path.open("rb") as handle:
                avatar_bytes = handle.read(_MAX_AVATAR_SIZE + 1)
        except FileNotFoundError:
            return False, None, None
        except OSError as exc:
            logger.warning("Failed to load avatar for %s: %s", user_id, exc)
            return False, None, None
        if len(avatar_bytes) > _MAX_AVATAR_SIZE or not avatar_bytes.startswith(b"\xff\xd8"):
            logger.warning("Ignoring invalid stored avatar for %s", user_id)
            return False, None, None
        return True, "image/jpeg", base64.b64encode(avatar_bytes).decode("ascii")

    def _avatar_path(self, user_id: str) -> Path:
        """Return the validated private avatar path."""
        return self._users_data_dir / validate_user_id(user_id) / "profile" / "avatar.jpg"

    def _atomic_write(self, path: Path, data: bytes) -> None:
        """Atomically write bytes in the destination directory."""
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=path.parent,
                prefix=".avatar-",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            temporary = None
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def _validate_preferences(self, preferences: dict, *, label: str = "Preferences") -> None:
        """Validate a JSON object, including keys, depth, values, and size."""
        self._check_string_keys(preferences)
        if any(key in _RESERVED_PROFILE_KEYS for key in preferences):
            raise ValueError(f"{label} contain reserved profile keys")
        self._check_nesting_depth(preferences)
        try:
            serialized = json.dumps(
                preferences,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} are not JSON-serializable") from exc
        if len(serialized.encode("utf-8")) > _MAX_PREFERENCE_SIZE:
            raise ValueError(f"{label} serialized size exceeds 32 KiB")

    def _check_string_keys(self, obj: object, depth: int = 0) -> None:
        """Recursively check all dict keys are strings."""
        if isinstance(obj, dict):
            for key, value in obj.items():
                if not isinstance(key, str):
                    raise ValueError(f"Preference keys must be strings, found {type(key).__name__}")
                self._check_string_keys(value, depth + 1)
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                self._check_string_keys(item, depth + 1)

    def _check_nesting_depth(self, obj: object, depth: int = 0) -> None:
        """Recursively check nesting depth <= 4 levels."""
        if depth > _MAX_NESTING_DEPTH:
            raise ValueError(f"Preferences have nesting depth exceeding {_MAX_NESTING_DEPTH}")
        if isinstance(obj, dict):
            for value in obj.values():
                self._check_nesting_depth(value, depth + 1)
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                self._check_nesting_depth(item, depth + 1)


__all__ = [
    "UserProfileView",
    "UserProfileService",
]
