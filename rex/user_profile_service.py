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
import json
import logging
from dataclasses import dataclass
from pathlib import Path

from rex.identity import validate_user_id
from rex.permissions import get_permissions
from rex.runtime_paths import memory_dir as get_memory_dir, users_data_dir as get_users_data_dir
from rex.voice_identity.embeddings_store import EmbeddingsStore

logger = logging.getLogger(__name__)

_MAX_AVATAR_SIZE = 2 * 1024 * 1024  # 2 MiB
_MAX_PREFERENCE_SIZE = 32 * 1024  # 32 KiB
_MAX_NESTING_DEPTH = 4

_RESERVED_PROFILE_KEYS = {"name", "role", "user", "created_at", "last_updated", "preferences"}


@dataclass(frozen=True)
class UserProfileView:
    """Immutable view of user profile data, JSON-safe for transport.

    All fields are JSON-serializable. Avatar data is base64-encoded if present.
    Scope labels indicate privacy boundaries: user-private fields are keyed
    by user_id, while shared household settings are household-scoped.
    """

    user_id: str
    name: str
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
        """Load and compose user profile from all authorities.

        If core.json is missing, returns a safe view using the validated
        user_id as name, with empty preferences and no permissions.

        Args:
            user_id: Validated user identifier.

        Returns:
            Immutable UserProfileView.

        Raises:
            ValueError: If user_id is filesystem-unsafe.
        """
        user_id = validate_user_id(user_id)

        # Load profile from Memory
        profile_data = self._load_profile(user_id)
        name = profile_data.get("name", user_id)
        profile_role = profile_data.get("role", "")
        preferences = profile_data.get("preferences", {})
        if not isinstance(preferences, dict):
            preferences = {}

        # Get live permissions (sorted)
        permissions = sorted(get_permissions(user_id))

        # Derive presentation role from permissions + profile role
        role = self._derive_role(permissions, profile_role)

        # Load voice enrollment summary
        voice_enrolled, voice_model_id, voice_sample_count, voice_updated_at = (
            self._load_voice_summary(user_id)
        )

        # Load avatar metadata
        avatar_present, avatar_mime_type, avatar_data = self._load_avatar(user_id)

        scope_labels = {
            "preferences": "user-private",
            "memory": "user-private",
            "household_settings": "shared",
        }

        return UserProfileView(
            user_id=user_id,
            name=name,
            role=role,
            permissions=permissions,
            preferences=preferences,
            voice_enrolled=voice_enrolled,
            voice_model_id=voice_model_id,
            voice_sample_count=voice_sample_count,
            voice_updated_at=voice_updated_at,
            avatar_present=avatar_present,
            avatar_mime_type=avatar_mime_type,
            avatar_data=avatar_data,
            scope_labels=scope_labels,
        )

    def update_preferences(self, user_id: str, preferences: dict) -> None:
        """Safely merge preferences into user profile.

        Creates a minimal profile if missing. Validates:
        - Reserved keys rejected
        - Nesting depth <= 4 levels
        - All values JSON-serializable
        - Serialized size <= 32 KiB

        Args:
            user_id: Validated user identifier.
            preferences: Dict of preference key/value pairs to merge.

        Raises:
            ValueError: If user_id is unsafe, keys are reserved, nesting is too deep,
                       values are not JSON-serializable, or serialized size exceeds 32 KiB.
        """
        user_id = validate_user_id(user_id)
        self._validate_preferences(preferences)

        # Load or create profile
        profile_data = self._load_profile(user_id)
        if not profile_data:
            profile_data = {"name": user_id}

        # Merge preferences
        existing_prefs = profile_data.get("preferences", {})
        if not isinstance(existing_prefs, dict):
            existing_prefs = {}
        existing_prefs.update(preferences)

        # Update timestamps
        from datetime import UTC, datetime

        now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        profile_data["preferences"] = existing_prefs
        profile_data["last_updated"] = now

        # Write back to disk
        profile_dir = self._memory_dir / user_id
        profile_dir.mkdir(parents=True, exist_ok=True)
        profile_path = profile_dir / "core.json"
        profile_path.write_text(json.dumps(profile_data, indent=2), encoding="utf-8")
        logger.info("Updated preferences for user %s", user_id)

    def set_avatar(self, user_id: str, image_data: bytes, mime_type: str) -> None:
        """Store a user avatar with validation.

        Accepts only image/jpeg or image/png. Validates:
        - MIME type is supported
        - Size <= 2 MiB
        - Data is valid image of declared format
        - Format and MIME match

        Converts to RGB, center-crops and resizes to 256x256, writes as JPEG quality 85.

        Args:
            user_id: Validated user identifier.
            image_data: Raw image bytes.
            mime_type: Declared MIME type (image/jpeg or image/png).

        Raises:
            ValueError: If MIME type unsupported, size exceeds 2 MiB,
                       data is invalid, or MIME/content mismatch.
        """
        user_id = validate_user_id(user_id)

        # Check MIME type
        if mime_type not in ("image/jpeg", "image/png"):
            raise ValueError("Only image/jpeg and image/png are supported")

        # Check size
        if len(image_data) > _MAX_AVATAR_SIZE:
            raise ValueError(f"Image is larger than 2 MiB")

        # Validate and process image with Pillow
        try:
            from PIL import Image
        except ImportError:
            raise ValueError("Pillow is required for avatar processing")

        try:
            img = Image.open(__import__("io").BytesIO(image_data))
        except Exception:
            raise ValueError("Image data is not a valid image")

        # Validate MIME/format match
        format_to_mime = {"JPEG": "image/jpeg", "PNG": "image/png"}
        actual_mime = format_to_mime.get(img.format, "")
        if actual_mime != mime_type:
            raise ValueError(f"Image format {img.format} does not match actual format {mime_type}")

        # Convert to RGB, center-crop and resize to 256x256
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Center-crop to square
        width, height = img.size
        crop_size = min(width, height)
        left = (width - crop_size) // 2
        top = (height - crop_size) // 2
        right = left + crop_size
        bottom = top + crop_size
        img = img.crop((left, top, right, bottom))

        # Resize to 256x256
        img = img.resize((256, 256), Image.Resampling.LANCZOS)

        # Write to avatar.jpg using internal users_data_dir
        avatar_dir = self._users_data_dir / user_id / "profile"
        avatar_dir.mkdir(parents=True, exist_ok=True)
        avatar_path = avatar_dir / "avatar.jpg"

        avatar_buffer = __import__("io").BytesIO()
        img.save(avatar_buffer, format="JPEG", quality=85)
        avatar_path.write_bytes(avatar_buffer.getvalue())
        logger.info("Set avatar for user %s at %s", user_id, avatar_path)

    def remove_avatar(self, user_id: str) -> None:
        """Remove user avatar (idempotent).

        Args:
            user_id: Validated user identifier.
        """
        user_id = validate_user_id(user_id)
        avatar_path = self._users_data_dir / user_id / "profile" / "avatar.jpg"
        if avatar_path.exists():
            avatar_path.unlink()
            logger.info("Removed avatar for user %s", user_id)

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _load_profile(self, user_id: str) -> dict:
        """Load profile JSON from Memory/<user_id>/core.json.

        Returns empty dict if file missing or corrupt, never raises.
        """
        profile_path = self._memory_dir / user_id / "core.json"
        if not profile_path.exists():
            return {}
        try:
            return json.loads(profile_path.read_text(encoding="utf-8"))  # type: ignore[no-any-return]
        except Exception as exc:
            logger.warning("Failed to load profile for %s: %s", user_id, exc)
            return {}

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
        """Load voice enrollment summary.

        Returns (enrolled, model_id, sample_count, updated_at).
        """
        try:
            embedding = self._get_embeddings_store().load(user_id)
        except Exception:
            embedding = None
        if embedding is None:
            return False, None, 0, None
        return True, embedding.model_id, embedding.sample_count, embedding.updated_at or None

    def _load_avatar(self, user_id: str) -> tuple[bool, str | None, str | None]:
        """Load avatar metadata and base64 content.

        Returns (present, mime_type, base64_data).
        """
        avatar_path = self._users_data_dir / user_id / "profile" / "avatar.jpg"
        if not avatar_path.exists():
            return False, None, None
        try:
            avatar_bytes = avatar_path.read_bytes()
            avatar_b64 = base64.b64encode(avatar_bytes).decode("ascii")
            return True, "image/jpeg", avatar_b64
        except Exception as exc:
            logger.warning("Failed to load avatar for %s: %s", user_id, exc)
            return False, None, None

    def _validate_preferences(self, preferences: dict) -> None:
        """Validate preferences for reserved keys, nesting depth, and JSON-ability.

        Raises ValueError if validation fails.
        """
        # Check for reserved keys
        if any(key in _RESERVED_PROFILE_KEYS for key in preferences.keys()):
            raise ValueError("Preferences contain reserved profile keys")

        # Check nesting depth
        self._check_nesting_depth(preferences)

        # Check JSON-serializable and size
        try:
            serialized = json.dumps(preferences)
            if len(serialized.encode("utf-8")) > _MAX_PREFERENCE_SIZE:
                raise ValueError(f"Preferences serialized size exceeds 32 KiB")
        except TypeError:
            raise ValueError("Preference values are not JSON-serializable") from None
        except ValueError:
            raise

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
