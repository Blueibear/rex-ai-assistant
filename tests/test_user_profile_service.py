"""Tests for the user profile service.

Tests are offline and deterministic. User profile composition,
avatar handling, and preference updates are isolated via fixtures.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from rex.user_profile_service import UserProfileService, UserProfileView

# ---------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------


@pytest.fixture()
def memory_dir(tmp_path: Path) -> Path:
    """Create a temporary Memory directory for testing."""
    mem = tmp_path / "Memory"
    mem.mkdir(parents=True)
    return mem


@pytest.fixture()
def users_data_dir(tmp_path: Path) -> Path:
    """Create a temporary users data directory for testing."""
    data = tmp_path / "data" / "users"
    data.mkdir(parents=True)
    return data


@pytest.fixture()
def service(memory_dir: Path, users_data_dir: Path) -> UserProfileService:
    """Create a UserProfileService with test directories."""
    return UserProfileService(memory_dir=memory_dir, users_data_dir=users_data_dir)


@pytest.fixture()
def mock_permissions():
    """Mock permissions to return controlled results."""
    with patch("rex.user_profile_service.get_permissions") as mock_perm:
        yield mock_perm


@pytest.fixture()
def mock_embeddings_store():
    """Mock EmbeddingsStore for voice enrollment."""
    with patch("rex.user_profile_service.EmbeddingsStore") as mock_store:
        yield mock_store


# ---------------------------------------------------------------
# User profile view tests
# ---------------------------------------------------------------


class TestUserProfileView:
    """Tests for the UserProfileView data structure."""

    def test_profile_view_has_required_fields(self):
        """UserProfileView must have all required fields."""
        view = UserProfileView(
            user_id="alice",
            name="Alice",
            initials="A",
            role="Administrator",
            permissions=["admin", "email_send"],
            preferences={"theme": "dark"},
            voice_enrolled=True,
            voice_model_id="synthetic",
            voice_sample_count=5,
            voice_updated_at="2026-08-06T12:00:00Z",
            avatar_present=False,
            avatar_mime_type=None,
            avatar_data=None,
            scope_labels={
                "preferences": "user-private",
                "memory": "user-private",
                "household_settings": "shared",
            },
        )
        assert view.user_id == "alice"
        assert view.name == "Alice"
        assert view.initials == "A"
        assert view.role == "Administrator"
        assert view.permissions == ["admin", "email_send"]
        assert view.preferences == {"theme": "dark"}

    def test_profile_view_is_json_serializable(self):
        """UserProfileView must be JSON-safe."""
        view = UserProfileView(
            user_id="alice",
            name="Alice",
            initials="A",
            role="Member",
            permissions=["email_send"],
            preferences={"theme": "light"},
            voice_enrolled=False,
            voice_model_id=None,
            voice_sample_count=0,
            voice_updated_at=None,
            avatar_present=False,
            avatar_mime_type=None,
            avatar_data=None,
            scope_labels={
                "preferences": "user-private",
                "memory": "user-private",
                "household_settings": "shared",
            },
        )
        # Should not raise when converting to dict
        data = {
            "user_id": view.user_id,
            "name": view.name,
            "initials": view.initials,
            "role": view.role,
            "permissions": view.permissions,
        }
        assert json.dumps(data)


# ---------------------------------------------------------------
# Profile composition tests
# ---------------------------------------------------------------


class TestProfileComposition:
    """Tests for composing user profiles from identity sources."""

    def test_get_profile_missing_returns_safe_view(self, service: UserProfileService):
        """Missing core.json returns safe view using validated ID as name."""
        view = service.get_profile("newuser")
        assert view.user_id == "newuser"
        assert view.name == "newuser"
        assert view.role == "Member"
        assert view.permissions == []

    def test_get_profile_existing(
        self, service: UserProfileService, memory_dir: Path, mock_permissions
    ):
        """Get existing profile from core.json."""
        mock_permissions.return_value = ["email_send"]
        profile_dir = memory_dir / "alice"
        profile_dir.mkdir(parents=True)
        (profile_dir / "core.json").write_text(
            json.dumps(
                {
                    "name": "Alice Smith",
                    "role": "Owner and primary user",
                    "preferences": {"theme": "dark"},
                }
            ),
            encoding="utf-8",
        )

        view = service.get_profile("alice")
        assert view.user_id == "alice"
        assert view.name == "Alice Smith"
        assert view.role == "Owner and primary user"
        assert view.preferences == {"theme": "dark"}
        assert view.permissions == ["email_send"]

    def test_get_profile_invalid_user_id_raises(self, service: UserProfileService):
        """Invalid user ID raises ValueError."""
        with pytest.raises(ValueError, match="Invalid user_id"):
            service.get_profile("../invalid")

    def test_profile_role_admin_from_permissions(
        self, service: UserProfileService, memory_dir: Path, mock_permissions
    ):
        """Presentation role is Administrator when admin permission present."""
        mock_permissions.return_value = ["admin", "email_send"]
        profile_dir = memory_dir / "alice"
        profile_dir.mkdir(parents=True)
        (profile_dir / "core.json").write_text(
            json.dumps({"name": "Alice", "role": "User"}), encoding="utf-8"
        )

        view = service.get_profile("alice")
        # Role from permissions takes precedence
        assert view.role == "Administrator"

    def test_profile_role_member_fallback(
        self, service: UserProfileService, memory_dir: Path, mock_permissions
    ):
        """Presentation role is Member when no admin and no profile role."""
        mock_permissions.return_value = ["email_send"]
        profile_dir = memory_dir / "alice"
        profile_dir.mkdir(parents=True)
        (profile_dir / "core.json").write_text(json.dumps({"name": "Alice"}), encoding="utf-8")

        view = service.get_profile("alice")
        assert view.role == "Member"

    def test_profile_permissions_sorted(
        self, service: UserProfileService, memory_dir: Path, mock_permissions
    ):
        """Permissions are returned sorted."""
        mock_permissions.return_value = ["email_send", "admin", "computer_control"]
        profile_dir = memory_dir / "alice"
        profile_dir.mkdir(parents=True)
        (profile_dir / "core.json").write_text(json.dumps({"name": "Alice"}), encoding="utf-8")

        view = service.get_profile("alice")
        assert view.permissions == ["admin", "computer_control", "email_send"]


# ---------------------------------------------------------------
# Voice enrollment tests
# ---------------------------------------------------------------


class TestVoiceEnrollment:
    """Tests for voice enrollment summary in profile."""

    def test_voice_not_enrolled(self, service: UserProfileService, mock_embeddings_store):
        """Voice enrollment false when no embeddings."""
        mock_store_instance = mock_embeddings_store.return_value
        mock_store_instance.load.return_value = None

        view = service.get_profile("alice")
        assert view.voice_enrolled is False
        assert view.voice_model_id is None
        assert view.voice_sample_count == 0
        assert view.voice_updated_at is None

    def test_voice_enrolled_with_metadata(self, service: UserProfileService, mock_embeddings_store):
        """Voice enrollment summary includes model and sample count."""
        from rex.voice_identity.types import VoiceEmbedding

        mock_store_instance = mock_embeddings_store.return_value
        embedding = VoiceEmbedding(
            vector=[0.1, 0.2, 0.3],
            model_id="welm-v1",
            sample_count=10,
            updated_at="2026-08-06T10:00:00Z",
        )
        mock_store_instance.load.return_value = embedding

        view = service.get_profile("alice")
        assert view.voice_enrolled is True
        assert view.voice_model_id == "welm-v1"
        assert view.voice_sample_count == 10
        assert view.voice_updated_at == "2026-08-06T10:00:00Z"


# ---------------------------------------------------------------
# Avatar tests
# ---------------------------------------------------------------


class TestAvatarHandling:
    """Tests for avatar set/read/remove."""

    def test_avatar_not_present(self, service: UserProfileService):
        """Avatar absent when no file exists."""
        view = service.get_profile("alice")
        assert view.avatar_present is False
        assert view.avatar_mime_type is None
        assert view.avatar_data is None

    def test_set_avatar_jpeg(self, service: UserProfileService, users_data_dir: Path):
        """Set avatar with JPEG image."""
        pytest.importorskip("PIL")
        import io

        from PIL import Image

        # Create a small JPEG
        img = Image.new("RGB", (100, 100), color="red")
        jpeg_bytes = io.BytesIO()
        img.save(jpeg_bytes, format="JPEG", quality=85)
        jpeg_data = jpeg_bytes.getvalue()

        service.set_avatar("alice", jpeg_data, "image/jpeg")

        avatar_path = users_data_dir / "alice" / "profile" / "avatar.jpg"
        assert avatar_path.exists()

    def test_set_avatar_png(self, service: UserProfileService, users_data_dir: Path):
        """Set avatar with PNG image."""
        pytest.importorskip("PIL")
        import io

        from PIL import Image

        img = Image.new("RGB", (100, 100), color="blue")
        png_bytes = io.BytesIO()
        img.save(png_bytes, format="PNG")
        png_data = png_bytes.getvalue()

        service.set_avatar("alice", png_data, "image/png")

        avatar_path = users_data_dir / "alice" / "profile" / "avatar.jpg"
        assert avatar_path.exists()

    def test_set_avatar_invalid_mime_type(self, service: UserProfileService):
        """Reject non-image MIME types."""
        with pytest.raises(ValueError, match="Only image/jpeg and image/png"):
            service.set_avatar("alice", b"fake", "text/plain")

    def test_set_avatar_oversize(self, service: UserProfileService):
        """Reject images larger than 2 MiB."""
        large_data = b"x" * (2 * 1024 * 1024 + 1)
        with pytest.raises(ValueError, match="larger than 2 MiB"):
            service.set_avatar("alice", large_data, "image/jpeg")

    def test_set_avatar_invalid_image(self, service: UserProfileService):
        """Reject invalid image data."""
        with pytest.raises(ValueError, match="not a valid image"):
            service.set_avatar("alice", b"not an image", "image/jpeg")

    def test_set_avatar_mime_mismatch(self, service: UserProfileService):
        """Reject MIME/content mismatch."""
        pytest.importorskip("PIL")
        import io

        from PIL import Image

        img = Image.new("RGB", (100, 100), color="red")
        jpeg_bytes = io.BytesIO()
        img.save(jpeg_bytes, format="JPEG", quality=85)
        jpeg_data = jpeg_bytes.getvalue()

        # Claim PNG but provide JPEG
        with pytest.raises(ValueError, match="does not match"):
            service.set_avatar("alice", jpeg_data, "image/png")

    def test_get_avatar_after_set(self, service: UserProfileService):
        """Read avatar after setting."""
        pytest.importorskip("PIL")
        import io

        from PIL import Image

        img = Image.new("RGB", (100, 100), color="green")
        jpeg_bytes = io.BytesIO()
        img.save(jpeg_bytes, format="JPEG", quality=85)
        jpeg_data = jpeg_bytes.getvalue()

        service.set_avatar("alice", jpeg_data, "image/jpeg")

        view = service.get_profile("alice")
        assert view.avatar_present is True
        assert view.avatar_mime_type == "image/jpeg"
        assert view.avatar_data is not None
        # Decode base64 and check it's valid
        decoded = base64.b64decode(view.avatar_data)
        assert len(decoded) > 0

    def test_remove_avatar_idempotent(self, service: UserProfileService):
        """Remove avatar is idempotent."""
        pytest.importorskip("PIL")
        import io

        from PIL import Image

        img = Image.new("RGB", (100, 100), color="yellow")
        jpeg_bytes = io.BytesIO()
        img.save(jpeg_bytes, format="JPEG", quality=85)
        jpeg_data = jpeg_bytes.getvalue()

        service.set_avatar("alice", jpeg_data, "image/jpeg")
        service.remove_avatar("alice")

        # Second removal should not raise
        service.remove_avatar("alice")

        view = service.get_profile("alice")
        assert view.avatar_present is False


# ---------------------------------------------------------------
# Preference update tests
# ---------------------------------------------------------------


class TestPreferenceUpdate:
    """Tests for safe preference merge and update."""

    def test_update_preferences_reserved_key_rejected(
        self, service: UserProfileService, memory_dir: Path
    ):
        """Reject reserved profile keys in preferences update."""
        profile_dir = memory_dir / "alice"
        profile_dir.mkdir(parents=True)
        (profile_dir / "core.json").write_text(json.dumps({"name": "Alice"}), encoding="utf-8")

        reserved_keys = ["name", "role", "user", "created_at", "last_updated", "preferences"]
        for key in reserved_keys:
            with pytest.raises(ValueError, match="reserved"):
                service.update_preferences("alice", {key: "value"})

    def test_update_preferences_nesting_depth_limit(
        self, service: UserProfileService, memory_dir: Path
    ):
        """Reject preferences with nesting deeper than 4 levels."""
        profile_dir = memory_dir / "alice"
        profile_dir.mkdir(parents=True)
        (profile_dir / "core.json").write_text(json.dumps({"name": "Alice"}), encoding="utf-8")

        # 5 levels deep should fail
        deep_prefs = {"a": {"b": {"c": {"d": {"e": "value"}}}}}
        with pytest.raises(ValueError, match="nesting depth"):
            service.update_preferences("alice", deep_prefs)

    def test_update_preferences_non_json_value(self, service: UserProfileService, memory_dir: Path):
        """Reject non-JSON values in preferences."""
        profile_dir = memory_dir / "alice"
        profile_dir.mkdir(parents=True)
        (profile_dir / "core.json").write_text(json.dumps({"name": "Alice"}), encoding="utf-8")

        import datetime

        with pytest.raises(ValueError, match="not JSON-serializable"):
            service.update_preferences("alice", {"date": datetime.datetime.now()})  # type: ignore

    def test_update_preferences_size_cap(self, service: UserProfileService, memory_dir: Path):
        """Reject preferences serialized size > 32 KiB."""
        profile_dir = memory_dir / "alice"
        profile_dir.mkdir(parents=True)
        (profile_dir / "core.json").write_text(json.dumps({"name": "Alice"}), encoding="utf-8")

        # Create a preferences dict that serializes to > 32 KiB
        huge_prefs = {"data": "x" * (33 * 1024)}
        with pytest.raises(ValueError, match="32 KiB"):
            service.update_preferences("alice", huge_prefs)

    def test_update_preferences_creates_minimal_profile_on_update(
        self, service: UserProfileService, memory_dir: Path, mock_permissions
    ):
        """Update on missing profile creates minimal profile."""
        mock_permissions.return_value = []
        assert not (memory_dir / "alice" / "core.json").exists()

        service.update_preferences("alice", {"theme": "dark"})

        profile_path = memory_dir / "alice" / "core.json"
        assert profile_path.exists()
        data = json.loads(profile_path.read_text(encoding="utf-8"))
        assert data["name"] == "alice"
        assert data["preferences"] == {"theme": "dark"}

    def test_update_preferences_merges_existing(
        self, service: UserProfileService, memory_dir: Path
    ):
        """Update merges with existing preferences."""
        profile_dir = memory_dir / "alice"
        profile_dir.mkdir(parents=True)
        (profile_dir / "core.json").write_text(
            json.dumps({"name": "Alice", "preferences": {"theme": "dark"}}),
            encoding="utf-8",
        )

        service.update_preferences("alice", {"language": "en"})

        data = json.loads((profile_dir / "core.json").read_text(encoding="utf-8"))
        assert data["preferences"] == {"theme": "dark", "language": "en"}


# ---------------------------------------------------------------
# User isolation tests
# ---------------------------------------------------------------


class TestUserIsolation:
    """Tests for user data isolation."""

    def test_avatar_isolated_by_user(self, service: UserProfileService, users_data_dir: Path):
        """Alice's avatar doesn't affect Bob's."""
        pytest.importorskip("PIL")
        import io

        from PIL import Image

        img = Image.new("RGB", (100, 100), color="red")
        jpeg_bytes = io.BytesIO()
        img.save(jpeg_bytes, format="JPEG", quality=85)
        jpeg_data = jpeg_bytes.getvalue()

        service.set_avatar("alice", jpeg_data, "image/jpeg")

        # Bob should have no avatar
        bob_view = service.get_profile("bob")
        assert bob_view.avatar_present is False

    def test_preferences_isolated_by_user(self, service: UserProfileService, memory_dir: Path):
        """Alice's preferences don't affect Bob's."""
        alice_dir = memory_dir / "alice"
        alice_dir.mkdir(parents=True)
        (alice_dir / "core.json").write_text(json.dumps({"name": "Alice"}), encoding="utf-8")

        service.update_preferences("alice", {"theme": "dark"})

        bob_view = service.get_profile("bob")
        assert bob_view.preferences == {}


# ---------------------------------------------------------------
# Corrupt/malformed profile tests
# ---------------------------------------------------------------


class TestMalformedProfile:
    """Tests for handling corrupt or malformed profiles."""

    def test_corrupt_json_returns_safe_view(self, service: UserProfileService, memory_dir: Path):
        """Corrupt core.json returns safe view without crashing."""
        profile_dir = memory_dir / "alice"
        profile_dir.mkdir(parents=True)
        (profile_dir / "core.json").write_text("{ not valid json", encoding="utf-8")

        # Should not raise
        view = service.get_profile("alice")
        assert view.user_id == "alice"
        assert view.name == "alice"

    def test_missing_name_field(self, service: UserProfileService, memory_dir: Path):
        """Missing name field uses user_id as fallback."""
        profile_dir = memory_dir / "alice"
        profile_dir.mkdir(parents=True)
        (profile_dir / "core.json").write_text(json.dumps({"role": "Admin"}), encoding="utf-8")

        view = service.get_profile("alice")
        assert view.name == "alice"

    def test_load_profile_handles_json_array(
        self, service: UserProfileService, memory_dir: Path, mock_permissions
    ):
        """Valid JSON array in core.json degrades to safe missing-profile view."""
        mock_permissions.return_value = []
        profile_dir = memory_dir / "alice"
        profile_dir.mkdir(parents=True)
        # Write a valid JSON array instead of object
        (profile_dir / "core.json").write_text("[1, 2, 3]", encoding="utf-8")

        # Should not crash, should return safe view
        view = service.get_profile("alice")
        assert view.user_id == "alice"
        assert view.name == "alice"
        assert view.preferences == {}

    def test_load_profile_handles_json_scalar(
        self, service: UserProfileService, memory_dir: Path, mock_permissions
    ):
        """Valid JSON scalar in core.json degrades to safe missing-profile view."""
        mock_permissions.return_value = []
        profile_dir = memory_dir / "alice"
        profile_dir.mkdir(parents=True)
        # Write a valid JSON scalar (string)
        (profile_dir / "core.json").write_text('"just a string"', encoding="utf-8")

        # Should not crash, should return safe view
        view = service.get_profile("alice")
        assert view.user_id == "alice"
        assert view.name == "alice"

    def test_load_profile_sanitizes_name(
        self, service: UserProfileService, memory_dir: Path, mock_permissions
    ):
        """Non-string name is sanitized to safe string."""
        mock_permissions.return_value = []
        profile_dir = memory_dir / "alice"
        profile_dir.mkdir(parents=True)
        # Write profile with non-string name
        (profile_dir / "core.json").write_text(
            json.dumps({"name": 12345, "role": "Admin"}), encoding="utf-8"
        )

        view = service.get_profile("alice")
        assert isinstance(view.name, str)
        assert view.name == "alice"

    def test_load_profile_sanitizes_role(
        self, service: UserProfileService, memory_dir: Path, mock_permissions
    ):
        """Non-string role is sanitized to safe string."""
        mock_permissions.return_value = []
        profile_dir = memory_dir / "alice"
        profile_dir.mkdir(parents=True)
        # Write profile with non-string role
        (profile_dir / "core.json").write_text(
            json.dumps({"name": "Alice", "role": 12345}), encoding="utf-8"
        )

        view = service.get_profile("alice")
        assert isinstance(view.role, str)
        # Should fall back through derivation
        assert view.role == "Member"


# ---------------------------------------------------------------
# Initials and scope label tests
# ---------------------------------------------------------------


class TestInitialsAndScopes:
    """Tests for initials field and explicit scope labels."""

    def test_profile_view_has_initials_field(self):
        """UserProfileView must have initials field."""
        view = UserProfileView(
            user_id="alice",
            name="Alice Smith",
            initials="AS",
            role="Member",
            permissions=[],
            preferences={},
            voice_enrolled=False,
            voice_model_id=None,
            voice_sample_count=0,
            voice_updated_at=None,
            avatar_present=False,
            avatar_mime_type=None,
            avatar_data=None,
            scope_labels={
                "preferences": "user-private",
                "memory": "user-private",
                "private_settings": "user-private",
                "avatar": "user-private",
                "voice_identity": "user-private",
                "profile": "user-private",
                "household_settings": "shared",
            },
        )
        assert view.initials == "AS"

    def test_initials_derived_from_display_name(
        self, service: UserProfileService, memory_dir: Path, mock_permissions
    ):
        """Initials derived safely from sanitized display name."""
        mock_permissions.return_value = []
        profile_dir = memory_dir / "alice"
        profile_dir.mkdir(parents=True)
        (profile_dir / "core.json").write_text(
            json.dumps({"name": "Alice Smith"}), encoding="utf-8"
        )

        view = service.get_profile("alice")
        assert view.initials == "AS"

    def test_initials_single_name(
        self, service: UserProfileService, memory_dir: Path, mock_permissions
    ):
        """Initials from single name uses one letter."""
        mock_permissions.return_value = []
        profile_dir = memory_dir / "alice"
        profile_dir.mkdir(parents=True)
        (profile_dir / "core.json").write_text(json.dumps({"name": "Alice"}), encoding="utf-8")

        view = service.get_profile("alice")
        assert view.initials == "A"

    def test_initials_fallback_to_user_id(self, service: UserProfileService, mock_permissions):
        """Initials fall back to user_id when name is missing."""
        mock_permissions.return_value = []
        # Missing profile - name will be user_id
        view = service.get_profile("alice")
        assert view.initials == "A"

    def test_initials_caps_at_two(
        self, service: UserProfileService, memory_dir: Path, mock_permissions
    ):
        """Initials capped at two letters maximum."""
        mock_permissions.return_value = []
        profile_dir = memory_dir / "alice"
        profile_dir.mkdir(parents=True)
        (profile_dir / "core.json").write_text(
            json.dumps({"name": "Alice Bob Charlie"}), encoding="utf-8"
        )

        view = service.get_profile("alice")
        assert view.initials == "AB"
        assert len(view.initials) <= 2

    def test_scope_labels_complete(self, service: UserProfileService, mock_permissions):
        """All scope labels present in profile view."""
        mock_permissions.return_value = []
        view = service.get_profile("alice")

        required_scopes = {
            "profile",
            "preferences",
            "memory",
            "private_settings",
            "avatar",
            "voice_identity",
            "household_settings",
        }
        assert set(view.scope_labels.keys()) >= required_scopes

        # Check correct labels
        user_private_scopes = {
            "profile",
            "preferences",
            "memory",
            "private_settings",
            "avatar",
            "voice_identity",
        }
        for scope in user_private_scopes:
            assert view.scope_labels[scope] == "user-private"

        assert view.scope_labels["household_settings"] == "shared"


# ---------------------------------------------------------------
# Avatar atomic write and size bound tests
# ---------------------------------------------------------------


class TestAvatarAtomicWriteAndBounds:
    """Tests for avatar atomic writes and read size bounds."""

    def test_avatar_write_is_atomic(self, service: UserProfileService, users_data_dir: Path):
        """Avatar writes use atomic tempfile + replace."""
        pytest.importorskip("PIL")
        import io

        from PIL import Image

        img = Image.new("RGB", (100, 100), color="red")
        jpeg_bytes = io.BytesIO()
        img.save(jpeg_bytes, format="JPEG", quality=85)
        jpeg_data = jpeg_bytes.getvalue()

        service.set_avatar("alice", jpeg_data, "image/jpeg")

        avatar_path = users_data_dir / "alice" / "profile" / "avatar.jpg"
        assert avatar_path.exists()
        # File should exist as final avatar.jpg, not temp file
        assert not any(f.name.startswith("tmp") for f in avatar_path.parent.iterdir())

    def test_avatar_read_bounded_to_2mib(self, service: UserProfileService, users_data_dir: Path):
        """Avatar reads refuse files larger than 2 MiB."""
        pytest.importorskip("PIL")

        # Create a directory structure for the user
        avatar_dir = users_data_dir / "alice" / "profile"
        avatar_dir.mkdir(parents=True, exist_ok=True)
        avatar_path = avatar_dir / "avatar.jpg"

        # Write a file larger than 2 MiB
        large_data = b"x" * (2 * 1024 * 1024 + 1)
        avatar_path.write_bytes(large_data)

        # When loading profile, should ignore oversized avatar
        view = service.get_profile("alice")
        assert view.avatar_present is False


class TestImageProcessingErrors:
    """Tests for robust image processing with proper error handling."""

    def test_image_processing_handles_truncation(self, service: UserProfileService):
        """Image processing handles truncated/corrupted images as ValueError."""
        pytest.importorskip("PIL")
        # Truncated JPEG (starts valid but is incomplete)
        truncated_jpeg = bytes.fromhex("ffd8ffe000104a46494600")  # Minimal truncated JPEG

        with pytest.raises(ValueError):
            service.set_avatar("alice", truncated_jpeg, "image/jpeg")

    def test_image_mime_mismatch_message(self, service: UserProfileService):
        """MIME mismatch error message is correct."""
        pytest.importorskip("PIL")
        import io

        from PIL import Image

        img = Image.new("RGB", (100, 100), color="red")
        jpeg_bytes = io.BytesIO()
        img.save(jpeg_bytes, format="JPEG", quality=85)
        jpeg_data = jpeg_bytes.getvalue()

        with pytest.raises(ValueError) as exc_info:
            service.set_avatar("alice", jpeg_data, "image/png")

        error_msg = str(exc_info.value)
        assert "does not match" in error_msg or "mismatch" in error_msg.lower()


# ---------------------------------------------------------------
# Preference validation enhancement tests
# ---------------------------------------------------------------


class TestPreferenceValidationEnhancements:
    """Tests for enhanced preference validation."""

    def test_update_preferences_rejects_non_dict(
        self, service: UserProfileService, memory_dir: Path
    ):
        """update_preferences must reject non-dict input at runtime."""
        profile_dir = memory_dir / "alice"
        profile_dir.mkdir(parents=True)
        (profile_dir / "core.json").write_text(json.dumps({"name": "Alice"}), encoding="utf-8")

        with pytest.raises((ValueError, TypeError)):
            service.update_preferences("alice", "not a dict")  # type: ignore

    def test_update_preferences_rejects_non_string_keys(
        self, service: UserProfileService, memory_dir: Path
    ):
        """Reject preferences with non-string keys at any depth."""
        profile_dir = memory_dir / "alice"
        profile_dir.mkdir(parents=True)
        (profile_dir / "core.json").write_text(json.dumps({"name": "Alice"}), encoding="utf-8")

        # Direct non-string key
        with pytest.raises(ValueError, match="non-string|key"):
            service.update_preferences("alice", {1: "value"})  # type: ignore

        # Nested non-string key
        with pytest.raises(ValueError, match="non-string|key"):
            service.update_preferences("alice", {"nested": {2: "value"}})  # type: ignore

    def test_update_preferences_rejects_nan_infinity(
        self, service: UserProfileService, memory_dir: Path
    ):
        """Reject NaN/Infinity values using allow_nan=False."""
        profile_dir = memory_dir / "alice"
        profile_dir.mkdir(parents=True)
        (profile_dir / "core.json").write_text(json.dumps({"name": "Alice"}), encoding="utf-8")

        # NaN
        with pytest.raises(ValueError, match="NaN|Infinity|JSON"):
            service.update_preferences("alice", {"value": float("nan")})  # type: ignore

        # Infinity
        with pytest.raises(ValueError, match="NaN|Infinity|JSON"):
            service.update_preferences("alice", {"value": float("inf")})  # type: ignore

    def test_update_preferences_merges_without_exceeding_size(
        self, service: UserProfileService, memory_dir: Path
    ):
        """Ensure merged preferences size is also checked."""
        profile_dir = memory_dir / "alice"
        profile_dir.mkdir(parents=True)
        # Create profile with some existing preferences
        (profile_dir / "core.json").write_text(
            json.dumps({"name": "Alice", "preferences": {"existing": "x" * 10000}}),
            encoding="utf-8",
        )

        # Try to add more that would exceed 32 KiB when merged
        huge_update = {"new": "y" * 25000}
        with pytest.raises(ValueError, match="32 KiB|size"):
            service.update_preferences("alice", huge_update)


# ---------------------------------------------------------------
# Deep immutability tests
# ---------------------------------------------------------------


class TestDeepImmutability:
    """Tests for deep immutability of returned views."""

    def test_profile_view_preferences_is_defensive_copy(
        self, service: UserProfileService, memory_dir: Path, mock_permissions
    ):
        """Mutations of returned preferences dict should not affect service state."""
        mock_permissions.return_value = []
        profile_dir = memory_dir / "alice"
        profile_dir.mkdir(parents=True)
        (profile_dir / "core.json").write_text(
            json.dumps({"name": "Alice", "preferences": {"theme": "dark"}}),
            encoding="utf-8",
        )

        view1 = service.get_profile("alice")
        # Attempt to mutate the returned preferences
        view1.preferences["theme"] = "light"  # type: ignore
        view1.preferences["new_key"] = "value"  # type: ignore

        # Get profile again - should be unchanged
        view2 = service.get_profile("alice")
        assert view2.preferences["theme"] == "dark"
        assert "new_key" not in view2.preferences

    def test_profile_view_permissions_is_defensive(
        self, service: UserProfileService, mock_permissions
    ):
        """Mutations of returned permissions list should not affect service state."""
        mock_permissions.return_value = ["admin", "email_send"]

        view1 = service.get_profile("alice")
        # Attempt to mutate the returned permissions
        view1.permissions.append("new_permission")  # type: ignore
        view1.permissions[0] = "modified"  # type: ignore

        # Get profile again - should be unchanged
        view2 = service.get_profile("alice")
        assert "new_permission" not in view2.permissions
        assert view2.permissions[0] == "admin"


class TestSupervisorReviewRegressions:
    """Regression tests for security and persistence findings from review."""

    def test_avatar_failed_replace_preserves_existing_and_cleans_temp(
        self, service: UserProfileService, users_data_dir: Path, monkeypatch
    ):
        pytest.importorskip("PIL")
        import io
        import os

        from PIL import Image

        def image_bytes(color: str) -> bytes:
            image = Image.new("RGB", (100, 100), color=color)
            output = io.BytesIO()
            image.save(output, format="JPEG")
            return output.getvalue()

        service.set_avatar("alice", image_bytes("red"), "image/jpeg")
        avatar_path = users_data_dir / "alice" / "profile" / "avatar.jpg"
        original = avatar_path.read_bytes()

        def fail_replace(source, destination):
            raise OSError("simulated replace failure")

        monkeypatch.setattr(os, "replace", fail_replace)
        with pytest.raises(OSError, match="replace failure"):
            service.set_avatar("alice", image_bytes("blue"), "image/jpeg")

        assert avatar_path.read_bytes() == original
        assert not list(avatar_path.parent.glob(".avatar-*.tmp"))

    def test_small_invalid_stored_avatar_is_ignored(
        self, service: UserProfileService, users_data_dir: Path
    ):
        avatar_dir = users_data_dir / "alice" / "profile"
        avatar_dir.mkdir(parents=True)
        (avatar_dir / "avatar.jpg").write_bytes(b"not-a-jpeg")

        view = service.get_profile("alice")
        assert view.avatar_present is False
        assert view.avatar_data is None

    def test_update_rejects_existing_corrupt_profile(
        self, service: UserProfileService, memory_dir: Path
    ):
        profile_dir = memory_dir / "alice"
        profile_dir.mkdir(parents=True)
        profile_path = profile_dir / "core.json"
        profile_path.write_text("[1, 2, 3]", encoding="utf-8")

        with pytest.raises(ValueError, match="existing profile data is invalid"):
            service.update_preferences("alice", {"theme": "dark"})

        assert profile_path.read_text(encoding="utf-8") == "[1, 2, 3]"

    def test_permission_store_failure_fails_closed(
        self, service: UserProfileService, mock_permissions
    ):
        mock_permissions.side_effect = OSError("database unavailable")

        view = service.get_profile("alice")
        assert view.permissions == []
        assert view.role == "Member"

    def test_whitespace_name_uses_validated_user_id(
        self, service: UserProfileService, memory_dir: Path, mock_permissions
    ):
        mock_permissions.return_value = []
        profile_dir = memory_dir / "alice"
        profile_dir.mkdir(parents=True)
        (profile_dir / "core.json").write_text(
            json.dumps({"name": "   ", "role": "   "}), encoding="utf-8"
        )

        view = service.get_profile("alice")
        assert view.name == "alice"
        assert view.initials == "A"
        assert view.role == "Member"

    def test_compressed_oversized_dimensions_are_rejected(self, service: UserProfileService):
        pytest.importorskip("PIL")
        import io

        from PIL import Image

        image = Image.new("1", (5000, 4000), color=1)
        output = io.BytesIO()
        image.save(output, format="PNG")
        assert len(output.getvalue()) < 2 * 1024 * 1024

        with pytest.raises(ValueError, match="dimensions"):
            service.set_avatar("alice", output.getvalue(), "image/png")
