"""Tests for the user profile service.

Tests are offline and deterministic. User profile composition,
avatar handling, and preference updates are isolated via fixtures.
"""

from __future__ import annotations

import base64
import json
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from rex.user_profile_service import UserProfileView, UserProfileService


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
        assert view.role == "Administrator"
        assert view.permissions == ["admin", "email_send"]
        assert view.preferences == {"theme": "dark"}

    def test_profile_view_is_json_serializable(self):
        """UserProfileView must be JSON-safe."""
        view = UserProfileView(
            user_id="alice",
            name="Alice",
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
        (profile_dir / "core.json").write_text(
            json.dumps({"name": "Alice"}), encoding="utf-8"
        )

        view = service.get_profile("alice")
        assert view.role == "Member"

    def test_profile_permissions_sorted(
        self, service: UserProfileService, memory_dir: Path, mock_permissions
    ):
        """Permissions are returned sorted."""
        mock_permissions.return_value = ["email_send", "admin", "computer_control"]
        profile_dir = memory_dir / "alice"
        profile_dir.mkdir(parents=True)
        (profile_dir / "core.json").write_text(
            json.dumps({"name": "Alice"}), encoding="utf-8"
        )

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

    def test_voice_enrolled_with_metadata(
        self, service: UserProfileService, mock_embeddings_store
    ):
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
        from PIL import Image
        import io

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
        from PIL import Image
        import io

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
        from PIL import Image
        import io

        img = Image.new("RGB", (100, 100), color="red")
        jpeg_bytes = io.BytesIO()
        img.save(jpeg_bytes, format="JPEG", quality=85)
        jpeg_data = jpeg_bytes.getvalue()

        # Claim PNG but provide JPEG
        with pytest.raises(ValueError, match="does not match actual format"):
            service.set_avatar("alice", jpeg_data, "image/png")

    def test_get_avatar_after_set(self, service: UserProfileService):
        """Read avatar after setting."""
        pytest.importorskip("PIL")
        from PIL import Image
        import io

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
        from PIL import Image
        import io

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
        (profile_dir / "core.json").write_text(
            json.dumps({"name": "Alice"}), encoding="utf-8"
        )

        reserved_keys = ["name", "role", "user", "created_at", "last_updated"]
        for key in reserved_keys:
            with pytest.raises(ValueError, match="reserved"):
                service.update_preferences("alice", {key: "value"})

    def test_update_preferences_nesting_depth_limit(
        self, service: UserProfileService, memory_dir: Path
    ):
        """Reject preferences with nesting deeper than 4 levels."""
        profile_dir = memory_dir / "alice"
        profile_dir.mkdir(parents=True)
        (profile_dir / "core.json").write_text(
            json.dumps({"name": "Alice"}), encoding="utf-8"
        )

        # 5 levels deep should fail
        deep_prefs = {"a": {"b": {"c": {"d": {"e": "value"}}}}}
        with pytest.raises(ValueError, match="nesting depth"):
            service.update_preferences("alice", deep_prefs)

    def test_update_preferences_non_json_value(
        self, service: UserProfileService, memory_dir: Path
    ):
        """Reject non-JSON values in preferences."""
        profile_dir = memory_dir / "alice"
        profile_dir.mkdir(parents=True)
        (profile_dir / "core.json").write_text(
            json.dumps({"name": "Alice"}), encoding="utf-8"
        )

        import datetime

        with pytest.raises(ValueError, match="not JSON-serializable"):
            service.update_preferences("alice", {"date": datetime.datetime.now()})  # type: ignore

    def test_update_preferences_size_cap(
        self, service: UserProfileService, memory_dir: Path
    ):
        """Reject preferences serialized size > 32 KiB."""
        profile_dir = memory_dir / "alice"
        profile_dir.mkdir(parents=True)
        (profile_dir / "core.json").write_text(
            json.dumps({"name": "Alice"}), encoding="utf-8"
        )

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

    def test_avatar_isolated_by_user(
        self, service: UserProfileService, users_data_dir: Path
    ):
        """Alice's avatar doesn't affect Bob's."""
        pytest.importorskip("PIL")
        from PIL import Image
        import io

        img = Image.new("RGB", (100, 100), color="red")
        jpeg_bytes = io.BytesIO()
        img.save(jpeg_bytes, format="JPEG", quality=85)
        jpeg_data = jpeg_bytes.getvalue()

        service.set_avatar("alice", jpeg_data, "image/jpeg")

        # Bob should have no avatar
        bob_view = service.get_profile("bob")
        assert bob_view.avatar_present is False

    def test_preferences_isolated_by_user(
        self, service: UserProfileService, memory_dir: Path
    ):
        """Alice's preferences don't affect Bob's."""
        alice_dir = memory_dir / "alice"
        alice_dir.mkdir(parents=True)
        (alice_dir / "core.json").write_text(
            json.dumps({"name": "Alice"}), encoding="utf-8"
        )

        service.update_preferences("alice", {"theme": "dark"})

        bob_view = service.get_profile("bob")
        assert bob_view.preferences == {}


# ---------------------------------------------------------------
# Corrupt/malformed profile tests
# ---------------------------------------------------------------


class TestMalformedProfile:
    """Tests for handling corrupt or malformed profiles."""

    def test_corrupt_json_returns_safe_view(
        self, service: UserProfileService, memory_dir: Path
    ):
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
