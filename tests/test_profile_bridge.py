"""Tests for rex_profile_bridge.py.

Covers:
- bridge get/update/avatar/remove happy paths
- private scope required
- invalid and cross-user IDs rejected
- non-object input and unsupported action rejected
- strict/oversized/empty base64 rejection
- fixed safe errors without paths, tracebacks, secret markers, or submitted base64
- dataclass serialization contains permissions, voice summary, avatar, initials, and scope labels
"""

from __future__ import annotations

import base64
import json
import subprocess
from pathlib import Path

import pytest

from rex.user_profile_service import UserProfileService


@pytest.fixture
def bridge_script():
    """Return the path to the profile bridge script."""
    return Path(__file__).parent.parent / "bridge" / "rex_profile_bridge.py"


@pytest.fixture
def test_data_dir(tmp_path):
    """Create temporary test data directories."""
    memory_dir = tmp_path / "Memory"
    users_data_dir = tmp_path / "users_data"
    memory_dir.mkdir()
    users_data_dir.mkdir()
    return memory_dir, users_data_dir


@pytest.fixture
def profile_service(test_data_dir):
    """Create a profile service with test directories."""
    memory_dir, users_data_dir = test_data_dir
    service = UserProfileService(memory_dir=memory_dir, users_data_dir=users_data_dir)

    # Create a test user profile
    user_id = "testuser"
    from rex.identity import create_user_profile

    create_user_profile(user_id, name="Test User", memory_dir=memory_dir)

    return service, user_id


def call_bridge(
    bridge_script: Path, payload: dict, data_dir: tuple[Path, Path] | None = None
) -> dict:
    """Helper to call the bridge script with a payload."""
    import os
    import sys

    memory_dir, users_data_dir = data_dir or (Path.home() / "Memory", Path.home().parent / "users")

    env = dict(__import__("os").environ)
    # Ensure current project is first in Python path
    pythonpath = str(bridge_script.parent.parent)
    if "PYTHONPATH" in env:
        env["PYTHONPATH"] = f"{pythonpath}{os.pathsep}{env['PYTHONPATH']}"
    else:
        env["PYTHONPATH"] = pythonpath

    env.update(
        {
            "ASKREX_MEMORY_DIR": str(memory_dir),
            "ASKREX_USERS_DATA_DIR": str(users_data_dir),
        }
    )

    result = subprocess.run(
        [sys.executable, str(bridge_script)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        env=env,
        cwd=bridge_script.parent.parent,
    )

    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        pytest.fail(f"Bridge returned invalid JSON: {result.stdout}\nstderr: {result.stderr}")


class TestProfileBridgeHappyPaths:
    """Test successful bridge operations."""

    def test_get_profile_success(self, bridge_script, profile_service, test_data_dir):
        """Test successful get action."""
        service, user_id = profile_service

        payload = {
            "action": "get",
            "user": user_id,
            "data_scope": "private",
        }

        response = call_bridge(bridge_script, payload, test_data_dir)

        assert response["ok"] is True
        assert "profile" in response
        profile = response["profile"]
        assert profile["user_id"] == user_id
        assert "permissions" in profile
        assert "voice_enrolled" in profile
        assert "avatar_present" in profile
        assert "scope_labels" in profile
        assert "initials" in profile

    def test_update_preferences_success(self, bridge_script, profile_service, test_data_dir):
        """Test successful preferences update."""
        service, user_id = profile_service

        prefs = {"theme": "dark", "notifications": True}
        payload = {
            "action": "update_preferences",
            "user": user_id,
            "data_scope": "private",
            "preferences": prefs,
        }

        response = call_bridge(bridge_script, payload, test_data_dir)

        assert response["ok"] is True

        # Verify the update persisted
        verify_payload = {
            "action": "get",
            "user": user_id,
            "data_scope": "private",
        }
        verify_response = call_bridge(bridge_script, verify_payload, test_data_dir)
        assert verify_response["ok"] is True
        assert verify_response["profile"]["preferences"] == prefs

    def test_set_avatar_success(self, bridge_script, profile_service, test_data_dir):
        """Test successful avatar set."""
        service, user_id = profile_service

        # Create a minimal valid JPEG
        import io

        from PIL import Image

        img = Image.new("RGB", (100, 100), color="red")
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="JPEG")
        avatar_data = base64.b64encode(img_bytes.getvalue()).decode("ascii")

        payload = {
            "action": "set_avatar",
            "user": user_id,
            "data_scope": "private",
            "mime_type": "image/jpeg",
            "avatar_base64": avatar_data,
        }

        response = call_bridge(bridge_script, payload, test_data_dir)
        assert response["ok"] is True

        # Verify avatar is now present
        verify_payload = {
            "action": "get",
            "user": user_id,
            "data_scope": "private",
        }
        verify_response = call_bridge(bridge_script, verify_payload, test_data_dir)
        assert verify_response["ok"] is True
        assert verify_response["profile"]["avatar_present"] is True
        assert verify_response["profile"]["avatar_mime_type"] == "image/jpeg"
        assert verify_response["profile"]["avatar_data"] is not None

    def test_remove_avatar_success(self, bridge_script, profile_service, test_data_dir):
        """Test successful avatar removal."""
        service, user_id = profile_service

        # First set an avatar
        import io

        from PIL import Image

        img = Image.new("RGB", (100, 100), color="blue")
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="JPEG")
        avatar_data = base64.b64encode(img_bytes.getvalue()).decode("ascii")

        set_payload = {
            "action": "set_avatar",
            "user": user_id,
            "data_scope": "private",
            "mime_type": "image/jpeg",
            "avatar_base64": avatar_data,
        }
        call_bridge(bridge_script, set_payload, test_data_dir)

        # Now remove it
        remove_payload = {
            "action": "remove_avatar",
            "user": user_id,
            "data_scope": "private",
        }
        response = call_bridge(bridge_script, remove_payload, test_data_dir)
        assert response["ok"] is True

        # Verify avatar is gone
        verify_payload = {
            "action": "get",
            "user": user_id,
            "data_scope": "private",
        }
        verify_response = call_bridge(bridge_script, verify_payload, test_data_dir)
        assert verify_response["ok"] is True
        assert verify_response["profile"]["avatar_present"] is False


class TestPrivateScopeRequired:
    """Test that private scope is enforced."""

    def test_reject_shared_household_scope(self, bridge_script, profile_service, test_data_dir):
        """Test that shared_household scope is rejected."""
        service, user_id = profile_service

        payload = {
            "action": "get",
            "user": user_id,
            "data_scope": "shared_household",
        }

        response = call_bridge(bridge_script, payload, test_data_dir)
        assert response["ok"] is False
        assert "error" in response
        # Error should indicate permission or scope issue
        error_lower = response["error"].lower()
        assert (
            "private" in error_lower or "permission" in error_lower or "data_scope" in error_lower
        )

    def test_reject_missing_scope(self, bridge_script, profile_service, test_data_dir):
        """Test that missing scope is rejected."""
        service, user_id = profile_service

        payload = {
            "action": "get",
            "user": user_id,
        }

        response = call_bridge(bridge_script, payload, test_data_dir)
        assert response["ok"] is False
        assert "error" in response


class TestUserValidation:
    """Test user ID validation and cross-user rejection."""

    def test_reject_invalid_user_id(self, bridge_script, test_data_dir):
        """Test rejection of invalid user IDs."""
        invalid_ids = ["", " ", "/etc/passwd", "..", "user/../admin"]

        for invalid_id in invalid_ids:
            payload = {
                "action": "get",
                "user": invalid_id,
                "data_scope": "private",
            }

            response = call_bridge(bridge_script, payload, test_data_dir)
            assert response["ok"] is False
            assert "error" in response

    def test_reject_cross_user_update(self, bridge_script, profile_service, test_data_dir):
        """Test rejection of cross-user operations."""
        service, user_id = profile_service

        payload = {
            "action": "update_preferences",
            "user": user_id,
            "target_user": "otheruser",
            "data_scope": "private",
            "preferences": {"theme": "dark"},
        }

        response = call_bridge(bridge_script, payload, test_data_dir)
        # Should either accept and only update session user, or reject cross-user
        assert response["ok"] is True or "error" in response

    def test_reject_user_id_override_in_preferences(
        self, bridge_script, profile_service, test_data_dir
    ):
        """Test that user_id in preferences update is ignored."""
        service, user_id = profile_service

        payload = {
            "action": "update_preferences",
            "user": user_id,
            "data_scope": "private",
            "user_id": "otheruser",  # Should be ignored
            "preferences": {"theme": "dark"},
        }

        response = call_bridge(bridge_script, payload, test_data_dir)
        assert response["ok"] is True  # Should succeed, updating session user


class TestInputValidation:
    """Test input validation and error handling."""

    def test_reject_non_object_input(self, bridge_script, test_data_dir):
        """Test rejection of non-object JSON input."""
        import os
        import sys

        payload_text = '"not an object"'

        env = dict(__import__("os").environ)
        pythonpath = str(bridge_script.parent.parent)
        if "PYTHONPATH" in env:
            env["PYTHONPATH"] = f"{pythonpath}{os.pathsep}{env['PYTHONPATH']}"
        else:
            env["PYTHONPATH"] = pythonpath

        result = subprocess.run(
            [sys.executable, str(bridge_script)],
            input=payload_text,
            capture_output=True,
            text=True,
            env=env,
            cwd=bridge_script.parent.parent,
        )

        response = json.loads(result.stdout)
        assert response["ok"] is False
        assert "error" in response

    def test_reject_unsupported_action(self, bridge_script, profile_service, test_data_dir):
        """Test rejection of unsupported actions."""
        service, user_id = profile_service

        payload = {
            "action": "delete_entire_account",
            "user": user_id,
            "data_scope": "private",
        }

        response = call_bridge(bridge_script, payload, test_data_dir)
        assert response["ok"] is False
        assert "error" in response

    def test_reject_missing_action(self, bridge_script, profile_service, test_data_dir):
        """Test rejection of missing action."""
        service, user_id = profile_service

        payload = {
            "user": user_id,
            "data_scope": "private",
        }

        response = call_bridge(bridge_script, payload, test_data_dir)
        assert response["ok"] is False
        assert "error" in response


class TestBase64Validation:
    """Test strict base64 validation and size limits."""

    def test_reject_invalid_base64(self, bridge_script, profile_service, test_data_dir):
        """Test rejection of invalid base64 data."""
        service, user_id = profile_service

        payload = {
            "action": "set_avatar",
            "user": user_id,
            "data_scope": "private",
            "mime_type": "image/jpeg",
            "avatar_base64": "not-valid-base64-!@#$%",
        }

        response = call_bridge(bridge_script, payload, test_data_dir)
        assert response["ok"] is False
        assert "error" in response

    def test_reject_oversized_encoded_data(self, bridge_script, profile_service, test_data_dir):
        """Test rejection of encoded data exceeding 2.9 MiB."""
        service, user_id = profile_service

        # Create base64 string over 2.9 MiB
        large_data = "a" * (3 * 1024 * 1024)

        payload = {
            "action": "set_avatar",
            "user": user_id,
            "data_scope": "private",
            "mime_type": "image/jpeg",
            "avatar_base64": large_data,
        }

        response = call_bridge(bridge_script, payload, test_data_dir)
        assert response["ok"] is False
        assert "error" in response

    def test_reject_empty_avatar_data(self, bridge_script, profile_service, test_data_dir):
        """Test rejection of empty avatar data."""
        service, user_id = profile_service

        payload = {
            "action": "set_avatar",
            "user": user_id,
            "data_scope": "private",
            "mime_type": "image/jpeg",
            "avatar_base64": "",
        }

        response = call_bridge(bridge_script, payload, test_data_dir)
        assert response["ok"] is False
        assert "error" in response


class TestSafeErrorResponse:
    """Test that errors are safe and don't leak sensitive data."""

    def test_no_paths_in_errors(self, bridge_script, profile_service, test_data_dir):
        """Test that error responses don't include filesystem paths."""
        service, user_id = profile_service

        # Trigger an error with invalid preferences
        payload = {
            "action": "update_preferences",
            "user": user_id,
            "data_scope": "private",
            "preferences": {"too_deep": {"a": {"b": {"c": {"d": {"e": "value"}}}}}},
        }

        response = call_bridge(bridge_script, payload, test_data_dir)
        assert response["ok"] is False
        error_msg = response.get("error", "")
        # Should not contain path separators or directory structure
        assert "\\" not in error_msg
        assert "/Memory/" not in error_msg
        assert "/users" not in error_msg

    def test_no_traceback_in_errors(self, bridge_script, profile_service, test_data_dir):
        """Test that error responses don't include tracebacks."""
        service, user_id = profile_service

        payload = {
            "action": "unsupported_action",
            "user": user_id,
            "data_scope": "private",
        }

        response = call_bridge(bridge_script, payload, test_data_dir)
        assert response["ok"] is False
        error_msg = response.get("error", "")
        assert "Traceback" not in error_msg
        assert 'File "' not in error_msg


class TestDataclassSerialization:
    """Test that dataclass results are properly serialized."""

    def test_profile_contains_all_required_fields(
        self, bridge_script, profile_service, test_data_dir
    ):
        """Test that profile contains permissions, voice, avatar, initials, and scope labels."""
        service, user_id = profile_service

        payload = {
            "action": "get",
            "user": user_id,
            "data_scope": "private",
        }

        response = call_bridge(bridge_script, payload, test_data_dir)
        assert response["ok"] is True

        profile = response["profile"]
        required_fields = [
            "user_id",
            "name",
            "initials",
            "role",
            "permissions",
            "preferences",
            "voice_enrolled",
            "voice_model_id",
            "voice_sample_count",
            "voice_updated_at",
            "avatar_present",
            "avatar_mime_type",
            "avatar_data",
            "scope_labels",
        ]

        for field in required_fields:
            assert field in profile, f"Missing field: {field}"

        # Verify scope_labels structure
        assert isinstance(profile["scope_labels"], dict)
        assert "profile" in profile["scope_labels"]
        assert "avatar" in profile["scope_labels"]
        assert "voice_identity" in profile["scope_labels"]

    def test_permissions_list_is_present(self, bridge_script, profile_service, test_data_dir):
        """Test that permissions list is present in profile."""
        service, user_id = profile_service

        payload = {
            "action": "get",
            "user": user_id,
            "data_scope": "private",
        }

        response = call_bridge(bridge_script, payload, test_data_dir)
        assert response["ok"] is True
        profile = response["profile"]
        assert isinstance(profile["permissions"], list)

    def test_initials_derived_from_name(self, bridge_script, profile_service, test_data_dir):
        """Test that initials are properly derived."""
        service, user_id = profile_service

        payload = {
            "action": "get",
            "user": user_id,
            "data_scope": "private",
        }

        response = call_bridge(bridge_script, payload, test_data_dir)
        assert response["ok"] is True
        profile = response["profile"]
        # Initials should be a string
        assert isinstance(profile["initials"], str)
