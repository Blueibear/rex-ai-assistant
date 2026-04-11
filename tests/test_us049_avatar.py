"""Tests for US-049: Profile picture support."""

from __future__ import annotations

import io
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("REX_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("REX_JWT_SECRET", "avatar-test-secret")
    return tmp_path


@pytest.fixture()
def flask_client(tmp_data_dir: Path):  # type: ignore[override]
    from rex.gui_app import _create_flask_app

    app = _create_flask_app(ui_enabled=False)
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def _register_and_login(client: object, username: str, password: str = "pass123") -> str:
    client.post(  # type: ignore[attr-defined]
        "/api/auth/register",
        json={"username": username, "password": password},
    )
    resp = client.post(  # type: ignore[attr-defined]
        "/api/auth/login",
        json={"username": username, "password": password},
    )
    return resp.get_json()["token"]


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _tiny_jpeg() -> bytes:
    """Generate a minimal valid 1x1 JPEG in memory."""
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (1, 1), color=(100, 149, 237)).save(buf, format="JPEG")
    return buf.getvalue()


def _tiny_png() -> bytes:
    """Generate a minimal valid 1x1 PNG in memory."""
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (1, 1), color=(100, 149, 237)).save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# POST /api/user/avatar
# ---------------------------------------------------------------------------


def test_upload_avatar_requires_auth(flask_client) -> None:  # type: ignore[override]
    resp = flask_client.post(
        "/api/user/avatar",
        data={"file": (io.BytesIO(_tiny_jpeg()), "avatar.jpg", "image/jpeg")},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 401


def test_upload_jpeg_avatar_succeeds(flask_client, tmp_data_dir: Path) -> None:  # type: ignore[override]
    token = _register_and_login(flask_client, "alice")
    resp = flask_client.post(
        "/api/user/avatar",
        headers=_auth(token),
        data={"file": (io.BytesIO(_tiny_jpeg()), "avatar.jpg", "image/jpeg")},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 200
    assert resp.get_json()["ok"] is True

    # Verify file is stored in data/avatars/
    avatars = list((tmp_data_dir / "avatars").glob("*.jpg"))
    assert len(avatars) == 1


def test_upload_png_avatar_succeeds(flask_client, tmp_data_dir: Path) -> None:  # type: ignore[override]
    token = _register_and_login(flask_client, "bob")
    resp = flask_client.post(
        "/api/user/avatar",
        headers=_auth(token),
        data={"file": (io.BytesIO(_tiny_png()), "avatar.png", "image/png")},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 200


def test_upload_avatar_wrong_content_type_rejected(flask_client) -> None:  # type: ignore[override]
    token = _register_and_login(flask_client, "carol")
    resp = flask_client.post(
        "/api/user/avatar",
        headers=_auth(token),
        data={"file": (io.BytesIO(b"GIF89a"), "avatar.gif", "image/gif")},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 415


def test_upload_avatar_too_large_rejected(flask_client) -> None:  # type: ignore[override]
    token = _register_and_login(flask_client, "dave")
    oversized = b"\xff\xd8" + b"x" * (2 * 1024 * 1024 + 1)
    resp = flask_client.post(
        "/api/user/avatar",
        headers=_auth(token),
        data={"file": (io.BytesIO(oversized), "big.jpg", "image/jpeg")},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 413


def test_upload_avatar_no_file_field_rejected(flask_client) -> None:  # type: ignore[override]
    token = _register_and_login(flask_client, "eve")
    resp = flask_client.post(
        "/api/user/avatar",
        headers=_auth(token),
        data={},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 400


def test_uploaded_avatar_is_256x256(flask_client, tmp_data_dir: Path) -> None:  # type: ignore[override]
    """Uploaded image must be resized to 256x256."""
    from PIL import Image

    token = _register_and_login(flask_client, "frank")
    flask_client.post(
        "/api/user/avatar",
        headers=_auth(token),
        data={"file": (io.BytesIO(_tiny_jpeg()), "avatar.jpg", "image/jpeg")},
        content_type="multipart/form-data",
    )
    stored = list((tmp_data_dir / "avatars").glob("*.jpg"))[0]
    img = Image.open(stored)
    assert img.size == (256, 256)


# ---------------------------------------------------------------------------
# GET /api/user/avatar
# ---------------------------------------------------------------------------


def test_get_avatar_default_when_no_upload(flask_client) -> None:  # type: ignore[override]
    """Returns a default SVG (200) when no avatar has been uploaded."""
    token = _register_and_login(flask_client, "grace")
    resp = flask_client.get("/api/user/avatar", headers=_auth(token))
    assert resp.status_code == 200
    assert "svg" in resp.content_type


def test_get_avatar_returns_default_without_auth(flask_client) -> None:  # type: ignore[override]
    """Unauthenticated GET returns the default avatar (not 401)."""
    resp = flask_client.get("/api/user/avatar")
    assert resp.status_code == 200
    assert "svg" in resp.content_type


def test_get_avatar_returns_uploaded_jpeg(flask_client) -> None:  # type: ignore[override]
    token = _register_and_login(flask_client, "heidi")
    flask_client.post(
        "/api/user/avatar",
        headers=_auth(token),
        data={"file": (io.BytesIO(_tiny_jpeg()), "avatar.jpg", "image/jpeg")},
        content_type="multipart/form-data",
    )
    resp = flask_client.get("/api/user/avatar", headers=_auth(token))
    assert resp.status_code == 200
    assert resp.content_type == "image/jpeg"


def test_avatar_stored_under_user_id(flask_client, tmp_data_dir: Path) -> None:  # type: ignore[override]
    """Avatar file is keyed by user ID, not username."""
    from rex.auth import get_current_user

    token = _register_and_login(flask_client, "ivan")
    flask_client.post(
        "/api/user/avatar",
        headers=_auth(token),
        data={"file": (io.BytesIO(_tiny_jpeg()), "avatar.jpg", "image/jpeg")},
        content_type="multipart/form-data",
    )
    user = get_current_user(token)
    expected_path = tmp_data_dir / "avatars" / f"{user['id']}.jpg"
    assert expected_path.is_file()


def test_users_avatars_are_isolated(flask_client, tmp_data_dir: Path) -> None:  # type: ignore[override]
    """User A's avatar does not bleed into User B's response."""
    token_a = _register_and_login(flask_client, "jane")
    token_b = _register_and_login(flask_client, "kate")

    # Upload avatar for jane only.
    flask_client.post(
        "/api/user/avatar",
        headers=_auth(token_a),
        data={"file": (io.BytesIO(_tiny_jpeg()), "avatar.jpg", "image/jpeg")},
        content_type="multipart/form-data",
    )

    # Kate has no avatar — should get default SVG.
    resp_b = flask_client.get("/api/user/avatar", headers=_auth(token_b))
    assert "svg" in resp_b.content_type

    # Jane gets her JPEG.
    resp_a = flask_client.get("/api/user/avatar", headers=_auth(token_a))
    assert resp_a.content_type == "image/jpeg"
