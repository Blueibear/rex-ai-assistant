"""Strict request payload validation helpers for mobile routes.

All external payloads are untrusted.  JSON is parsed strictly enough to
distinguish malformed JSON (400) from a wrong content type (415) and from an
empty object.  Client-supplied identity/authorization fields are never
accepted anywhere.
"""

from __future__ import annotations

import re
import uuid
from typing import Any

from rex.mobile_api import errors as merr
from rex.mobile_api.errors import MobileApiError
from rex.mobile_api.sessions import DeviceInfo

_DEVICE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_MAX_DEVICE_TEXT_LENGTH = 128
_MAX_TOKEN_LENGTH = 512
_MAX_CREDENTIAL_LENGTH = 512


def parse_json_body() -> dict[str, Any]:
    """Return the request's JSON object, failing closed on bad input.

    Raises:
        MobileApiError: 415 ``INVALID_MEDIA`` for a non-JSON content type,
            400 ``BAD_REQUEST`` for malformed JSON or a non-object payload.
    """
    from flask import request  # noqa: PLC0415

    if request.mimetype != "application/json":
        raise MobileApiError(
            merr.INVALID_MEDIA,
            "Content-Type must be application/json.",
            415,
        )
    try:
        payload = request.get_json(force=False, silent=False)
    except Exception as exc:
        raise MobileApiError(merr.BAD_REQUEST, "Request body is not valid JSON.", 400) from exc
    if not isinstance(payload, dict):
        raise MobileApiError(merr.BAD_REQUEST, "Request body must be a JSON object.", 400)
    return payload


def require_string_field(
    payload: dict[str, Any],
    name: str,
    *,
    max_length: int = _MAX_CREDENTIAL_LENGTH,
) -> str:
    """Return a required non-empty string field, or raise 400."""
    value = payload.get(name)
    if not isinstance(value, str) or not value.strip():
        raise MobileApiError(merr.BAD_REQUEST, f"Field '{name}' is required.", 400)
    if len(value) > max_length:
        raise MobileApiError(merr.BAD_REQUEST, f"Field '{name}' is too long.", 400)
    return value


def parse_refresh_token_field(payload: dict[str, Any]) -> str:
    """Return the ``refresh_token`` field without echoing its value anywhere."""
    value = payload.get("refresh_token")
    if not isinstance(value, str) or not value.strip() or len(value) > _MAX_TOKEN_LENGTH:
        raise MobileApiError(merr.BAD_REQUEST, "Field 'refresh_token' is required.", 400)
    return value.strip()


def parse_device_info(payload: dict[str, Any]) -> DeviceInfo:
    """Validate optional login device metadata.

    ``device_id`` is an app-generated stable random identifier — validated
    for length/format but never trusted as identity.  A missing device object
    gets a server-generated random device ID.

    Raises:
        MobileApiError: 400 when the device object or device_id is invalid.
    """
    device = payload.get("device")
    if device is None:
        return DeviceInfo(device_id=str(uuid.uuid4()))
    if not isinstance(device, dict):
        raise MobileApiError(merr.BAD_REQUEST, "Field 'device' must be an object.", 400)

    device_id = device.get("device_id")
    if device_id is None:
        device_id = str(uuid.uuid4())
    elif not isinstance(device_id, str) or not _DEVICE_ID_PATTERN.fullmatch(device_id):
        raise MobileApiError(merr.BAD_REQUEST, "Field 'device.device_id' is invalid.", 400)

    def _text(name: str) -> str:
        value = device.get(name, "")
        if value is None:
            return ""
        if not isinstance(value, str):
            raise MobileApiError(merr.BAD_REQUEST, f"Field 'device.{name}' must be a string.", 400)
        return value.strip()[:_MAX_DEVICE_TEXT_LENGTH]

    return DeviceInfo(
        device_id=device_id,
        name=_text("name"),
        platform=_text("platform"),
        app_version=_text("app_version"),
    )


__all__ = [
    "parse_device_info",
    "parse_json_body",
    "parse_refresh_token_field",
    "require_string_field",
]
