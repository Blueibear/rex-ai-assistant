"""Request body schema validation for Flask endpoints.

Provides a ``validate_json_body`` decorator that validates incoming JSON
payloads against a Pydantic model before the view function executes.

On success the validated model instance is stored as ``flask.g.validated_body``
so handlers can access it without re-parsing the request.

On failure a 400 error response is returned with the field name(s) that failed
validation — no business logic runs.

Usage::

    from rex.validation import validate_json_body
    from rex.validation import ChatRequest

    @app.route("/api/chat", methods=["POST"])
    @require_auth
    @validate_json_body(ChatRequest)
    def chat():
        body = g.validated_body  # type: ChatRequest
        ...
"""

from __future__ import annotations

import logging
from functools import wraps
from typing import Any, Callable, TypeVar

import pydantic
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

_M = TypeVar("_M", bound=BaseModel)


# ---------------------------------------------------------------------------
# Request body schemas
# ---------------------------------------------------------------------------


class LoginRequest(BaseModel):
    """Schema for POST /api/dashboard/login."""

    password: str = Field(default="", max_length=1024)


class ChatRequest(BaseModel):
    """Schema for POST /api/chat."""

    message: str = Field(max_length=32_000)

    @field_validator("message")
    @classmethod
    def message_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("message must not be blank")
        return v


class CreateJobRequest(BaseModel):
    """Schema for POST /api/scheduler/jobs."""

    name: str = Field(max_length=256)
    schedule: str = Field(default="interval:3600", max_length=256)
    enabled: bool = Field(default=True, strict=True)
    callback_name: str | None = None
    workflow_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def name_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("name must not be blank")
        return v


# ---------------------------------------------------------------------------
# Decorator
# ---------------------------------------------------------------------------


def validate_json_body(
    schema_cls: type[_M],
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator that validates the JSON request body against *schema_cls*.

    Stores the validated model instance in ``flask.g.validated_body`` before
    calling the wrapped view function.  Returns a 400 error response if:

    - The ``Content-Type`` is not JSON or the body is not parseable JSON.
    - One or more fields fail Pydantic validation.

    The 400 message lists the failing field name(s) so clients can surface
    specific errors to users.

    Args:
        schema_cls: A Pydantic ``BaseModel`` subclass describing the expected
            request body shape.
    """

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            from flask import g, request  # noqa: PLC0415

            from rex.http_errors import BAD_REQUEST, error_response  # noqa: PLC0415

            raw = request.get_json(silent=True)
            if raw is None:
                return error_response(BAD_REQUEST, "Request body must be valid JSON", 400)

            try:
                g.validated_body = schema_cls.model_validate(raw)
            except pydantic.ValidationError as exc:
                fields = [".".join(str(loc) for loc in err["loc"]) for err in exc.errors()]
                field_list = ", ".join(fields) if fields else "unknown"
                return error_response(
                    BAD_REQUEST,
                    f"Validation failed on field(s): {field_list}",
                    400,
                )

            return fn(*args, **kwargs)

        return wrapper

    return decorator


__all__ = [
    "ChatRequest",
    "CreateJobRequest",
    "LoginRequest",
    "validate_json_body",
]
