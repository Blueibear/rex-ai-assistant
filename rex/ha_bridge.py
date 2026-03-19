"""Home Assistant intent bridge for Rex."""

from __future__ import annotations

import importlib
import importlib.util
import logging
import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

try:
    import requests as _imported_requests
except ImportError as exc:
    requests: Any | None = None
    _REQUESTS_IMPORT_ERROR: Exception | None = exc
else:
    requests = _imported_requests
    _REQUESTS_IMPORT_ERROR = None
if TYPE_CHECKING:
    from flask import Blueprint

from rex.config import settings

logger = logging.getLogger(__name__)

_flask_blueprint = None
_flask_jsonify = None
_flask_request = None


def _require_requests() -> None:
    if requests is None:
        raise RuntimeError(
            "The Home Assistant bridge requires the 'requests' package. "
            "Install with: pip install requests"
        ) from _REQUESTS_IMPORT_ERROR


def _require_flask():
    global _flask_blueprint, _flask_jsonify, _flask_request
    if _flask_blueprint is not None:
        return _flask_blueprint, _flask_jsonify, _flask_request
    if importlib.util.find_spec("flask") is None:
        raise RuntimeError(
            "Flask is required for Home Assistant bridge. Install with: pip install flask"
        )
    flask = importlib.import_module("flask")
    _flask_blueprint = flask.Blueprint
    _flask_jsonify = flask.jsonify
    _flask_request = flask.request
    return _flask_blueprint, _flask_jsonify, _flask_request


def _utc_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


@dataclass
class IntentMatch:
    """Represents an intent matched from natural language."""

    domain: str
    service: str
    entity_id: str
    data: dict[str, Any]
    description: str
    source: str


class HABridge:
    """Translate Rex intents into Home Assistant service calls."""

    _TAG_PATTERN = re.compile(
        r"\[\[ha:(?P<domain>[a-z_]+)\.(?P<service>[a-z_]+)"
        r"(?P<params>(?:\s+[a-z_]+=[^;\]]+)*(?:\s*;\s*[a-z_]+=[^;\]]+)*)\]\]",
        re.IGNORECASE,
    )
    _TURN_ON_PATTERN = re.compile(
        r"\bturn\s+on\s+(?:the\s+)?(?P<entity>[a-z0-9\s]+)", re.IGNORECASE
    )
    _TURN_OFF_PATTERN = re.compile(
        r"\bturn\s+off\s+(?:the\s+)?(?P<entity>[a-z0-9\s]+)", re.IGNORECASE
    )
    _SET_TEMP_PATTERN = re.compile(
        r"\bset\s+(?:the\s+)?(?P<entity>[a-z0-9\s]+?)\s+to\s+(?P<value>\d{2,3})(?:\s*degrees)?",
        re.IGNORECASE,
    )
    _SET_PERCENT_PATTERN = re.compile(
        r"\b(set|dim)\s+(?:the\s+)?(?P<entity>[a-z0-9\s]+)\s+to\s+(?P<value>\d{1,3})\s*%",
        re.IGNORECASE,
    )
    _ACTIVATE_PATTERN = re.compile(
        r"\b(activate|start|run)\s+(?:the\s+)?(?P<entity>[a-z0-9\s]+)", re.IGNORECASE
    )
    _LOCK_PATTERN = re.compile(
        r"\b(lock|unlock)\s+(?:the\s+)?(?P<entity>[a-z0-9\s]+)", re.IGNORECASE
    )

    SUPPORTED_INTENTS = [
        {"intent": "turn_on", "description": "Turn on a light, switch, or scene"},
        {"intent": "turn_off", "description": "Turn off a light, switch, or scene"},
        {"intent": "set_temperature", "description": "Set thermostat temperature"},
        {"intent": "set_percentage", "description": "Set brightness or fan speed"},
        {"intent": "activate", "description": "Activate a scene or script"},
        {"intent": "lock_control", "description": "Lock or unlock a lock"},
    ]

    def __init__(
        self,
        *,
        base_url: str | None = None,
        token: str | None = None,
        secret: str | None = None,
        verify_ssl: bool | None = None,
        timeout: float | None = None,
        entity_map: dict[str, str] | None = None,
    ) -> None:
        cfg = settings
        self._base_url = (base_url or cfg.ha_base_url or "").rstrip("/")
        self._token = token or cfg.ha_token or ""
        self._secret = secret or cfg.ha_secret or ""
        self._verify_ssl = cfg.ha_verify_ssl if verify_ssl is None else verify_ssl
        self._timeout = cfg.ha_timeout if timeout is None else timeout
        self._entity_map = {
            alias.lower(): entity_id
            for alias, entity_id in (entity_map or cfg.ha_entity_map or {}).items()
            if isinstance(alias, str) and isinstance(entity_id, str)
        }
        _require_requests()
        requests_module = requests
        assert requests_module is not None
        self._session = requests_module.Session()
        self._entity_cache: dict[str, str] = {}
        self._entity_cache_ts: float = 0.0
        self._entity_cache_ttl: float = 60.0
        self._lock = threading.Lock()
        self._log_path = Path("logs/test_ha_integration.log")

    @staticmethod
    def _request_exception() -> type[Exception]:
        requests_module = requests
        if requests_module is None:
            return Exception
        return cast(type[Exception], requests_module.RequestException)

    # ------------------------------------------------------------------ #
    # Public properties
    # ------------------------------------------------------------------ #

    @property
    def enabled(self) -> bool:
        return bool(self._base_url and self._token)

    @property
    def secret(self) -> str:
        return self._secret

    # ------------------------------------------------------------------ #
    # Transcript and response handling
    # ------------------------------------------------------------------ #

    def process_transcript(self, transcript: str) -> str | None:
        """Detect and execute intents from a user transcript."""
        if not self.enabled:
            return None
        match = self._match_transcript(transcript)
        if not match:
            return None
        success, message = self._execute_intent(match)
        self._log_event(match, success, message)
        if success:
            return message
        return f"I attempted to perform that action but hit an error: {message}"

    def post_process_response(self, response: str) -> str:
        """Execute inline HA commands embedded in an LLM response."""
        if not self.enabled or "[[ha:" not in response.lower():
            return response

        messages: list[str] = []
        sanitized = response

        for tag in list(self._TAG_PATTERN.finditer(response)):
            domain = tag.group("domain").lower()
            service = tag.group("service").lower()
            params = self._parse_params(tag.group("params") or "")
            entity_id = params.pop("entity_id", None)
            if not entity_id:
                continue
            match = IntentMatch(
                domain=domain,
                service=service,
                entity_id=entity_id,
                data={"entity_id": entity_id, **params},
                description=f"{domain}.{service} {entity_id}",
                source="response",
            )
            success, message = self._execute_intent(match)
            self._log_event(match, success, message)
            sanitized = sanitized.replace(tag.group(0), "").strip()
            if success:
                messages.append(message)
            else:
                messages.append(f"Home Assistant error: {message}")

        if messages:
            " ".join(messages)
            sanitized = f"{sanitized.rstrip()} {' '.join(messages)}".strip()
        return sanitized

    # ------------------------------------------------------------------ #
    # HTTP endpoints helpers
    # ------------------------------------------------------------------ #

    def list_entities(self) -> list[dict[str, Any]]:
        """Return cached Home Assistant entities (refreshing as needed)."""
        if not self.enabled:
            raise RuntimeError("Home Assistant bridge is not configured.")
        self._refresh_entity_cache(force=True)
        result = []
        for alias, entity_id in sorted(self._entity_cache.items()):
            result.append({"friendly_name": alias, "entity_id": entity_id})
        return result

    def control_light(
        self,
        entity_id: str,
        action: str,
        *,
        brightness_pct: int | None = None,
    ) -> dict[str, Any]:
        """Turn on/off a light entity with optional brightness.

        Args:
            entity_id: The HA entity id (e.g. ``light.living_room``).
            action: ``"turn_on"`` or ``"turn_off"``.
            brightness_pct: Optional 0-100 brightness percentage (only used with turn_on).

        Returns:
            The raw HA API response dict.
        """
        if not entity_id:
            raise ValueError("entity_id is required")
        action = action.lower()
        if action not in ("turn_on", "turn_off"):
            raise ValueError(f"action must be 'turn_on' or 'turn_off', got {action!r}")
        data: dict[str, Any] = {"entity_id": entity_id}
        if action == "turn_on" and brightness_pct is not None:
            data["brightness_pct"] = max(0, min(100, int(brightness_pct)))
        intent = IntentMatch(
            domain="light",
            service=action,
            entity_id=entity_id,
            data=data,
            description=f"{action.replace('_', ' ')} {entity_id}",
            source="api",
        )
        success, message = self._execute_intent(intent)
        self._log_event(intent, success, message)
        return {"success": success, "message": message, "entity_id": entity_id}

    def control_switch(
        self,
        entity_id: str,
        action: str,
    ) -> dict[str, Any]:
        """Turn on/off a switch entity.

        Args:
            entity_id: The HA entity id (e.g. ``switch.garage``).
            action: ``"turn_on"`` or ``"turn_off"``.

        Returns:
            The raw HA API response dict.
        """
        if not entity_id:
            raise ValueError("entity_id is required")
        action = action.lower()
        if action not in ("turn_on", "turn_off"):
            raise ValueError(f"action must be 'turn_on' or 'turn_off', got {action!r}")
        intent = IntentMatch(
            domain="switch",
            service=action,
            entity_id=entity_id,
            data={"entity_id": entity_id},
            description=f"{action.replace('_', ' ')} {entity_id}",
            source="api",
        )
        success, message = self._execute_intent(intent)
        self._log_event(intent, success, message)
        return {"success": success, "message": message, "entity_id": entity_id}

    def call_script(
        self, script_id: str, variables: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        if not script_id:
            raise ValueError("script identifier is required")
        return self._request(  # type: ignore[no-any-return]
            "POST",
            "/api/services/script/turn_on",
            json={"entity_id": script_id, "variables": variables or {}},
        )

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _match_transcript(self, transcript: str) -> IntentMatch | None:
        text = transcript.strip().lower()
        if not text:
            return None

        turn_on = self._TURN_ON_PATTERN.search(text)
        if turn_on:
            entity = turn_on.group("entity").strip()
            entity_id = self._resolve_entity(entity)
            if entity_id:
                return IntentMatch(
                    domain=entity_id.split(".", 1)[0],
                    service="turn_on",
                    entity_id=entity_id,
                    data={"entity_id": entity_id},
                    description=f"turn on {entity}",
                    source="transcript",
                )

        turn_off = self._TURN_OFF_PATTERN.search(text)
        if turn_off:
            entity = turn_off.group("entity").strip()
            entity_id = self._resolve_entity(entity)
            if entity_id:
                return IntentMatch(
                    domain=entity_id.split(".", 1)[0],
                    service="turn_off",
                    entity_id=entity_id,
                    data={"entity_id": entity_id},
                    description=f"turn off {entity}",
                    source="transcript",
                )

        temp = self._SET_TEMP_PATTERN.search(text)
        if temp:
            entity = temp.group("entity").strip()
            entity_id = self._resolve_entity(entity)
            value = temp.group("value")
            if entity_id and value:
                return IntentMatch(
                    domain="climate",
                    service="set_temperature",
                    entity_id=entity_id,
                    data={"entity_id": entity_id, "temperature": float(value)},
                    description=f"set {entity} to {value}",
                    source="transcript",
                )

        percent = self._SET_PERCENT_PATTERN.search(text)
        if percent:
            entity = percent.group("entity").strip()
            entity_id = self._resolve_entity(entity)
            value = percent.group("value")
            if entity_id and value:
                level = max(0, min(100, int(value)))
                service = "set_percentage" if entity_id.startswith("fan.") else "turn_on"
                data = {"entity_id": entity_id}
                if entity_id.startswith("light.") or entity_id.startswith("switch."):
                    data["brightness_pct"] = level  # type: ignore[assignment]
                else:
                    data["percentage"] = level  # type: ignore[assignment]
                return IntentMatch(
                    domain=entity_id.split(".", 1)[0],
                    service=service,
                    entity_id=entity_id,
                    data=data,
                    description=f"set {entity} to {level}%",
                    source="transcript",
                )

        activate = self._ACTIVATE_PATTERN.search(text)
        if activate:
            entity = activate.group("entity").strip()
            entity_id = self._resolve_entity(entity)
            if entity_id:
                domain = entity_id.split(".", 1)[0]
                service = "turn_on" if domain in {"scene", "script"} else "start"
                return IntentMatch(
                    domain=domain,
                    service=service,
                    entity_id=entity_id,
                    data={"entity_id": entity_id},
                    description=f"activate {entity}",
                    source="transcript",
                )

        lock = self._LOCK_PATTERN.search(text)
        if lock:
            entity = lock.group("entity").strip()
            entity_id = self._resolve_entity(entity)
            if entity_id and entity_id.startswith("lock."):
                intent = lock.group(1).lower()
                service = "lock" if intent == "lock" else "unlock"
                return IntentMatch(
                    domain="lock",
                    service=service,
                    entity_id=entity_id,
                    data={"entity_id": entity_id},
                    description=f"{service} {entity}",
                    source="transcript",
                )

        return None

    def _execute_intent(self, intent: IntentMatch) -> tuple[bool, str]:
        try:
            self._request(
                "POST",
                f"/api/services/{intent.domain}/{intent.service}",
                json=intent.data,
            )
            pretty = intent.description.capitalize()
            return True, f"{pretty}."
        except self._request_exception() as exc:
            logger.warning("Home Assistant request failed: %s", exc)
            return False, str(exc)

    def _parse_params(self, params: str) -> dict[str, Any]:
        data: dict[str, Any] = {}
        if not params:
            return data
        parts = re.split(r"[;\s]+", params.strip())
        for part in parts:
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            key = key.strip().lower()
            value = value.strip()
            if value.isdigit():
                data[key] = int(value)
            else:
                try:
                    data[key] = float(value)
                except ValueError:
                    data[key] = value
        return data

    def _resolve_entity(self, name: str) -> str | None:
        key = name.strip().lower()
        if not key:
            return None
        if key in self._entity_map:
            return self._entity_map[key]
        for alias, entity_id in self._entity_map.items():
            if key in alias or alias in key:
                return entity_id

        self._refresh_entity_cache()
        if key in self._entity_cache:
            return self._entity_cache[key]

        for alias, entity_id in self._entity_cache.items():
            if key in alias or alias in key:
                return entity_id
        return None

    def _refresh_entity_cache(self, force: bool = False) -> None:
        if not self.enabled:
            return
        with self._lock:
            if not force and (time.perf_counter() - self._entity_cache_ts) < self._entity_cache_ttl:
                return
            try:
                payload = self._request("GET", "/api/states")
            except self._request_exception() as exc:
                logger.debug("Failed to refresh Home Assistant entities: %s", exc)
                return

            cache: dict[str, str] = {}
            for item in payload if isinstance(payload, list) else []:
                entity_id = item.get("entity_id")
                friendly = (item.get("attributes") or {}).get("friendly_name")
                if isinstance(entity_id, str) and isinstance(friendly, str):
                    cache[friendly.lower()] = entity_id
            self._entity_cache = cache
            self._entity_cache_ts = time.perf_counter()

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        if not self.enabled:
            raise RuntimeError("Home Assistant bridge is not configured.")
        url = f"{self._base_url}{path}"
        headers = kwargs.pop("headers", {})
        headers["Authorization"] = f"Bearer {self._token}"
        headers["Content-Type"] = "application/json"
        response = self._session.request(
            method=method,
            url=url,
            headers=headers,
            timeout=self._timeout,
            verify=self._verify_ssl,
            **kwargs,
        )
        response.raise_for_status()
        if response.content:
            try:
                return response.json()
            except ValueError:
                return response.text
        return {}

    def _log_event(self, intent: IntentMatch, success: bool, message: str) -> None:
        try:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            status = "SUCCESS" if success else "ERROR"
            with self._log_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    f"[{_utc_iso()}] {status} {intent.source} {intent.domain}.{intent.service} "
                    f"{intent.entity_id} -> {message}\n"
                )
        except Exception:  # pragma: no cover - logging must not fail flow
            logger.debug("Unable to write HA bridge log.", exc_info=True)


def create_blueprint(bridge: HABridge | None = None) -> Blueprint:
    """Create a Flask blueprint exposing HA bridge helpers."""
    Blueprint, jsonify, request = _require_flask()
    bridge = bridge or HABridge()
    bp = Blueprint("ha_bridge", __name__)

    from rex.http_errors import (  # noqa: PLC0415
        BAD_REQUEST,
        FORBIDDEN,
        INTERNAL_ERROR,
        SERVICE_UNAVAILABLE,
        error_response,
    )

    @bp.before_request
    def _validate_secret() -> Any:
        secret = bridge.secret
        if secret and request.headers.get("HASS_SECRET") != secret:
            return error_response(FORBIDDEN, "Forbidden", 403)
        return None

    def _require_enabled():
        if not bridge.enabled:
            return error_response(SERVICE_UNAVAILABLE, "Home Assistant bridge disabled", 503)
        return None

    @bp.route("/ha/intents", methods=["GET"])
    def list_intents():
        if (resp := _require_enabled()) is not None:
            return resp
        return jsonify({"intents": HABridge.SUPPORTED_INTENTS})

    @bp.route("/ha/entities", methods=["GET"])
    def list_entities():
        if (resp := _require_enabled()) is not None:
            return resp
        try:
            entities = bridge.list_entities()
            return jsonify({"entities": entities})
        except Exception as exc:
            logger.warning("Failed to list Home Assistant entities: %s", exc)
            return error_response(INTERNAL_ERROR, str(exc), 500)

    @bp.route("/ha/script", methods=["POST"])
    def run_script():
        if (resp := _require_enabled()) is not None:
            return resp
        payload = request.get_json(silent=True) or {}
        script_id = payload.get("script")
        variables = payload.get("variables") or {}
        if not script_id:
            return error_response(BAD_REQUEST, "script field is required", 400)
        if not isinstance(script_id, str):
            return error_response(BAD_REQUEST, "script must be a string", 400)
        if len(script_id) > 256:
            return error_response(
                BAD_REQUEST, "script exceeds maximum length of 256 characters", 400
            )
        if not isinstance(variables, dict):
            return error_response(BAD_REQUEST, "variables must be an object", 400)
        try:
            bridge.call_script(script_id, variables)
            return jsonify({"status": "ok", "script": script_id})
        except Exception as exc:
            logger.warning("Failed to execute Home Assistant script: %s", exc)
            return error_response(INTERNAL_ERROR, str(exc), 500)

    return bp  # type: ignore[no-any-return]
