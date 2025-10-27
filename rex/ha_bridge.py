"""Home Assistant intent bridge for Rex."""

from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from flask import Blueprint, jsonify, request

from rex.config import settings

logger = logging.getLogger(__name__)


def _utc_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


@dataclass
class IntentMatch:
    """Represents an intent matched from natural language."""

    domain: str
    service: str
    entity_id: str
    data: Dict[str, Any]
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
        base_url: Optional[str] = None,
        token: Optional[str] = None,
        secret: Optional[str] = None,
        verify_ssl: Optional[bool] = None,
        timeout: Optional[float] = None,
        entity_map: Optional[Dict[str, str]] = None,
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
        self._session = requests.Session()
        self._entity_cache: Dict[str, str] = {}
        self._entity_cache_ts: float = 0.0
        self._entity_cache_ttl: float = 60.0
        self._lock = threading.Lock()
        self._log_path = Path("logs/test_ha_integration.log")

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

    def process_transcript(self, transcript: str) -> Optional[str]:
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

        messages: List[str] = []
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
            suffix = " ".join(messages)
            sanitized = f"{sanitized.rstrip()} {' '.join(messages)}".strip()
        return sanitized

    # ------------------------------------------------------------------ #
    # HTTP endpoints helpers
    # ------------------------------------------------------------------ #

    def list_entities(self) -> List[Dict[str, Any]]:
        """Return cached Home Assistant entities (refreshing as needed)."""
        if not self.enabled:
            raise RuntimeError("Home Assistant bridge is not configured.")
        self._refresh_entity_cache(force=True)
        result = []
        for alias, entity_id in sorted(self._entity_cache.items()):
            result.append({"friendly_name": alias, "entity_id": entity_id})
        return result

    def call_script(self, script_id: str, variables: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if not script_id:
            raise ValueError("script identifier is required")
        return self._request(
            "POST",
            f"/api/services/script/turn_on",
            json={"entity_id": script_id, "variables": variables or {}},
        )

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _match_transcript(self, transcript: str) -> Optional[IntentMatch]:
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
                    data["brightness_pct"] = level
                else:
                    data["percentage"] = level
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

    def _execute_intent(self, intent: IntentMatch) -> Tuple[bool, str]:
        try:
            response = self._request(
                "POST",
                f"/api/services/{intent.domain}/{intent.service}",
                json=intent.data,
            )
            pretty = intent.description.capitalize()
            return True, f"{pretty}."
        except requests.RequestException as exc:
            logger.warning("Home Assistant request failed: %s", exc)
            return False, str(exc)

    def _parse_params(self, params: str) -> Dict[str, Any]:
        data: Dict[str, Any] = {}
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

    def _resolve_entity(self, name: str) -> Optional[str]:
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
            except requests.RequestException as exc:
                logger.debug("Failed to refresh Home Assistant entities: %s", exc)
                return

            cache: Dict[str, str] = {}
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


def create_blueprint(bridge: Optional[HABridge] = None) -> Blueprint:
    """Create a Flask blueprint exposing HA bridge helpers."""
    bridge = bridge or HABridge()
    bp = Blueprint("ha_bridge", __name__)

    @bp.before_request
    def _validate_secret() -> Any:
        secret = bridge.secret
        if secret and request.headers.get("HASS_SECRET") != secret:
            return jsonify({"error": "Forbidden"}), 403
        return None

    def _require_enabled():
        if not bridge.enabled:
            return jsonify({"error": "Home Assistant bridge disabled"}), 503
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
            return jsonify({"error": str(exc)}), 500

    @bp.route("/ha/script", methods=["POST"])
    def run_script():
        if (resp := _require_enabled()) is not None:
            return resp
        payload = request.get_json(silent=True) or {}
        script_id = payload.get("script")
        variables = payload.get("variables") or {}
        if not script_id:
            return jsonify({"error": "script field is required"}), 400
        try:
            bridge.call_script(script_id, variables)
            return jsonify({"status": "ok", "script": script_id})
        except Exception as exc:
            logger.warning("Failed to execute Home Assistant script: %s", exc)
            return jsonify({"error": str(exc)}), 500

    return bp
