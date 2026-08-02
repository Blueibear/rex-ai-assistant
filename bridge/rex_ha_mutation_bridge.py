"""Electron bridge for policy-controlled Home Assistant mutations."""

from __future__ import annotations

import hashlib
import json
import sys
from typing import Any

import requests

from rex.bridge_utils import bridge_safe_error_response
from rex.config import settings
from rex.ha.mutation_service import HAMutation, HAMutationService
from rex.identity import validate_user_id


class RequestsHAClient:
    def __init__(self, base_url: str, token: str, *, timeout: float, verify_ssl: bool) -> None:
        self._base_url = base_url.rstrip("/")
        self._headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        self._timeout = timeout
        self._verify_ssl = verify_ssl

    def call_service(self, domain: str, service: str, data: dict[str, Any]) -> None:
        response = requests.post(
            f"{self._base_url}/api/services/{domain}/{service}",
            headers=self._headers,
            json=data,
            timeout=self._timeout,
            verify=self._verify_ssl,
        )
        response.raise_for_status()

    def get_state(self, entity_id: str) -> dict[str, Any] | None:
        response = requests.get(
            f"{self._base_url}/api/states/{entity_id}",
            headers=self._headers,
            timeout=self._timeout,
            verify=self._verify_ssl,
        )
        if response.status_code == 404:
            return None
        response.raise_for_status()
        value = response.json()
        return value if isinstance(value, dict) else None


def main() -> None:
    try:
        payload = json.loads(sys.stdin.read())
        user_id = validate_user_id(str(payload.get("user") or ""))
        if payload.get("data_scope") != "private":
            raise PermissionError("Home Assistant mutations require private Electron scope")
        entity_id = str(payload.get("entity_id") or "")
        domain = str(payload.get("domain") or "")
        service = str(payload.get("service") or "")
        parameters = payload.get("parameters") or {}
        if not isinstance(parameters, dict):
            raise ValueError("parameters must be an object")
        request_id = str(payload.get("request_id") or "")
        confirmation_token = payload.get("confirmation_token")
        if confirmation_token is not None and not isinstance(confirmation_token, str):
            raise ValueError("confirmation_token must be a string")

        base_url = str(getattr(settings, "ha_base_url", "") or "")
        token = str(getattr(settings, "ha_token", "") or "")
        if not base_url or not token:
            raise RuntimeError("Home Assistant is not configured")
        secret = hashlib.sha256(f"askrex-ha-confirmation:{token}".encode()).digest()
        client = RequestsHAClient(
            base_url,
            token,
            timeout=float(getattr(settings, "ha_timeout", 5.0)),
            verify_ssl=bool(getattr(settings, "ha_verify_ssl", True)),
        )
        result = HAMutationService(client, confirmation_secret=secret).execute(
            HAMutation(
                user_id=user_id,
                entity_id=entity_id,
                domain=domain,
                service=service,
                parameters=parameters,
                request_id=request_id,
                confirmation_token=confirmation_token,
            )
        )
        print(json.dumps({"ok": True, **result.to_dict()}), flush=True)
    except Exception as exc:
        print(
            json.dumps(
                bridge_safe_error_response(
                    exc,
                    messages={
                        PermissionError: (
                            "Home Assistant mutations require private Electron scope"
                        ),
                        ValueError: "Home Assistant mutation request is invalid",
                        RuntimeError: "Home Assistant is not configured",
                    },
                    default="Home Assistant mutation failed",
                )
            ),
            flush=True,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
