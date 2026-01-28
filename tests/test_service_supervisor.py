from __future__ import annotations

import json
import socket
import time
import urllib.request

from rex.service_supervisor import ServiceSupervisor


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_service_restart_on_failed_health() -> None:
    supervisor = ServiceSupervisor(health_check_port=_free_port())
    state = {"running": False, "healthy": True}

    def start():
        state["running"] = True

    def stop():
        state["running"] = False

    def health():
        return state["healthy"]

    supervisor.register_service(
        name="demo",
        start_func=start,
        stop_func=stop,
        health_check_func=health,
        max_restart_attempts=1,
        initial_backoff_seconds=0.0,
        backoff_multiplier=1.0,
    )

    service = supervisor.services["demo"]
    supervisor._start_service(service)
    assert service.is_running is True

    state["healthy"] = False
    supervisor._perform_health_check()

    assert service.restart_count == 1


def test_metrics_endpoint_returns_json() -> None:
    port = _free_port()
    supervisor = ServiceSupervisor(health_check_port=port)

    def start():
        return None

    def stop():
        return None

    supervisor.register_service(
        name="demo",
        start_func=start,
        stop_func=stop,
        health_check_func=lambda: True,
    )

    supervisor.start()
    try:
        time.sleep(0.2)
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=5) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        assert "services" in payload
        assert "demo" in payload["services"]
    finally:
        supervisor.stop()
