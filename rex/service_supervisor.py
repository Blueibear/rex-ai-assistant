"""
Service supervision and health management for Rex AI Assistant.

Provides automatic service restart on crash, exponential backoff retry logic,
and a simple health check endpoint.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, HTTPServer

logger = logging.getLogger(__name__)


@dataclass
class ServiceHealthStatus:
    """Health status of a single service."""

    name: str
    is_running: bool
    last_checked: datetime = field(default_factory=lambda: datetime.now(UTC))
    last_error: str | None = None
    restart_count: int = 0
    uptime_seconds: float = 0.0


@dataclass
class ServiceMetrics:
    """Cumulative metrics for service supervision."""

    total_restarts: int = 0
    total_crashes: int = 0
    total_uptime_seconds: float = 0.0
    last_health_check: datetime = field(default_factory=lambda: datetime.now(UTC))
    successful_checks: int = 0
    failed_checks: int = 0


class ManagedService:
    """Represents a service managed by the supervisor."""

    def __init__(
        self,
        name: str,
        start_func: Callable[[], None],
        stop_func: Callable[[], None],
        health_check_func: Callable[[], bool] = None,  # type: ignore[assignment]
        metrics_func: Callable[[], dict] | None = None,
        max_restart_attempts: int = 5,
        backoff_multiplier: float = 1.5,
        initial_backoff_seconds: float = 1.0,
    ):
        self.name = name
        self.start_func = start_func
        self.stop_func = stop_func
        self.health_check_func = health_check_func or (lambda: True)  # type: ignore[truthy-function]
        self.metrics_func = metrics_func or (lambda: {})
        self.max_restart_attempts = max_restart_attempts
        self.backoff_multiplier = backoff_multiplier
        self.initial_backoff_seconds = initial_backoff_seconds

        self.thread: threading.Thread | None = None
        self.is_running = False
        self.restart_count = 0
        self.start_time: datetime | None = None
        self.last_error: str | None = None

    def status(self) -> ServiceHealthStatus:
        """Get current health status."""
        uptime = 0.0
        if self.start_time:
            uptime = (datetime.now(UTC) - self.start_time).total_seconds()
        return ServiceHealthStatus(
            name=self.name,
            is_running=self.is_running,
            last_error=self.last_error,
            restart_count=self.restart_count,
            uptime_seconds=uptime,
        )


class ServiceSupervisor:
    """
    Supervises critical services with automatic restart and health checks.

    Manages service lifecycle with exponential backoff retry logic and
    provides a simple health check endpoint.
    """

    def __init__(self, health_check_port: int = 8765):
        self.services: dict[str, ManagedService] = {}
        self.metrics = ServiceMetrics()
        self.health_check_port = health_check_port
        self.is_running = False
        self._health_check_thread: threading.Thread | None = None
        self._http_server_thread: threading.Thread | None = None
        self._http_server: HTTPServer | None = None
        self._lock = threading.RLock()

    def register_service(
        self,
        name: str,
        start_func: Callable[[], None],
        stop_func: Callable[[], None],
        health_check_func: Callable[[], bool] = None,  # type: ignore[assignment]
        metrics_func: Callable[[], dict] | None = None,
        max_restart_attempts: int = 5,
        backoff_multiplier: float = 1.5,
        initial_backoff_seconds: float = 1.0,
    ) -> None:
        """Register a service to be supervised."""
        with self._lock:
            service = ManagedService(
                name=name,
                start_func=start_func,
                stop_func=stop_func,
                health_check_func=health_check_func,
                metrics_func=metrics_func,
                max_restart_attempts=max_restart_attempts,
                backoff_multiplier=backoff_multiplier,
                initial_backoff_seconds=initial_backoff_seconds,
            )
            self.services[name] = service
            logger.info(f"Registered service: {name}")

    def start(self) -> None:
        """Start all registered services."""
        with self._lock:
            if self.is_running:
                logger.warning("Supervisor is already running")
                return

            logger.info("Starting service supervisor")
            self.is_running = True

            # Start all services
            for service in self.services.values():
                self._start_service(service)

            # Start health check loop
            self._health_check_thread = threading.Thread(
                target=self._health_check_loop, daemon=True, name="rex-health-check"
            )
            self._health_check_thread.start()

            # Start HTTP health endpoint
            self._http_server_thread = threading.Thread(
                target=self._run_health_server, daemon=True, name="rex-health-server"
            )
            self._http_server_thread.start()
            logger.info(f"Health check server started on port {self.health_check_port}")

    def stop(self) -> None:
        """Stop all services gracefully."""
        with self._lock:
            if not self.is_running:
                logger.warning("Supervisor is not running")
                return

            logger.info("Stopping service supervisor")
            self.is_running = False

            # Stop all services in reverse order
            for service in reversed(list(self.services.values())):
                self._stop_service(service)

            # Stop HTTP server
            if self._http_server:
                self._http_server.shutdown()
                self._http_server = None

    def _start_service(self, service: ManagedService) -> None:
        """Start a single service with error handling."""
        try:
            logger.info(f"Starting service: {service.name}")
            service.start_time = datetime.now(UTC)
            service.is_running = True
            service.start_func()
            logger.info(f"Service started: {service.name}")
        except Exception as e:
            logger.error(f"Failed to start service {service.name}: {e}")
            service.is_running = False
            service.last_error = str(e)

    def _stop_service(self, service: ManagedService) -> None:
        """Stop a single service gracefully."""
        try:
            logger.info(f"Stopping service: {service.name}")
            service.stop_func()
            service.is_running = False
            logger.info(f"Service stopped: {service.name}")
        except Exception as e:
            logger.error(f"Error stopping service {service.name}: {e}")

    def _health_check_loop(self) -> None:
        """Continuously monitor service health."""
        while self.is_running:
            try:
                self._perform_health_check()
                time.sleep(5)  # Check every 5 seconds
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
                self.metrics.failed_checks += 1
                time.sleep(5)

    def _perform_health_check(self) -> None:
        """Check health of all services and restart if needed."""
        with self._lock:
            self.metrics.last_health_check = datetime.now(UTC)
            all_healthy = True

            for service in self.services.values():
                try:
                    # Check if service thinks it's healthy
                    if service.is_running and not service.health_check_func():
                        logger.warning(f"Health check failed for {service.name}")
                        all_healthy = False
                        self._restart_service(service)
                except Exception as e:
                    logger.error(f"Health check exception for {service.name}: {e}")
                    all_healthy = False
                    self._restart_service(service)

            if all_healthy:
                self.metrics.successful_checks += 1
            else:
                self.metrics.failed_checks += 1

    def _restart_service(self, service: ManagedService, attempt: int = 0) -> None:
        """Restart a service with exponential backoff."""
        if attempt >= service.max_restart_attempts:
            logger.error(
                f"Service {service.name} exceeded max restart attempts ({service.max_restart_attempts})"
            )
            service.is_running = False
            return

        # Calculate backoff
        backoff_seconds = service.initial_backoff_seconds * (service.backoff_multiplier**attempt)
        logger.info(
            f"Restarting service {service.name} in {backoff_seconds:.1f}s (attempt {attempt + 1}/{service.max_restart_attempts})"
        )

        # Wait with backoff
        time.sleep(backoff_seconds)

        # Try to restart
        try:
            self._stop_service(service)
            time.sleep(0.5)
            self._start_service(service)
            service.restart_count += 1
            self.metrics.total_restarts += 1
            logger.info(f"Service {service.name} restarted successfully")
        except Exception as e:
            logger.error(f"Failed to restart {service.name}: {e}")
            service.last_error = str(e)
            self.metrics.total_crashes += 1
            # Retry with next backoff
            self._restart_service(service, attempt + 1)

    def get_service_status(self, service_name: str) -> ServiceHealthStatus | None:
        """Get health status of a specific service."""
        with self._lock:
            service = self.services.get(service_name)
            if service:
                return service.status()
            return None

    def get_all_status(self) -> dict[str, ServiceHealthStatus]:
        """Get health status of all services."""
        with self._lock:
            return {name: service.status() for name, service in self.services.items()}

    def get_all_metrics(self) -> dict[str, dict]:
        """Get metrics payload from all services."""
        with self._lock:
            metrics: dict[str, dict] = {}
            for name, service in self.services.items():
                try:
                    metrics[name] = service.metrics_func()
                except Exception as exc:
                    logger.warning("Failed to collect metrics for %s: %s", name, exc)
                    metrics[name] = {"error": str(exc)}
            return metrics

    def get_metrics(self) -> ServiceMetrics:
        """Get cumulative supervisor metrics."""
        with self._lock:
            return self.metrics

    def _run_health_server(self) -> None:
        """Run HTTP health check server."""
        try:
            handler = self._create_health_handler()
            self._http_server = HTTPServer(("127.0.0.1", self.health_check_port), handler)
            logger.info(f"Health server listening on port {self.health_check_port}")
            self._http_server.serve_forever()
        except Exception as e:
            logger.error(f"Failed to start health server: {e}")

    def _create_health_handler(self):
        """Create HTTP request handler for health checks."""
        supervisor = self

        class HealthHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/health":
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()

                    status = supervisor.get_all_status()
                    all_healthy = all(s.is_running for s in status.values())

                    response = {
                        "status": "healthy" if all_healthy else "degraded",
                        "services": {
                            name: {
                                "running": s.is_running,
                                "restarts": s.restart_count,
                                "uptime_seconds": s.uptime_seconds,
                                "error": s.last_error,
                            }
                            for name, s in status.items()
                        },
                        "service_metrics": supervisor.get_all_metrics(),
                        "metrics": {
                            "total_restarts": supervisor.metrics.total_restarts,
                            "total_crashes": supervisor.metrics.total_crashes,
                            "successful_checks": supervisor.metrics.successful_checks,
                            "failed_checks": supervisor.metrics.failed_checks,
                        },
                    }
                    self.wfile.write(json.dumps(response).encode())
                elif self.path == "/ready":
                    status = supervisor.get_all_status()
                    all_ready = all(s.is_running for s in status.values())
                    if all_ready:
                        self.send_response(200)
                        self.send_header("Content-Type", "text/plain")
                        self.end_headers()
                        self.wfile.write(b"Ready")
                    else:
                        self.send_response(503)
                        self.send_header("Content-Type", "text/plain")
                        self.end_headers()
                        self.wfile.write(b"Not ready")
                elif self.path == "/metrics":
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    payload = {
                        "supervisor": {
                            "total_restarts": supervisor.metrics.total_restarts,
                            "total_crashes": supervisor.metrics.total_crashes,
                            "successful_checks": supervisor.metrics.successful_checks,
                            "failed_checks": supervisor.metrics.failed_checks,
                        },
                        "services": {
                            name: {
                                "status": status.is_running,
                                "uptime_seconds": status.uptime_seconds,
                                "restarts": status.restart_count,
                                "last_error": status.last_error,
                                "metrics": supervisor.services[name].metrics_func(),
                            }
                            for name, status in supervisor.get_all_status().items()
                        },
                    }
                    self.wfile.write(json.dumps(payload).encode())
                else:
                    self.send_response(404)
                    self.send_header("Content-Type", "text/plain")
                    self.end_headers()
                    self.wfile.write(b"Not found")

            def log_message(self, format, *args):
                # Suppress default logging
                pass

        return HealthHandler


__all__ = [
    "ServiceSupervisor",
    "ManagedService",
    "ServiceHealthStatus",
    "ServiceMetrics",
]
