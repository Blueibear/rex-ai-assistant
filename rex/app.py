"""
Runtime entrypoint for Rex AI Assistant.

This module starts and supervises long-running services (scheduler, event bus,
workflow runner, etc.) with automatic restart on crash.

Usage:
    python -m rex.app [--port 8765]
    rex-run [--port 8765]
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time

from rex.audio.speaker_discovery import start_smart_speaker_discovery
from rex.credentials import get_credential_manager
from rex.logging_utils import _LEVEL_NAMES, configure_logging
from rex.memory import get_long_term_memory, get_working_memory
from rex.service_supervisor import ServiceSupervisor
from rex.services import initialize_services

logger = logging.getLogger(__name__)

# Global supervisor instance
_supervisor: ServiceSupervisor | None = None


def _setup_signal_handlers(supervisor: ServiceSupervisor) -> None:
    """Setup graceful shutdown on SIGTERM and SIGINT."""

    def handle_signal(signum, frame):
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        supervisor.stop()
        sys.exit(0)

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)


def main(argv: list[str] | None = None) -> int:
    """
    Start the Rex runtime with supervised services.

    This is the main long-running entrypoint for the system. It:
    1. Initializes logging
    2. Loads core services (scheduler, event bus, etc.)
    3. Starts the service supervisor
    4. Runs the health check server
    5. Handles graceful shutdown

    Args:
        argv: Command-line arguments (for testing)

    Returns:
        Exit code (0 on success, 1 on error)
    """
    parser = argparse.ArgumentParser(
        prog="rex-run",
        description="Start the Rex AI Assistant runtime with service supervision.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Port for health check server (default: 8765)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: INFO)",
    )
    parser.add_argument(
        "--services",
        type=str,
        default=None,
        help=(
            "Comma-separated list of services to supervise "
            "(default: scheduler,event_bus,workflow_runner,memory_store,credential_manager)"
        ),
    )

    args = parser.parse_args(argv)

    # Configure logging
    configure_logging(level=_LEVEL_NAMES.get(args.log_level.upper(), None))
    logger.info(f"Starting Rex runtime on health port {args.port}")
    start_smart_speaker_discovery()

    try:
        default_services = {
            "scheduler",
            "event_bus",
            "workflow_runner",
            "memory_store",
            "credential_manager",
        }
        if args.services:
            services_to_manage = {name.strip() for name in args.services.split(",") if name.strip()}
        else:
            services_to_manage = default_services
        unknown_services = services_to_manage - default_services
        if unknown_services:
            logger.warning(
                "Unknown service names specified: %s",
                ", ".join(sorted(unknown_services)),
            )

        # Initialize core services
        logger.info("Initializing core services...")
        services = initialize_services()

        # Create supervisor
        global _supervisor
        _supervisor = ServiceSupervisor(health_check_port=args.port)

        # Register services
        logger.info("Registering managed services...")

        if "scheduler" in services_to_manage:
            # Scheduler service
            def start_scheduler():
                services.scheduler.start()

            def stop_scheduler():
                services.scheduler.stop()

            def check_scheduler():
                # Simple health check: scheduler thread is alive
                return services.scheduler.is_running  # type: ignore[attr-defined]

            _supervisor.register_service(
                name="scheduler",
                start_func=start_scheduler,
                stop_func=stop_scheduler,
                health_check_func=check_scheduler,
                metrics_func=services.scheduler.get_metrics,
            )

        if "event_bus" in services_to_manage:
            # Event bus service
            def start_event_bus():
                # Event bus is passive, just mark as ready
                logger.info("Event bus initialized")

            def stop_event_bus():
                logger.info("Stopping event bus")

            def check_event_bus():
                return True  # Event bus is always healthy (passive)

            _supervisor.register_service(
                name="event_bus",
                start_func=start_event_bus,
                stop_func=stop_event_bus,
                health_check_func=check_event_bus,
                metrics_func=services.event_bus.get_metrics,
            )

        if "workflow_runner" in services_to_manage:
            # Workflow runner service (passive, but monitored for queue depth)
            def start_workflow_runner():
                logger.info("Workflow runner initialized")

            def stop_workflow_runner():
                logger.info("Stopping workflow runner")

            def check_workflow_runner():
                # Check if workflow runner is healthy (not overwhelmed)
                # In a real implementation, check queue depth, etc.
                return True

            _supervisor.register_service(
                name="workflow_runner",
                start_func=start_workflow_runner,
                stop_func=stop_workflow_runner,
                health_check_func=check_workflow_runner,
            )

        if "memory_store" in services_to_manage:

            def start_memory_store():
                get_working_memory()
                get_long_term_memory()
                logger.info("Memory store initialized")

            def stop_memory_store():
                logger.info("Memory store shutdown complete")

            def check_memory_store():
                return True

            def memory_metrics():
                wm = get_working_memory()
                ltm = get_long_term_memory()
                return {
                    "working_memory": wm.stats(),
                    "long_term_memory": ltm.stats(),
                }

            _supervisor.register_service(
                name="memory_store",
                start_func=start_memory_store,
                stop_func=stop_memory_store,
                health_check_func=check_memory_store,
                metrics_func=memory_metrics,
            )

        if "credential_manager" in services_to_manage:

            def start_credential_manager():
                get_credential_manager()
                logger.info("Credential manager initialized")

            def stop_credential_manager():
                logger.info("Credential manager shutdown complete")

            def check_credential_manager():
                return True

            _supervisor.register_service(
                name="credential_manager",
                start_func=start_credential_manager,
                stop_func=stop_credential_manager,
                health_check_func=check_credential_manager,
            )

        # Setup graceful shutdown
        _setup_signal_handlers(_supervisor)

        # Start supervisor
        logger.info("Starting service supervisor")
        _supervisor.start()

        # Wait indefinitely (supervisor runs in background threads)
        logger.info("Rex runtime is now running")
        logger.info(f"Health check available at http://127.0.0.1:{args.port}/health")
        logger.info(f"Ready check available at http://127.0.0.1:{args.port}/ready")

        # Keep the main thread alive (signal.pause() is Unix-only)
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        if _supervisor:
            _supervisor.stop()
        return 0
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        if _supervisor:
            _supervisor.stop()
        return 1


if __name__ == "__main__":
    sys.exit(main())
