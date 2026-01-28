"""
Runtimee entrypoint for Rex AI Assistant.

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
from pathlib import Path

from rex.logging_utils import configure_logging
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

    args = parser.parse_args(argv)

    # Configure logging
    configure_logging(level=args.log_level)
    logger.info(f"Starting Rex runtime on health port {args.port}")

    try:
        # Initialize core services
        logger.info("Initializing core services...")
        services = initialize_services()

        # Create supervisor
        global _supervisor
        _supervisor = ServiceSupervisor(health_check_port=args.port)

        # Register services
        logger.info("Registering managed services...")

        # Scheduler service
        def start_scheduler():
            services.scheduler.start()

        def stop_scheduler():
            services.scheduler.stop()

        def check_scheduler():
            # Simple health check: scheduler thread is alive
            return services.scheduler.is_running

        _supervisor.register_service(
            name="scheduler",
            start_func=start_scheduler,
            stop_func=stop_scheduler,
            health_check_func=check_scheduler,
        )

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
        )

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

        # Setup graceful shutdown
        _setup_signal_handlers(_supervisor)

        # Start supervisor
        logger.info("Starting service supervisor")
        _supervisor.start()

        # Wait indefinitely (supervisor runs in background threads)
        logger.info("Rex runtime is now running")
        logger.info(f"Health check available at http://127.0.0.1:{args.port}/health")
        logger.info(f"Ready check available at http://127.0.0.1:{args.port}/ready")

        # Keep the main thread alive
        while True:
            signal.pause()

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
