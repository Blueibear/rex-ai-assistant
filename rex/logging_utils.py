def configure_logging(
    level: int = logging.INFO, handlers: Iterable[logging.Handler] | None = None
) -> None:
    """Configure application-wide logging with optional file handlers.

    By default, logs to stdout. File logging is enabled only if:
    - REX_FILE_LOGGING_ENABLED=true (default: false in containers)
    - Or explicitly requested via handlers parameter

    This makes logging container-friendly and follows 12-factor app principles.
    """

    root_logger = logging.getLogger()
    if root_logger.handlers:
        return

    if handlers is None:
        import sys

        # Stream handler (stdout)
        stream_handler = logging.StreamHandler(sys.stdout)
        if hasattr(sys.stdout, 'reconfigure'):
            try:
                sys.stdout.reconfigure(encoding='utf-8', errors='replace')
            except (AttributeError, ValueError):
                pass  # Fallback if reconfigure fails

        handlers_list = [stream_handler]

        # Enable file logging only if explicitly allowed via env
        file_logging_enabled = os.getenv("REX_FILE_LOGGING_ENABLED", "false").lower() in {
            "true", "1", "yes"
        }

        if file_logging_enabled:
            log_path = _resolve_path(
                getattr(settings, "log_path", DEFAULT_LOG_FILE), DEFAULT_LOG_FILE
            )
            error_path = _resolve_path(
                getattr(settings, "error_log_path", DEFAULT_ERROR_FILE), DEFAULT_ERROR_FILE
            )

            log_path.parent.mkdir(parents=True, exist_ok=True)
            error_path.parent.mkdir(parents=True, exist_ok=True)

            file_handler = RotatingFileHandler(log_path, maxBytes=1_000_000, backupCount=5, encoding="utf-8")
            error_handler = RotatingFileHandler(error_path, maxBytes=1_000_000, backupCount=5, encoding="utf-8")
            error_handler.setLevel(logging.ERROR)

            handlers_list.extend([file_handler, error_handler])

        handlers = tuple(handlers_list)

    logging.basicConfig(level=level, format=LOG_FORMAT, handlers=list(handlers))
