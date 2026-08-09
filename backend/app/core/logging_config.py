import logging
import sys


APP_LOGGER_NAME = "app"
_HANDLER_MARKER = "_knowledge_assistant_handler"


def configure_application_logging(log_level: str = "INFO") -> None:
    """配置应用业务日志，并避免依赖 Uvicorn 的默认 logger 层级。"""

    normalized_level = log_level.strip().upper()
    resolved_level = getattr(logging, normalized_level, None)

    if not isinstance(resolved_level, int):
        raise ValueError(f"unsupported log level: {log_level}")

    app_logger = logging.getLogger(APP_LOGGER_NAME)
    app_logger.setLevel(resolved_level)
    app_logger.propagate = False

    handler = next(
        (
            existing_handler
            for existing_handler in app_logger.handlers
            if getattr(existing_handler, _HANDLER_MARKER, False)
        ),
        None,
    )

    if handler is None:
        handler = logging.StreamHandler(sys.stdout)
        setattr(handler, _HANDLER_MARKER, True)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)s %(name)s %(message)s"
            )
        )
        app_logger.addHandler(handler)

    handler.setLevel(resolved_level)
