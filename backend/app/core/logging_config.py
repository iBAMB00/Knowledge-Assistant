import logging
import sys

from app.core.request_context import get_request_id


APP_LOGGER_NAME = "app"
_HANDLER_MARKER = "_knowledge_assistant_handler"
_REQUEST_ID_FILTER_MARKER = "_knowledge_assistant_request_id_filter"


class RequestContextFilter(logging.Filter):
    """把当前Request ID注入每条应用业务日志。"""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = get_request_id()
        return True


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
        app_logger.addHandler(handler)

    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s "
            "request_id=%(request_id)s %(message)s"
        )
    )

    if not any(
        getattr(existing_filter, _REQUEST_ID_FILTER_MARKER, False)
        for existing_filter in handler.filters
    ):
        request_id_filter = RequestContextFilter()
        setattr(
            request_id_filter,
            _REQUEST_ID_FILTER_MARKER,
            True,
        )
        handler.addFilter(request_id_filter)

    handler.setLevel(resolved_level)
