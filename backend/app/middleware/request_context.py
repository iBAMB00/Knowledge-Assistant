import logging
from time import perf_counter
from typing import Any

from app.core.request_context import (
    REQUEST_ID_HEADER,
    reset_request_id,
    resolve_request_id,
    set_request_id,
)


logger = logging.getLogger(__name__)


class RequestContextMiddleware:
    """为单次HTTP请求建立Request ID上下文并记录请求级耗时。"""

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(
        self,
        scope: dict[str, Any],
        receive: Any,
        send: Any,
    ) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        request_id = resolve_request_id(
            self._read_request_id(scope)
        )
        token = set_request_id(request_id)
        started_at = perf_counter()
        status_code = 500

        async def send_with_request_id(
            message: dict[str, Any],
        ) -> None:
            nonlocal status_code

            if message.get("type") == "http.response.start":
                status_code = int(message.get("status", 500))
                headers = list(message.get("headers", []))
                headers = [
                    (name, value)
                    for name, value in headers
                    if name.lower() != b"x-request-id"
                ]
                headers.append(
                    (
                        REQUEST_ID_HEADER.lower().encode("ascii"),
                        request_id.encode("ascii"),
                    )
                )
                message = {
                    **message,
                    "headers": headers,
                }

            await send(message)

        try:
            await self.app(
                scope,
                receive,
                send_with_request_id,
            )
        except Exception:
            logger.exception(
                "request failed: method=%s path=%s total_ms=%.2f",
                scope.get("method", "-"),
                scope.get("path", "-"),
                self._elapsed_ms(started_at),
            )
            raise
        else:
            logger.info(
                "request completed: method=%s path=%s "
                "status_code=%d total_ms=%.2f",
                scope.get("method", "-"),
                scope.get("path", "-"),
                status_code,
                self._elapsed_ms(started_at),
            )
        finally:
            reset_request_id(token)

    @staticmethod
    def _read_request_id(
        scope: dict[str, Any],
    ) -> str | None:
        """从ASGI Header中读取客户端传入的Request ID。"""

        for raw_name, raw_value in scope.get("headers", []):
            if raw_name.lower() != b"x-request-id":
                continue

            try:
                return raw_value.decode("ascii")
            except UnicodeDecodeError:
                return None

        return None

    @staticmethod
    def _elapsed_ms(started_at: float) -> float:
        """计算请求耗时，单位为毫秒。"""

        return (perf_counter() - started_at) * 1000
