from sqlalchemy.exc import OperationalError


class ProcessingJobRetryPolicy:
    """判断任务异常是否适合由 Celery 做短暂重试。"""

    RETRYABLE_STATUS_CODES = {408, 425, 429, 500, 502, 503, 504}
    RETRYABLE_ERROR_NAMES = {
        "APIConnectionError",
        "APITimeoutError",
        "ConnectError",
        "ConnectTimeout",
        "ConnectionError",
        "ConnectionResetError",
        "InternalServerError",
        "NetworkError",
        "PoolTimeout",
        "RateLimitError",
        "ReadError",
        "ReadTimeout",
        "RemoteProtocolError",
        "TimeoutError",
        "WriteError",
        "WriteTimeout",
    }

    def __init__(self, base_delay_seconds: int, max_delay_seconds: int) -> None:
        if base_delay_seconds <= 0:
            raise ValueError("base_delay_seconds must be greater than zero")
        if max_delay_seconds < base_delay_seconds:
            raise ValueError("max_delay_seconds must be >= base_delay_seconds")

        self.base_delay_seconds = base_delay_seconds
        self.max_delay_seconds = max_delay_seconds

    def should_retry(self, exc: Exception) -> bool:
        """仅重试网络、限流、服务端错误和数据库连接类瞬时异常。"""
        for current in self._walk_exception_chain(exc):
            if isinstance(current, (TimeoutError, ConnectionError, OperationalError)):
                return True

            status_code = getattr(current, "status_code", None)
            if isinstance(status_code, int) and status_code in self.RETRYABLE_STATUS_CODES:
                return True

            if type(current).__name__ in self.RETRYABLE_ERROR_NAMES:
                return True

        return False

    def retry_delay_seconds(self, current_retry_count: int) -> int:
        """按 1、2、4... 倍指数退避计算下一次重试延迟。"""
        if current_retry_count < 0:
            raise ValueError("current_retry_count cannot be negative")

        delay = self.base_delay_seconds * (2 ** current_retry_count)
        return min(delay, self.max_delay_seconds)

    @staticmethod
    def _walk_exception_chain(exc: Exception):
        """遍历异常及其 cause/context，避免 SDK 包装后丢失瞬时错误类型。"""
        current: BaseException | None = exc
        visited: set[int] = set()

        while current is not None and id(current) not in visited:
            visited.add(id(current))
            yield current
            current = current.__cause__ or current.__context__
