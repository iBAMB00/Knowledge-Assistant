from contextvars import ContextVar, Token
import re
from uuid import uuid4


REQUEST_ID_HEADER = "X-Request-ID"
DEFAULT_REQUEST_ID = "-"
MAX_REQUEST_ID_LENGTH = 64
_REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")

_request_id_context: ContextVar[str] = ContextVar(
    "request_id",
    default=DEFAULT_REQUEST_ID,
)


def resolve_request_id(candidate: str | None) -> str:
    """校验外部请求ID；非法或缺失时生成新的服务端请求ID。"""

    if candidate is not None:
        normalized = candidate.strip()
        if (
            normalized
            and len(normalized) <= MAX_REQUEST_ID_LENGTH
            and _REQUEST_ID_PATTERN.fullmatch(normalized)
        ):
            return normalized

    return uuid4().hex


def get_request_id() -> str:
    """获取当前执行上下文中的请求ID。"""

    return _request_id_context.get()


def set_request_id(request_id: str) -> Token[str]:
    """设置当前执行上下文的请求ID，并返回可用于恢复的Token。"""

    return _request_id_context.set(request_id)


def reset_request_id(token: Token[str]) -> None:
    """恢复进入当前请求前的请求ID上下文。"""

    _request_id_context.reset(token)
