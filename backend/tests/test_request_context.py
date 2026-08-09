import re

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient

from app.core.logging_config import RequestContextFilter
from app.core.request_context import (
    DEFAULT_REQUEST_ID,
    get_request_id,
    reset_request_id,
    resolve_request_id,
    set_request_id,
)
from app.middleware.request_context import RequestContextMiddleware


def _build_client() -> TestClient:
    """创建只包含Request Context Middleware的最小测试应用。"""

    app = FastAPI()
    app.add_middleware(RequestContextMiddleware)

    @app.get("/request-id")
    def read_request_id() -> dict[str, str]:
        return {"request_id": get_request_id()}

    @app.get("/stream-request-id")
    def stream_request_id() -> StreamingResponse:
        def generate():
            yield get_request_id()

        return StreamingResponse(
            generate(),
            media_type="text/plain",
        )

    return TestClient(app)


def test_request_id_middleware_preserves_valid_client_id() -> None:
    """验证合法客户端Request ID进入上下文并回写响应头。"""

    client = _build_client()

    response = client.get(
        "/request-id",
        headers={"X-Request-ID": "client-request-123"},
    )

    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "client-request-123"
    assert response.json() == {
        "request_id": "client-request-123"
    }


def test_request_id_middleware_generates_id_for_invalid_header() -> None:
    """验证非法Request ID不会原样进入日志上下文。"""

    client = _build_client()

    response = client.get(
        "/request-id",
        headers={"X-Request-ID": "bad request id"},
    )

    generated = response.headers["X-Request-ID"]

    assert generated != "bad request id"
    assert re.fullmatch(r"[0-9a-f]{32}", generated)
    assert response.json() == {
        "request_id": generated
    }


def test_request_id_context_survives_streaming_response() -> None:
    """验证SSE/StreamingResponse执行期间仍保留同一Request ID。"""

    client = _build_client()

    response = client.get(
        "/stream-request-id",
        headers={"X-Request-ID": "stream-request-456"},
    )

    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "stream-request-456"
    assert response.text == "stream-request-456"


def test_request_context_filter_injects_request_id() -> None:
    """验证业务日志Filter会自动注入当前Request ID。"""

    import logging

    record = logging.LogRecord(
        name="app.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="test",
        args=(),
        exc_info=None,
    )

    token = set_request_id("filter-request-789")
    try:
        assert RequestContextFilter().filter(record) is True
        assert record.request_id == "filter-request-789"
    finally:
        reset_request_id(token)

    assert get_request_id() == DEFAULT_REQUEST_ID


def test_resolve_request_id_rejects_log_injection_characters() -> None:
    """验证Request ID仅接受有限安全字符。"""

    generated = resolve_request_id("unsafe\nrequest")

    assert generated != "unsafe\nrequest"
    assert re.fullmatch(r"[0-9a-f]{32}", generated)
