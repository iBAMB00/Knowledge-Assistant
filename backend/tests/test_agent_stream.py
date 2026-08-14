from collections.abc import Iterator
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

import app.api.agent as agent_api
from app.agent.context import ToolExecutionContext
from app.agent.native_agent import AgentTurnLimitError
from app.agent.run_event import (
    AgentMessageEvent,
    AgentRunEvent,
    AgentStatusEvent,
    AgentToolCallEvent,
    AgentToolResultEvent,
)
from app.api.dependencies.agent import (
    get_agent_access_policy,
    get_agent_execution_service,
)
from app.api.dependencies.auth import get_current_user
from app.core.database import get_db
from app.middleware.request_context import RequestContextMiddleware
from app.models.database.user import User
from app.services.knowledge_base_access_policy import ResourceAccessNotFoundError


class FakeAccessPolicy:
    """记录 SSE 在开始流之前是否先完成 KB 授权。"""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.error: Exception | None = None

    def get_accessible_knowledge_base(self, **kwargs: Any) -> object:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return object()


class FakeStreamingAgentRunner:
    """按顺序产出预设 Runtime 事件，并记录可信上下文。"""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.events: list[AgentRunEvent] = []
        self.error: Exception | None = None
        self.closed = False

    def run_events(
        self,
        *,
        db: Session,
        context: ToolExecutionContext,
        message: str,
    ) -> Iterator[AgentRunEvent]:
        self.calls.append(
            {
                "db": db,
                "context": context,
                "message": message,
            }
        )

        try:
            for event in self.events:
                yield event

            if self.error is not None:
                raise self.error
        finally:
            self.closed = True


@pytest.fixture
def stream_client(
    db: Session,
) -> tuple[TestClient, FakeAccessPolicy, FakeStreamingAgentRunner]:
    """创建只包含 Agent Router 的 SSE 测试应用。"""

    access_policy = FakeAccessPolicy()
    runner = FakeStreamingAgentRunner()

    app = FastAPI()
    app.add_middleware(RequestContextMiddleware)
    app.include_router(agent_api.router)

    def override_get_db():
        yield db

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_current_user] = lambda: User(
        id=7,
        email="stream-agent@example.com",
        password_hash="hash",
        role="user",
        is_active=True,
    )
    app.dependency_overrides[get_agent_access_policy] = lambda: access_policy
    app.dependency_overrides[get_agent_execution_service] = lambda: runner

    return TestClient(app), access_policy, runner


def test_agent_stream_returns_safe_agent_lifecycle_events(
    stream_client: tuple[
        TestClient,
        FakeAccessPolicy,
        FakeStreamingAgentRunner,
    ],
) -> None:
    """SSE 必须表达 Agent 生命周期，但不能泄漏 Tool 参数/结果正文。"""

    client, access_policy, runner = stream_client
    runner.events = [
        AgentStatusEvent(turn=1),
        AgentToolCallEvent(
            turn=1,
            call_id="call_001",
            tool_name="search_knowledge",
        ),
        AgentToolResultEvent(
            turn=1,
            call_id="call_001",
            tool_name="search_knowledge",
            ok=True,
            duration_ms=12,
        ),
        AgentStatusEvent(turn=2),
        AgentMessageEvent(
            content="根据知识库，建议先灰度部署。",
            turns=2,
            tool_call_count=1,
        ),
    ]

    response = client.post(
        "/agent/chat/stream",
        headers={"X-Request-ID": "agent-stream-001"},
        json={
            "message": " 查询内部部署说明 ",
            "knowledge_base_id": 21,
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-cache"
    assert response.headers["X-Request-ID"] == "agent-stream-001"

    body = response.text
    assert "event: status" in body
    assert 'data: {"stage":"model","turn":1}' in body
    assert "event: tool_call" in body
    assert '"tool_name":"search_knowledge"' in body
    assert "event: tool_result" in body
    assert '"ok":true' in body
    assert "event: message" in body
    assert 'data: {"content":"根据知识库，建议先灰度部署。"}' in body
    assert "event: done\ndata: {}" in body

    # API 不暴露 Runtime 内部计数，也没有 Tool 参数/检索结果正文字段。
    assert "tool_call_count" not in body
    assert '"turns"' not in body
    assert "arguments_json" not in body
    assert "content_json" not in body
    assert "duration_ms" not in body

    assert len(access_policy.calls) == 1
    assert access_policy.calls[0]["knowledge_base_id"] == 21
    assert len(runner.calls) == 1
    assert runner.calls[0]["message"] == "查询内部部署说明"

    context = runner.calls[0]["context"]
    assert context.user_id == 7
    assert context.knowledge_base_id == 21
    assert context.request_id == "agent-stream-001"
    assert runner.closed is True


def test_agent_stream_rejects_inaccessible_kb_before_stream_starts(
    stream_client: tuple[
        TestClient,
        FakeAccessPolicy,
        FakeStreamingAgentRunner,
    ],
) -> None:
    """未授权 KB 仍保持普通 HTTP 404，不允许先发 200 SSE 再报错。"""

    client, access_policy, runner = stream_client
    access_policy.error = ResourceAccessNotFoundError(
        "knowledge base not found"
    )

    response = client.post(
        "/agent/chat/stream",
        json={
            "message": "越权查询",
            "knowledge_base_id": 999,
        },
    )

    assert response.status_code == 404
    assert response.json() == {"detail": "knowledge base not found"}
    assert runner.calls == []


def test_agent_stream_rejects_empty_message_before_stream_starts(
    stream_client: tuple[
        TestClient,
        FakeAccessPolicy,
        FakeStreamingAgentRunner,
    ],
) -> None:
    """可在建流前发现的输入错误继续使用 400，而不是 SSE error。"""

    client, _, runner = stream_client

    response = client.post(
        "/agent/chat/stream",
        json={
            "message": "   ",
            "knowledge_base_id": 21,
        },
    )

    assert response.status_code == 400
    assert response.json() == {"detail": "message cannot be empty"}
    assert runner.calls == []


def test_agent_stream_maps_runtime_error_to_safe_sse_error(
    stream_client: tuple[
        TestClient,
        FakeAccessPolicy,
        FakeStreamingAgentRunner,
    ],
) -> None:
    """流已开始后的 Budget 错误必须使用 error event，且不发送 done。"""

    client, _, runner = stream_client
    runner.events = [AgentStatusEvent(turn=1)]
    runner.error = AgentTurnLimitError("sensitive runtime detail")

    response = client.post(
        "/agent/chat/stream",
        json={
            "message": "复杂任务",
            "knowledge_base_id": 21,
        },
    )

    assert response.status_code == 200
    body = response.text
    assert "event: status" in body
    assert "event: error" in body
    assert "Agent暂时无法完成请求" in body
    assert "sensitive runtime detail" not in body
    assert "event: done" not in body
    assert runner.closed is True


def test_agent_stream_maps_unexpected_error_without_secret_leak(
    stream_client: tuple[
        TestClient,
        FakeAccessPolicy,
        FakeStreamingAgentRunner,
    ],
) -> None:
    """Provider 等未知错误在流内统一脱敏。"""

    client, _, runner = stream_client
    runner.error = RuntimeError("provider-secret-detail")

    response = client.post(
        "/agent/chat/stream",
        json={
            "message": "查询资料",
            "knowledge_base_id": 21,
        },
    )

    assert response.status_code == 200
    body = response.text
    assert "event: error" in body
    assert "Agent问答失败" in body
    assert "provider-secret-detail" not in body
    assert "event: done" not in body
    assert runner.closed is True


def test_agent_sse_generator_close_propagates_to_runtime_iterator(
    db: Session,
) -> None:
    """客户端取消 SSE 时必须关闭底层 Agent 事件生成器。"""

    runner = FakeStreamingAgentRunner()
    runner.events = [
        AgentStatusEvent(turn=1),
        AgentStatusEvent(turn=2),
    ]
    context = ToolExecutionContext(
        user_id=7,
        role="user",
        knowledge_base_id=21,
        request_id="cancel-agent-stream",
    )

    sse = agent_api.generate_agent_chat_sse(
        db=db,
        context=context,
        message="取消测试",
        agent_runner=runner,  # type: ignore[arg-type]
    )

    first_event = next(sse)
    assert "event: status" in first_event

    sse.close()

    assert runner.closed is True
