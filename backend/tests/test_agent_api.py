from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

import app.api.agent as agent_api
from app.agent.context import ToolExecutionContext
from app.agent.native_agent import (
    AgentTurnLimitError,
    NativeAgentResult,
)
from app.api.dependencies.agent import (
    get_agent_access_policy,
    get_agent_runtime_selector,
)
from app.api.dependencies.auth import get_current_user
from app.core.database import get_db
from app.middleware.request_context import RequestContextMiddleware
from app.models.database.user import User
from app.constants.agent_runtime import AgentRuntime
from app.services.knowledge_base_access_policy import ResourceAccessNotFoundError


class FakeAccessPolicy:
    """记录 Agent API 是否先完成知识库授权。"""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.error: Exception | None = None

    def get_accessible_knowledge_base(self, **kwargs: Any) -> object:
        self.calls.append(kwargs)

        if self.error is not None:
            raise self.error

        return object()


class FakeAgentRunner:
    """记录 HTTP 层实际注入给 Agent Runtime 的可信上下文。"""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.error: Exception | None = None
        self.result = NativeAgentResult(
            answer="根据企业知识库，建议先完成灰度验证。",
            turns=2,
            tool_call_count=1,
        )

    def run(
        self,
        *,
        db: Session,
        context: ToolExecutionContext,
        message: str,
    ) -> NativeAgentResult:
        self.calls.append(
            {
                "db": db,
                "context": context,
                "message": message,
            }
        )

        if self.error is not None:
            raise self.error

        if not message or not message.strip():
            raise ValueError("message cannot be empty")

        return self.result

class FakeRuntimeSelector:
    """为既有 Native API 测试固定选择同一个 Fake Runner。"""

    def __init__(self, runner: FakeAgentRunner) -> None:
        self.runner = runner
        self.runtimes: list[AgentRuntime] = []

    def select(self, runtime: AgentRuntime) -> FakeAgentRunner:
        self.runtimes.append(runtime)
        return self.runner


@pytest.fixture
def client(
    db: Session,
) -> tuple[TestClient, FakeAccessPolicy, FakeAgentRunner]:
    """创建只包含 Agent Router 的最小测试应用。"""

    access_policy = FakeAccessPolicy()
    runner = FakeAgentRunner()

    app = FastAPI()
    app.add_middleware(RequestContextMiddleware)
    app.include_router(agent_api.router)

    def override_get_db():
        yield db

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_current_user] = lambda: User(
        id=7,
        email="agent-user@example.com",
        password_hash="hash",
        role="user",
        is_active=True,
    )
    app.dependency_overrides[get_agent_access_policy] = lambda: access_policy
    selector = FakeRuntimeSelector(runner)
    app.dependency_overrides[get_agent_runtime_selector] = lambda: selector

    return TestClient(app), access_policy, runner


def test_agent_chat_builds_trusted_context_after_permission_check(
    client: tuple[TestClient, FakeAccessPolicy, FakeAgentRunner],
) -> None:
    """HTTP 身份、KB Scope、Request ID 必须由服务端组装进 Context。"""

    test_client, access_policy, runner = client

    response = test_client.post(
        "/agent/chat",
        headers={"X-Request-ID": "agent-request-001"},
        json={
            "message": "请查询生产部署要求",
            "knowledge_base_id": 21,
        },
    )

    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "agent-request-001"
    assert response.json() == {
        "answer": "根据企业知识库，建议先完成灰度验证。"
    }

    assert len(access_policy.calls) == 1
    assert access_policy.calls[0]["knowledge_base_id"] == 21
    assert access_policy.calls[0]["user"].id == 7

    assert len(runner.calls) == 1
    context = runner.calls[0]["context"]

    assert runner.calls[0]["message"] == "请查询生产部署要求"
    assert context.user_id == 7
    assert context.role.value == "user"
    assert context.knowledge_base_id == 21
    assert context.request_id == "agent-request-001"
    assert context.agent_run_id is None


def test_agent_chat_rejects_inaccessible_kb_before_agent_runs(
    client: tuple[TestClient, FakeAccessPolicy, FakeAgentRunner],
) -> None:
    """未授权 KB 必须在建立 Agent 执行范围前被拒绝。"""

    test_client, access_policy, runner = client
    access_policy.error = ResourceAccessNotFoundError(
        "knowledge base not found"
    )

    response = test_client.post(
        "/agent/chat",
        json={
            "message": "即使模型想直接回答也不能绕过权限",
            "knowledge_base_id": 999,
        },
    )

    assert response.status_code == 404
    assert response.json() == {
        "detail": "knowledge base not found"
    }
    assert runner.calls == []


def test_agent_chat_maps_empty_message_to_400(
    client: tuple[TestClient, FakeAccessPolicy, FakeAgentRunner],
) -> None:
    """同步 Runner 的业务输入错误继续使用 400 语义。"""

    test_client, _, runner = client

    response = test_client.post(
        "/agent/chat",
        json={
            "message": "   ",
            "knowledge_base_id": 21,
        },
    )

    assert response.status_code == 400
    assert response.json() == {
        "detail": "message cannot be empty"
    }
    assert len(runner.calls) == 1


def test_agent_chat_maps_runtime_budget_error_to_503_without_internal_detail(
    client: tuple[TestClient, FakeAccessPolicy, FakeAgentRunner],
) -> None:
    """Agent Loop 预算/超时类失败不得把内部运行细节暴露给客户端。"""

    test_client, _, runner = client
    runner.error = AgentTurnLimitError(
        "sensitive runtime detail"
    )

    response = test_client.post(
        "/agent/chat",
        json={
            "message": "复杂任务",
            "knowledge_base_id": 21,
        },
    )

    assert response.status_code == 503
    assert response.json() == {
        "detail": "Agent暂时无法完成请求"
    }
    assert "sensitive runtime detail" not in response.text


def test_agent_chat_maps_unexpected_error_to_generic_500(
    client: tuple[TestClient, FakeAccessPolicy, FakeAgentRunner],
) -> None:
    """未知内部异常统一收口，避免 Provider/Secret 细节泄漏。"""

    test_client, _, runner = client
    runner.error = RuntimeError("provider-secret-detail")

    response = test_client.post(
        "/agent/chat",
        json={
            "message": "查询资料",
            "knowledge_base_id": 21,
        },
    )

    assert response.status_code == 500
    assert response.json() == {
        "detail": "Agent问答失败"
    }
    assert "provider-secret-detail" not in response.text


@pytest.mark.parametrize(
    ("payload", "field_name"),
    [
        (
            {
                "message": "查询资料",
                "knowledge_base_id": 0,
            },
            "knowledge_base_id",
        ),
        (
            {
                "message": "查询资料",
                "knowledge_base_id": 21,
                "user_id": 999,
            },
            "user_id",
        ),
    ],
)
def test_agent_chat_rejects_invalid_or_forged_request_fields(
    client: tuple[TestClient, FakeAccessPolicy, FakeAgentRunner],
    payload: dict[str, Any],
    field_name: str,
) -> None:
    """HTTP 请求同样不能伪造可信 Agent 身份字段。"""

    test_client, _, runner = client

    response = test_client.post(
        "/agent/chat",
        json=payload,
    )

    assert response.status_code == 422
    error_fields = [
        error["loc"][-1]
        for error in response.json()["detail"]
    ]
    assert field_name in error_fields
    assert runner.calls == []
