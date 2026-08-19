from collections.abc import Iterator
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

import app.api.agent as agent_api
from app.agent.context import ToolExecutionContext
from app.agent.frameworks.langchain.runner import (
    LangChainAgentResult,
    LangChainAgentTurnLimitError,
)
from app.agent.native_agent import NativeAgentResult
from app.agent.run_event import AgentMessageEvent, AgentRunEvent, AgentStatusEvent
from app.api.dependencies.agent import (
    get_agent_access_policy,
    get_agent_runtime_selector,
)
from app.api.dependencies.auth import get_current_user
from app.constants.agent_runtime import AgentRuntime
from app.core.database import get_db
from app.middleware.request_context import RequestContextMiddleware
from app.models.database.user import User
from app.services.agent_runtime_selector import (
    AgentRuntimeSelector,
    AgentRuntimeUnavailableError,
)


class FakeAccessPolicy:
    """A7 只验证 Runtime 路由；KB 权限仍先经过现有 Policy。"""

    def get_accessible_knowledge_base(self, **kwargs: Any) -> object:
        return object()


class FakeNativeService:
    """记录 Native 是否被同步 HTTP 入口选中。"""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def run(
        self,
        *,
        db: Session,
        context: ToolExecutionContext,
        message: str,
    ) -> NativeAgentResult:
        self.calls.append(
            {"db": db, "context": context, "message": message}
        )
        return NativeAgentResult(
            answer="native-answer",
            turns=1,
            tool_call_count=0,
        )

    def run_events(
        self,
        *,
        db: Session,
        context: ToolExecutionContext,
        message: str,
    ) -> Iterator[AgentRunEvent]:
        self.calls.append(
            {"db": db, "context": context, "message": message}
        )
        yield AgentStatusEvent(turn=1)
        yield AgentMessageEvent(
            content="native-answer",
            turns=1,
            tool_call_count=0,
        )


class FakeLangChainService:
    """记录 Candidate 是否被同步 HTTP 入口选中。"""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.error: Exception | None = None

    def run(
        self,
        *,
        db: Session,
        context: ToolExecutionContext,
        message: str,
    ) -> LangChainAgentResult:
        self.calls.append(
            {"db": db, "context": context, "message": message}
        )
        if self.error is not None:
            raise self.error
        return LangChainAgentResult(
            answer="langchain-answer",
            turns=1,
            tool_call_count=0,
        )

    def run_events(
        self,
        *,
        db: Session,
        context: ToolExecutionContext,
        message: str,
    ) -> Iterator[AgentRunEvent]:
        self.calls.append(
            {"db": db, "context": context, "message": message}
        )
        if self.error is not None:
            raise self.error
        yield AgentStatusEvent(turn=1)
        yield AgentMessageEvent(
            content="langchain-answer",
            turns=1,
            tool_call_count=0,
        )


def _build_client(
    *,
    db: Session,
    selector: AgentRuntimeSelector,
) -> TestClient:
    app = FastAPI()
    app.add_middleware(RequestContextMiddleware)
    app.include_router(agent_api.router)

    def override_get_db():
        yield db

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_current_user] = lambda: User(
        id=7,
        email="runtime-selector@example.com",
        password_hash="hash",
        role="user",
        is_active=True,
    )
    app.dependency_overrides[get_agent_access_policy] = FakeAccessPolicy
    app.dependency_overrides[get_agent_runtime_selector] = lambda: selector
    return TestClient(app)


def test_agent_chat_defaults_to_native_without_building_candidate(
    db: Session,
) -> None:
    """没有 runtime 参数时必须保持 Native，且不能初始化 Candidate。"""

    native = FakeNativeService()
    candidate_factory_calls = 0

    def candidate_factory() -> FakeLangChainService:
        nonlocal candidate_factory_calls
        candidate_factory_calls += 1
        return FakeLangChainService()

    selector = AgentRuntimeSelector(
        native_factory=lambda: native,
        langchain_factory=candidate_factory,
        langchain_candidate_enabled=True,
    )
    client = _build_client(db=db, selector=selector)

    response = client.post(
        "/agent/chat",
        json={"message": "默认运行时", "knowledge_base_id": 21},
    )

    assert response.status_code == 200
    assert response.json() == {"answer": "native-answer"}
    assert len(native.calls) == 1
    assert candidate_factory_calls == 0


def test_agent_chat_routes_explicit_langchain_candidate(
    db: Session,
) -> None:
    """显式 runtime=langchain 才切到已经开放的 Candidate。"""

    native = FakeNativeService()
    candidate = FakeLangChainService()
    selector = AgentRuntimeSelector(
        native_factory=lambda: native,
        langchain_factory=lambda: candidate,
        langchain_candidate_enabled=True,
    )
    client = _build_client(db=db, selector=selector)

    response = client.post(
        "/agent/chat?runtime=langchain",
        headers={"X-Request-ID": "langchain-http-001"},
        json={"message": "走 Candidate", "knowledge_base_id": 21},
    )

    assert response.status_code == 200
    assert response.json() == {"answer": "langchain-answer"}
    assert native.calls == []
    assert len(candidate.calls) == 1
    context = candidate.calls[0]["context"]
    assert context.user_id == 7
    assert context.knowledge_base_id == 21
    assert context.request_id == "langchain-http-001"
    assert context.agent_run_id is None


def test_agent_chat_rejects_disabled_langchain_candidate(
    db: Session,
) -> None:
    """配置未开放 Candidate 时返回安全 503，且绝不构造 LangChain Runtime。"""

    native = FakeNativeService()
    candidate_factory_calls = 0

    def candidate_factory() -> FakeLangChainService:
        nonlocal candidate_factory_calls
        candidate_factory_calls += 1
        return FakeLangChainService()

    selector = AgentRuntimeSelector(
        native_factory=lambda: native,
        langchain_factory=candidate_factory,
        langchain_candidate_enabled=False,
    )
    client = _build_client(db=db, selector=selector)

    response = client.post(
        "/agent/chat?runtime=langchain",
        json={"message": "不应执行", "knowledge_base_id": 21},
    )

    assert response.status_code == 503
    assert response.json() == {"detail": "Agent运行时暂不可用"}
    assert native.calls == []
    assert candidate_factory_calls == 0


def test_agent_chat_maps_langchain_runtime_policy_error_to_safe_503(
    db: Session,
) -> None:
    """Candidate 的预算类异常必须与 Native 一样脱敏映射为 503。"""

    candidate = FakeLangChainService()
    candidate.error = LangChainAgentTurnLimitError(
        "candidate-sensitive-runtime-detail"
    )
    selector = AgentRuntimeSelector(
        native_factory=FakeNativeService,
        langchain_factory=lambda: candidate,
        langchain_candidate_enabled=True,
    )
    client = _build_client(db=db, selector=selector)

    response = client.post(
        "/agent/chat?runtime=langchain",
        json={"message": "复杂任务", "knowledge_base_id": 21},
    )

    assert response.status_code == 503
    assert response.json() == {"detail": "Agent暂时无法完成请求"}
    assert "candidate-sensitive-runtime-detail" not in response.text


def test_agent_chat_rejects_unknown_runtime_at_http_validation_layer(
    db: Session,
) -> None:
    """未知 Runtime 不进入业务执行层，直接使用 FastAPI/Pydantic 422。"""

    native = FakeNativeService()
    selector = AgentRuntimeSelector(
        native_factory=lambda: native,
        langchain_factory=FakeLangChainService,
        langchain_candidate_enabled=True,
    )
    client = _build_client(db=db, selector=selector)

    response = client.post(
        "/agent/chat?runtime=unknown",
        json={"message": "非法 runtime", "knowledge_base_id": 21},
    )

    assert response.status_code == 422
    assert native.calls == []


def test_runtime_selector_rejects_disabled_candidate_without_factory_call() -> None:
    """Selector 本身也锁住 Candidate feature gate 的 lazy factory 语义。"""

    called = False

    def candidate_factory() -> FakeLangChainService:
        nonlocal called
        called = True
        return FakeLangChainService()

    selector = AgentRuntimeSelector(
        native_factory=FakeNativeService,
        langchain_factory=candidate_factory,
        langchain_candidate_enabled=False,
    )

    with pytest.raises(AgentRuntimeUnavailableError):
        selector.select(AgentRuntime.LANGCHAIN)

    assert called is False


def test_agent_stream_routes_explicit_langchain_candidate(
    db: Session,
) -> None:
    """SSE 与同步入口必须复用同一个 Candidate selector / feature gate。"""

    native = FakeNativeService()
    candidate = FakeLangChainService()
    selector = AgentRuntimeSelector(
        native_factory=lambda: native,
        langchain_factory=lambda: candidate,
        langchain_candidate_enabled=True,
    )
    client = _build_client(db=db, selector=selector)

    response = client.post(
        "/agent/chat/stream?runtime=langchain",
        headers={"X-Request-ID": "langchain-sse-001"},
        json={"message": "Candidate SSE", "knowledge_base_id": 21},
    )

    assert response.status_code == 200
    assert "event: status" in response.text
    assert 'data: {"content":"langchain-answer"}' in response.text
    assert "event: done" in response.text
    assert native.calls == []
    assert len(candidate.calls) == 1
    assert candidate.calls[0]["context"].request_id == "langchain-sse-001"


def test_agent_stream_rejects_disabled_langchain_before_sse_starts(
    db: Session,
) -> None:
    """Candidate gate 关闭时 SSE 必须在返回 200 前以普通 503 拒绝。"""

    candidate_factory_calls = 0

    def candidate_factory() -> FakeLangChainService:
        nonlocal candidate_factory_calls
        candidate_factory_calls += 1
        return FakeLangChainService()

    selector = AgentRuntimeSelector(
        native_factory=FakeNativeService,
        langchain_factory=candidate_factory,
        langchain_candidate_enabled=False,
    )
    client = _build_client(db=db, selector=selector)

    response = client.post(
        "/agent/chat/stream?runtime=langchain",
        json={"message": "不应建流", "knowledge_base_id": 21},
    )

    assert response.status_code == 503
    assert response.json() == {"detail": "Agent运行时暂不可用"}
    assert candidate_factory_calls == 0
