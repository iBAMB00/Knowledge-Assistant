from collections.abc import Iterator
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

import app.api.agent as agent_api
import app.api.knowledge_chat as knowledge_chat_api
from app.agent.context import ToolExecutionContext
from app.agent.native_agent import NativeAgentResult
from app.agent.run_event import AgentMessageEvent, AgentRunEvent, AgentStatusEvent
from app.api.dependencies.agent import get_agent_access_policy, get_agent_runtime_selector
from app.api.dependencies.auth import get_current_user
from app.api.dependencies.conversation import get_conversation_service
from app.constants.agent_runtime import AgentRuntime
from app.constants.conversation_mode import ConversationMode
from app.core.database import get_db
from app.middleware.request_context import RequestContextMiddleware
from app.models.database.knowledge_base import KnowledgeBase
from app.models.database.user import User
from app.repositories.conversation_message_repository import ConversationMessageRepository
from app.repositories.conversation_repository import ConversationRepository
from app.repositories.document_repository import DocumentRepository
from app.repositories.knowledge_base_repository import KnowledgeBaseRepository
from app.schemas.knowledge_chat_response import KnowledgeChatResponse
from app.services.conversation_service import ConversationService
from app.services.knowledge_base_access_policy import KnowledgeBaseAccessPolicy


class FakeAccessPolicy:
    def get_accessible_knowledge_base(self, **kwargs: Any) -> object:
        return object()

    def ensure_document_in_knowledge_base(self, **kwargs: Any) -> None:
        return None


class FakeKnowledgeChatService:
    def __init__(self) -> None:
        self.answer = "RAG 持久化回答"
        self.stream_parts = ["RAG ", "流式回答"]
        self.stream_error: Exception | None = None

    def chat(self, **kwargs: Any) -> KnowledgeChatResponse:
        question = kwargs["question"].strip()
        if not question:
            raise ValueError("question cannot be empty")
        return KnowledgeChatResponse(answer=self.answer, sources=[])

    def prepare(self, **kwargs: Any):
        question = kwargs["question"].strip()
        if not question:
            raise ValueError("question cannot be empty")
        from app.services.knowledge_chat_service import KnowledgeChatPreparation

        return KnowledgeChatPreparation(prompt="prompt", sources=[])

    def stream_chat(self, _preparation: Any) -> Iterator[str]:
        yield from self.stream_parts
        if self.stream_error is not None:
            raise self.stream_error


class FakeAgentRunner:
    def run(
        self,
        *,
        db: Session,
        context: ToolExecutionContext,
        message: str,
    ) -> NativeAgentResult:
        return NativeAgentResult(
            answer="Agent 持久化回答",
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
        yield AgentStatusEvent(turn=1)
        yield AgentMessageEvent(
            content="Agent 流式最终回答",
            turns=1,
            tool_call_count=0,
        )


class FakeRuntimeSelector:
    def __init__(self, runner: FakeAgentRunner) -> None:
        self.runner = runner

    def select(self, runtime: AgentRuntime) -> FakeAgentRunner:
        return self.runner


def _create_user_and_kb(db: Session, suffix: str) -> tuple[User, KnowledgeBase]:
    user = User(
        email=f"chat-persistence-{suffix}@example.com",
        password_hash="hash",
        role="user",
        is_active=True,
    )
    db.add(user)
    db.flush()

    kb = KnowledgeBase(
        owner_id=user.id,
        name=f"Chat Persistence {suffix}",
    )
    db.add(kb)
    db.commit()
    db.refresh(user)
    db.refresh(kb)
    return user, kb


def _conversation_service() -> ConversationService:
    return ConversationService(
        conversation_repository=ConversationRepository(),
        message_repository=ConversationMessageRepository(),
        access_policy=KnowledgeBaseAccessPolicy(
            knowledge_base_repository=KnowledgeBaseRepository(),
            document_repository=DocumentRepository(),
        ),
    )


def _messages(
    service: ConversationService,
    db: Session,
    user_id: int,
    conversation_id: int,
) -> list[tuple[str, str]]:
    return [
        (item.role, item.content)
        for item in service.list_messages(
            db=db,
            user_id=user_id,
            conversation_id=conversation_id,
        )
    ]


def test_rag_sync_and_stream_persist_bound_conversation(
    db: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user, kb = _create_user_and_kb(db, "rag")
    conversation_service = _conversation_service()
    conversation = conversation_service.create(
        db=db,
        user=user,
        mode=ConversationMode.RAG,
        knowledge_base_id=kb.id,
    )

    fake_chat = FakeKnowledgeChatService()
    monkeypatch.setattr(knowledge_chat_api, "knowledge_chat_service", fake_chat)
    monkeypatch.setattr(
        knowledge_chat_api,
        "knowledge_base_access_policy",
        FakeAccessPolicy(),
    )

    app = FastAPI()
    app.include_router(knowledge_chat_api.router)
    def override_get_db():
        yield db

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_current_user] = lambda: user
    app.dependency_overrides[get_conversation_service] = lambda: conversation_service
    client = TestClient(app)

    sync_response = client.post(
        "/knowledge/chat",
        json={
            "question": "  同步问题  ",
            "knowledge_base_id": kb.id,
            "conversation_id": conversation.id,
        },
    )
    assert sync_response.status_code == 200

    stream_response = client.post(
        "/knowledge/chat/stream",
        json={
            "question": "流式问题",
            "knowledge_base_id": kb.id,
            "conversation_id": conversation.id,
        },
    )
    assert stream_response.status_code == 200
    assert "event: done" in stream_response.text

    assert _messages(
        conversation_service,
        db,
        user.id,
        conversation.id,
    ) == [
        ("user", "同步问题"),
        ("assistant", "RAG 持久化回答"),
        ("user", "流式问题"),
        ("assistant", "RAG 流式回答"),
    ]


def test_agent_sync_and_stream_persist_bound_conversation(db: Session) -> None:
    user, kb = _create_user_and_kb(db, "agent")
    conversation_service = _conversation_service()
    conversation = conversation_service.create(
        db=db,
        user=user,
        mode=ConversationMode.AGENT,
        knowledge_base_id=kb.id,
    )
    runner = FakeAgentRunner()

    app = FastAPI()
    app.add_middleware(RequestContextMiddleware)
    app.include_router(agent_api.router)
    def override_get_db():
        yield db

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_current_user] = lambda: user
    app.dependency_overrides[get_agent_access_policy] = lambda: FakeAccessPolicy()
    app.dependency_overrides[get_agent_runtime_selector] = lambda: FakeRuntimeSelector(runner)
    app.dependency_overrides[get_conversation_service] = lambda: conversation_service
    client = TestClient(app)

    sync_response = client.post(
        "/agent/chat",
        json={
            "message": "  Agent 同步问题  ",
            "knowledge_base_id": kb.id,
            "conversation_id": conversation.id,
        },
    )
    assert sync_response.status_code == 200

    stream_response = client.post(
        "/agent/chat/stream",
        json={
            "message": "Agent 流式问题",
            "knowledge_base_id": kb.id,
            "conversation_id": conversation.id,
        },
    )
    assert stream_response.status_code == 200
    assert "event: done" in stream_response.text

    assert _messages(
        conversation_service,
        db,
        user.id,
        conversation.id,
    ) == [
        ("user", "Agent 同步问题"),
        ("assistant", "Agent 持久化回答"),
        ("user", "Agent 流式问题"),
        ("assistant", "Agent 流式最终回答"),
    ]



def test_rag_stream_failure_does_not_persist_partial_assistant(
    db: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user, kb = _create_user_and_kb(db, "stream-failure")
    conversation_service = _conversation_service()
    conversation = conversation_service.create(
        db=db,
        user=user,
        mode=ConversationMode.RAG,
        knowledge_base_id=kb.id,
    )

    fake_chat = FakeKnowledgeChatService()
    fake_chat.stream_parts = ["半截回答"]
    fake_chat.stream_error = RuntimeError("stream failed")
    monkeypatch.setattr(knowledge_chat_api, "knowledge_chat_service", fake_chat)
    monkeypatch.setattr(
        knowledge_chat_api,
        "knowledge_base_access_policy",
        FakeAccessPolicy(),
    )

    app = FastAPI()
    app.include_router(knowledge_chat_api.router)

    def override_get_db():
        yield db

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_current_user] = lambda: user
    app.dependency_overrides[get_conversation_service] = lambda: conversation_service
    client = TestClient(app)

    response = client.post(
        "/knowledge/chat/stream",
        json={
            "question": "失败流式问题",
            "knowledge_base_id": kb.id,
            "conversation_id": conversation.id,
        },
    )

    assert response.status_code == 200
    assert "event: error" in response.text
    assert "event: done" not in response.text
    assert _messages(
        conversation_service,
        db,
        user.id,
        conversation.id,
    ) == [("user", "失败流式问题")]

def test_chat_conversation_scope_conflict_is_rejected_before_persistence(
    db: Session,
) -> None:
    user, kb = _create_user_and_kb(db, "scope")
    conversation_service = _conversation_service()
    rag_conversation = conversation_service.create(
        db=db,
        user=user,
        mode=ConversationMode.RAG,
        knowledge_base_id=kb.id,
    )

    app = FastAPI()
    app.add_middleware(RequestContextMiddleware)
    app.include_router(agent_api.router)
    def override_get_db():
        yield db

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_current_user] = lambda: user
    app.dependency_overrides[get_agent_access_policy] = lambda: FakeAccessPolicy()
    app.dependency_overrides[get_agent_runtime_selector] = lambda: FakeRuntimeSelector(
        FakeAgentRunner()
    )
    app.dependency_overrides[get_conversation_service] = lambda: conversation_service
    client = TestClient(app)

    response = client.post(
        "/agent/chat",
        json={
            "message": "不能写入 RAG Conversation",
            "knowledge_base_id": kb.id,
            "conversation_id": rag_conversation.id,
        },
    )
    assert response.status_code == 409
    assert _messages(
        conversation_service,
        db,
        user.id,
        rag_conversation.id,
    ) == []


def test_chat_without_conversation_id_remains_stateless_compatible(
    db: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user, kb = _create_user_and_kb(db, "stateless")
    conversation_service = _conversation_service()
    fake_chat = FakeKnowledgeChatService()
    monkeypatch.setattr(knowledge_chat_api, "knowledge_chat_service", fake_chat)
    monkeypatch.setattr(
        knowledge_chat_api,
        "knowledge_base_access_policy",
        FakeAccessPolicy(),
    )

    app = FastAPI()
    app.include_router(knowledge_chat_api.router)
    def override_get_db():
        yield db

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_current_user] = lambda: user
    app.dependency_overrides[get_conversation_service] = lambda: conversation_service
    client = TestClient(app)

    response = client.post(
        "/knowledge/chat",
        json={
            "question": "旧客户端请求",
            "knowledge_base_id": kb.id,
        },
    )
    assert response.status_code == 200
    assert conversation_service.list_owned(db=db, user_id=user.id) == []
