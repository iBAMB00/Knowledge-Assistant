from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

import app.api.agent as agent_api
from app.agent.checkpoint import AgentExecutionCheckpointPayload
from app.agent.context import ToolExecutionContext
from app.agent.hitl import AgentApprovalRequirement
from app.agent.model_response import LLMToolCall, LLMToolResponse
from app.agent.native_agent import NativeAgentResult
from app.agent.run_event import AgentMessageEvent, AgentRunEvent
from app.agent.state import AgentState, AgentThreadIdentity
from app.api.dependencies.agent import (
    get_agent_access_policy,
    get_agent_hitl_service,
    get_agent_run_control_service,
    get_agent_runtime_selector,
    get_agent_thread_status_service,
)
from app.api.dependencies.auth import get_current_user
from app.api.dependencies.conversation import get_conversation_service
from app.constants.agent_state_status import AgentStateStatus
from app.constants.conversation_mode import ConversationMode
from app.core.database import get_db
from app.middleware.request_context import RequestContextMiddleware
from app.models.database.user import User
from app.repositories.conversation_message_repository import ConversationMessageRepository
from app.repositories.conversation_repository import ConversationRepository
from app.repositories.document_repository import DocumentRepository
from app.repositories.knowledge_base_repository import KnowledgeBaseRepository
from app.schemas.conversation_contract import ConversationScope
from app.services.agent_checkpoint_service import AgentCheckpointService
from app.services.agent_hitl_service import AgentHITLService
from app.services.agent_run_control_service import AgentRunControlService
from app.services.agent_runtime_selector import AgentRuntimeSelector
from app.services.agent_thread_status_service import AgentThreadStatusService
from app.services.conversation_service import ConversationService
from app.services.knowledge_base_access_policy import KnowledgeBaseAccessPolicy
from app.services.knowledge_base_service import KnowledgeBaseService


class FakeAccessPolicy:
    def get_accessible_knowledge_base(self, **kwargs: Any) -> object:
        return object()


class FakeStatefulExecutionService:
    def __init__(self) -> None:
        self.resume_calls: list[tuple[str, ToolExecutionContext]] = []
        self.run_calls: list[tuple[str, ToolExecutionContext]] = []

    def run(
        self,
        *,
        db: Session,
        context: ToolExecutionContext,
        message: str,
    ) -> NativeAgentResult:
        self.run_calls.append((message, context))
        return NativeAgentResult(
            answer="stateful-http-answer",
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
        self.run_calls.append((message, context))
        yield AgentMessageEvent(
            content="stateful-http-answer",
            turns=1,
            tool_call_count=0,
        )

    def resume(
        self,
        *,
        db: Session,
        context: ToolExecutionContext,
        thread_id: str,
    ) -> NativeAgentResult:
        self.resume_calls.append((thread_id, context))
        return NativeAgentResult(
            answer="恢复后的回答",
            turns=2,
            tool_call_count=1,
        )

    def resume_events(
        self,
        *,
        db: Session,
        context: ToolExecutionContext,
        thread_id: str,
    ) -> Iterator[AgentRunEvent]:
        self.resume_calls.append((thread_id, context))
        yield AgentMessageEvent(
            content="恢复后的回答",
            turns=2,
            tool_call_count=1,
        )


def _create_user(db: Session, email: str) -> User:
    user = User(
        email=email,
        password_hash="hash",
        role="user",
        is_active=True,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def _domain_services() -> tuple[ConversationService, KnowledgeBaseService]:
    kb_repo = KnowledgeBaseRepository()
    doc_repo = DocumentRepository()
    access = KnowledgeBaseAccessPolicy(
        knowledge_base_repository=kb_repo,
        document_repository=doc_repo,
    )
    return (
        ConversationService(
            conversation_repository=ConversationRepository(),
            message_repository=ConversationMessageRepository(),
            access_policy=access,
        ),
        KnowledgeBaseService(
            knowledge_base_repository=kb_repo,
            document_repository=doc_repo,
            access_policy=access,
        ),
    )


def _payload(
    *,
    conversation_id: int,
    user_id: int,
    knowledge_base_id: int,
    status: AgentStateStatus,
    waiting: bool = False,
) -> AgentExecutionCheckpointPayload:
    tool_call = LLMToolCall(
        id="call-approval",
        name="sensitive_tool",
        arguments_json='{"secret":"not-public"}',
    )
    kwargs: dict[str, Any] = {}
    if waiting:
        kwargs = {
            "pending_tool_calls": (tool_call,),
            "last_model_response": LLMToolResponse(
                tool_calls=[tool_call]
            ),
            "pending_approvals": (
                AgentApprovalRequirement(
                    call_id=tool_call.id,
                    tool_name=tool_call.name,
                    reason="需要人工确认",
                ),
            ),
            "turn": 1,
            "tool_call_count": 1,
        }

    return AgentExecutionCheckpointPayload(
        agent_state=AgentState(
            conversation=ConversationScope(
                conversation_id=conversation_id,
                user_id=user_id,
                mode=ConversationMode.AGENT,
                knowledge_base_id=knowledge_base_id,
            ),
            thread=AgentThreadIdentity(
                thread_id=f"conversation:{conversation_id}",
                conversation_id=conversation_id,
            ),
            status=status,
            task="执行 Stateful 任务",
        ),
        **kwargs,
    )


def _client(
    *,
    db: Session,
    user: User,
    conversation_service: ConversationService,
    checkpoint_service: AgentCheckpointService,
    runtime_service: FakeStatefulExecutionService | None = None,
) -> TestClient:
    app = FastAPI()
    app.add_middleware(RequestContextMiddleware)
    app.include_router(agent_api.router)

    def override_db():
        yield db

    app.dependency_overrides[get_db] = override_db
    app.dependency_overrides[get_current_user] = lambda: user
    app.dependency_overrides[get_agent_access_policy] = FakeAccessPolicy
    app.dependency_overrides[get_conversation_service] = lambda: conversation_service
    app.dependency_overrides[get_agent_thread_status_service] = lambda: (
        AgentThreadStatusService(checkpoint_service)
    )
    app.dependency_overrides[get_agent_hitl_service] = lambda: AgentHITLService(
        checkpoint_service
    )
    app.dependency_overrides[get_agent_run_control_service] = lambda: (
        AgentRunControlService(checkpoint_service)
    )

    if runtime_service is not None:
        selector = AgentRuntimeSelector(
            native_factory=lambda: object(),
            langchain_factory=lambda: object(),
            langchain_candidate_enabled=False,
            langgraph_factory=lambda: runtime_service,
            langgraph_candidate_enabled=True,
        )
        app.dependency_overrides[get_agent_runtime_selector] = lambda: selector

    return TestClient(app)


def test_agent_chat_routes_langgraph_with_trusted_conversation_context(
    db: Session,
) -> None:
    conversation_service, kb_service = _domain_services()
    user = _create_user(db, "stateful-chat-api@example.com")
    kb = kb_service.create(db, user, "Stateful Chat API KB")
    conversation = conversation_service.create(
        db=db,
        user=user,
        mode=ConversationMode.AGENT,
        knowledge_base_id=kb.id,
    )
    checkpoint_service = AgentCheckpointService()
    runtime_service = FakeStatefulExecutionService()

    with _client(
        db=db,
        user=user,
        conversation_service=conversation_service,
        checkpoint_service=checkpoint_service,
        runtime_service=runtime_service,
    ) as client:
        response = client.post(
            "/agent/chat?runtime=langgraph",
            json={
                "message": "执行 Stateful HTTP 任务",
                "knowledge_base_id": kb.id,
                "conversation_id": conversation.id,
            },
        )

    assert response.status_code == 200
    assert response.json() == {"answer": "stateful-http-answer"}
    assert len(runtime_service.run_calls) == 1
    message, context = runtime_service.run_calls[0]
    assert message == "执行 Stateful HTTP 任务"
    assert context.user_id == user.id
    assert context.knowledge_base_id == kb.id
    assert context.conversation_id == conversation.id

    messages = conversation_service.list_messages(
        db=db,
        user_id=user.id,
        conversation_id=conversation.id,
    )
    assert [(item.role, item.content) for item in messages] == [
        ("user", "执行 Stateful HTTP 任务"),
        ("assistant", "stateful-http-answer"),
    ]


def test_agent_chat_rejects_langgraph_without_conversation(
    db: Session,
) -> None:
    conversation_service, kb_service = _domain_services()
    user = _create_user(db, "stateful-chat-no-conversation@example.com")
    kb = kb_service.create(db, user, "Stateful No Conversation KB")
    checkpoint_service = AgentCheckpointService()
    runtime_service = FakeStatefulExecutionService()

    with _client(
        db=db,
        user=user,
        conversation_service=conversation_service,
        checkpoint_service=checkpoint_service,
        runtime_service=runtime_service,
    ) as client:
        response = client.post(
            "/agent/chat?runtime=langgraph",
            json={
                "message": "缺少 conversation",
                "knowledge_base_id": kb.id,
            },
        )

    assert response.status_code == 400
    assert response.json() == {
        "detail": "langgraph runtime requires conversation_id"
    }
    assert runtime_service.run_calls == []


def test_thread_status_and_cancel_api_are_safe_and_durable(db: Session) -> None:
    conversation_service, kb_service = _domain_services()
    user = _create_user(db, "thread-control@example.com")
    kb = kb_service.create(db, user, "Thread Control KB")
    conversation = conversation_service.create(
        db=db,
        user=user,
        mode=ConversationMode.AGENT,
        knowledge_base_id=kb.id,
    )
    checkpoint_service = AgentCheckpointService()
    checkpoint_service.save_checkpoint(
        db,
        _payload(
            conversation_id=conversation.id,
            user_id=user.id,
            knowledge_base_id=kb.id,
            status=AgentStateStatus.RUNNING,
        ),
    )
    thread_id = f"conversation:{conversation.id}"

    with _client(
        db=db,
        user=user,
        conversation_service=conversation_service,
        checkpoint_service=checkpoint_service,
    ) as client:
        status_response = client.get(
            f"/agent/threads/{thread_id}",
            params={"knowledge_base_id": kb.id},
        )
        cancel_response = client.post(
            f"/agent/threads/{thread_id}/cancel",
            json={"knowledge_base_id": kb.id},
        )

    assert status_response.status_code == 200
    payload = status_response.json()
    assert payload["status"] == "running"
    assert payload["can_resume"] is True
    assert payload["can_cancel"] is True
    rendered = status_response.text
    assert "arguments_json" not in rendered
    assert "not-public" not in rendered

    assert cancel_response.status_code == 200
    assert cancel_response.json()["status"] == "cancelled"
    assert cancel_response.json()["can_resume"] is False
    assert cancel_response.json()["can_cancel"] is False


def test_approve_api_exposes_only_safe_metadata_and_enables_resume(db: Session) -> None:
    conversation_service, kb_service = _domain_services()
    user = _create_user(db, "thread-approval@example.com")
    kb = kb_service.create(db, user, "Approval KB")
    conversation = conversation_service.create(
        db=db,
        user=user,
        mode=ConversationMode.AGENT,
        knowledge_base_id=kb.id,
    )
    checkpoint_service = AgentCheckpointService()
    checkpoint_service.save_checkpoint(
        db,
        _payload(
            conversation_id=conversation.id,
            user_id=user.id,
            knowledge_base_id=kb.id,
            status=AgentStateStatus.WAITING,
            waiting=True,
        ),
    )
    thread_id = f"conversation:{conversation.id}"

    with _client(
        db=db,
        user=user,
        conversation_service=conversation_service,
        checkpoint_service=checkpoint_service,
    ) as client:
        response = client.post(
            f"/agent/threads/{thread_id}/approve",
            json={"knowledge_base_id": kb.id, "call_ids": []},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "waiting"
    assert payload["can_resume"] is True
    assert payload["can_approve"] is False
    assert payload["pending_approvals"] == [
        {
            "call_id": "call-approval",
            "tool_name": "sensitive_tool",
            "reason": "需要人工确认",
        }
    ]
    assert "not-public" not in response.text


def test_resume_api_uses_stateful_runtime_and_persists_final_assistant_message(
    db: Session,
) -> None:
    conversation_service, kb_service = _domain_services()
    user = _create_user(db, "thread-resume@example.com")
    kb = kb_service.create(db, user, "Resume KB")
    conversation = conversation_service.create(
        db=db,
        user=user,
        mode=ConversationMode.AGENT,
        knowledge_base_id=kb.id,
    )
    checkpoint_service = AgentCheckpointService()
    checkpoint_service.save_checkpoint(
        db,
        _payload(
            conversation_id=conversation.id,
            user_id=user.id,
            knowledge_base_id=kb.id,
            status=AgentStateStatus.RUNNING,
        ),
    )
    runtime_service = FakeStatefulExecutionService()
    thread_id = f"conversation:{conversation.id}"

    with _client(
        db=db,
        user=user,
        conversation_service=conversation_service,
        checkpoint_service=checkpoint_service,
        runtime_service=runtime_service,
    ) as client:
        response = client.post(
            f"/agent/threads/{thread_id}/resume",
            json={"knowledge_base_id": kb.id},
        )

    assert response.status_code == 200
    assert response.json() == {"answer": "恢复后的回答"}
    assert len(runtime_service.resume_calls) == 1
    _, context = runtime_service.resume_calls[0]
    assert context.user_id == user.id
    assert context.knowledge_base_id == kb.id
    assert context.conversation_id == conversation.id

    messages = conversation_service.list_messages(
        db=db,
        user_id=user.id,
        conversation_id=conversation.id,
    )
    assert [(item.role, item.content) for item in messages] == [
        ("assistant", "恢复后的回答")
    ]
