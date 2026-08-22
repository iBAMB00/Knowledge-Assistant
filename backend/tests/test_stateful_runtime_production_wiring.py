from __future__ import annotations

from collections.abc import Iterator

import pytest
from sqlalchemy.orm import Session

from app.agent.context import ToolExecutionContext
from app.agent.hitl import AgentApprovalRequirement, AgentInterruptRequired
from app.agent.run_event import (
    AgentMessageEvent,
    AgentRunEvent,
    AgentStatusEvent,
    AgentToolCallEvent,
)
from app.agent.version_snapshot import AgentRuntimeVersionSnapshot
from app.constants.agent_run_status import AgentRunStatus
from app.constants.agent_runtime import AgentRuntime
from app.constants.conversation_mode import ConversationMode
from app.constants.user_role import UserRole
from app.models.database.agent_run import AgentRun
from app.models.database.agent_tool_call import AgentToolCall
from app.models.database.user import User
from app.repositories.agent_run_repository import AgentRunRepository
from app.repositories.agent_tool_call_repository import AgentToolCallRepository
from app.repositories.conversation_message_repository import ConversationMessageRepository
from app.repositories.conversation_repository import ConversationRepository
from app.repositories.document_repository import DocumentRepository
from app.repositories.knowledge_base_repository import KnowledgeBaseRepository
from app.services.agent_checkpoint_service import AgentCheckpointService
from app.services.agent_hitl_service import AgentHITLService
from app.services.agent_recovery_service import AgentRecoveryService
from app.services.agent_runtime_selector import (
    AgentRuntimeSelector,
    AgentRuntimeUnavailableError,
)
from app.services.conversation_service import ConversationService
from app.services.knowledge_base_access_policy import KnowledgeBaseAccessPolicy
from app.services.knowledge_base_service import KnowledgeBaseService
from app.services.langgraph_agent_execution_service import (
    LangGraphAgentExecutionService,
)


class FakeStatefulRunner:
    tool_contracts = ()

    def __init__(self, *, interrupt: bool = False) -> None:
        self.interrupt = interrupt
        self.received_state = None

    def run_events(
        self,
        *,
        db: Session,
        context: ToolExecutionContext,
        message: str,
        state,
        observer=None,
    ) -> Iterator[AgentRunEvent]:
        self.received_state = state
        yield AgentStatusEvent(turn=1)
        if self.interrupt:
            yield AgentToolCallEvent(
                turn=1,
                call_id="call-1",
                tool_name="sensitive_tool",
            )
            raise AgentInterruptRequired(
                thread_id=state.thread.thread_id,
                approvals=(
                    AgentApprovalRequirement(
                        call_id="call-1",
                        tool_name="sensitive_tool",
                        reason="needs approval",
                    ),
                ),
            )
        yield AgentMessageEvent(
            content="stateful-answer",
            turns=1,
            tool_call_count=0,
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


def _services() -> tuple[ConversationService, KnowledgeBaseService]:
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


def _execution_service(runner: FakeStatefulRunner) -> LangGraphAgentExecutionService:
    checkpoint_service = AgentCheckpointService()
    return LangGraphAgentExecutionService(
        agent_runner=runner,  # type: ignore[arg-type]
        agent_run_repository=AgentRunRepository(),
        tool_call_repository=AgentToolCallRepository(),
        model_provider="test",
        model_name="test-model",
        version_snapshot=AgentRuntimeVersionSnapshot(
            agent_version="langgraph-v1:test",
            prompt_version="test-prompt",
            toolset_version="toolset-v2:test",
            retrieval_config_version="retrieval-v1:test",
        ),
        checkpoint_service=checkpoint_service,
        recovery_service=AgentRecoveryService(checkpoint_service),
        hitl_service=AgentHITLService(checkpoint_service),
    )


def test_langgraph_selector_is_lazy_and_feature_gated() -> None:
    called = 0

    def factory():
        nonlocal called
        called += 1
        return object()

    disabled = AgentRuntimeSelector(
        native_factory=lambda: object(),
        langchain_factory=lambda: object(),
        langchain_candidate_enabled=False,
        langgraph_factory=factory,
        langgraph_candidate_enabled=False,
    )
    with pytest.raises(AgentRuntimeUnavailableError):
        disabled.select(AgentRuntime.LANGGRAPH)
    assert called == 0

    enabled = AgentRuntimeSelector(
        native_factory=lambda: object(),
        langchain_factory=lambda: object(),
        langchain_candidate_enabled=False,
        langgraph_factory=factory,
        langgraph_candidate_enabled=True,
    )
    enabled.select(AgentRuntime.LANGGRAPH)
    assert called == 1


def test_langgraph_execution_binds_conversation_and_persists_agent_run(
    db: Session,
) -> None:
    conversation_service, kb_service = _services()
    user = _create_user(db, "stateful-prod@example.com")
    kb = kb_service.create(db, user, "Stateful Prod KB")
    conversation = conversation_service.create(
        db=db,
        user=user,
        mode=ConversationMode.AGENT,
        knowledge_base_id=kb.id,
    )
    runner = FakeStatefulRunner()
    service = _execution_service(runner)

    result = service.run(
        db=db,
        context=ToolExecutionContext(
            user_id=user.id,
            role=UserRole.USER,
            knowledge_base_id=kb.id,
            request_id="stateful-prod-001",
            conversation_id=conversation.id,
        ),
        message="查询发布规则",
    )

    assert result.answer == "stateful-answer"
    assert runner.received_state is not None
    assert runner.received_state.conversation.conversation_id == conversation.id
    assert runner.received_state.thread.thread_id == f"conversation:{conversation.id}"

    agent_run = db.query(AgentRun).one()
    assert agent_run.status == AgentRunStatus.SUCCEEDED.value
    assert agent_run.agent_version == "langgraph-v1:test"


def test_waiting_state_marks_attempt_interrupted_and_closes_open_tool_call(
    db: Session,
) -> None:
    conversation_service, kb_service = _services()
    user = _create_user(db, "stateful-waiting@example.com")
    kb = kb_service.create(db, user, "Stateful Waiting KB")
    conversation = conversation_service.create(
        db=db,
        user=user,
        mode=ConversationMode.AGENT,
        knowledge_base_id=kb.id,
    )
    service = _execution_service(FakeStatefulRunner(interrupt=True))
    context = ToolExecutionContext(
        user_id=user.id,
        role=UserRole.USER,
        knowledge_base_id=kb.id,
        request_id="stateful-wait-001",
        conversation_id=conversation.id,
    )

    with pytest.raises(AgentInterruptRequired):
        list(
            service.run_events(
                db=db,
                context=context,
                message="执行敏感操作",
            )
        )

    agent_run = db.query(AgentRun).one()
    assert agent_run.status == AgentRunStatus.INTERRUPTED.value
    assert agent_run.error_type == "approval_required"
    tool_call = db.query(AgentToolCall).one()
    assert tool_call.status == "failed"
    assert tool_call.error_type == "approval_required"
