import pytest
from sqlalchemy.orm import Session

from app.agent.checkpoint import AgentExecutionCheckpointPayload
from app.agent.hitl import (
    AgentApprovalRequirement,
    AgentApprovalSelection,
    AgentApprovalStateError,
)
from app.agent.model_response import LLMToolCall, LLMToolResponse
from app.agent.state import AgentState, AgentThreadIdentity
from app.constants.agent_state_status import AgentStateStatus
from app.constants.conversation_message_role import ConversationMessageRole
from app.constants.conversation_mode import ConversationMode
from app.models.database.user import User
from app.repositories.conversation_message_repository import ConversationMessageRepository
from app.repositories.conversation_repository import ConversationRepository
from app.repositories.document_repository import DocumentRepository
from app.repositories.knowledge_base_repository import KnowledgeBaseRepository
from app.schemas.conversation_contract import ConversationMessagePayload, ConversationScope
from app.services.agent_checkpoint_service import AgentCheckpointService
from app.services.agent_hitl_service import AgentHITLService
from app.services.conversation_service import ConversationService
from app.services.knowledge_base_access_policy import KnowledgeBaseAccessPolicy
from app.services.knowledge_base_service import KnowledgeBaseService


def _create_user(db: Session, email: str) -> User:
    user = User(
        email=email,
        password_hash="test-password-hash",
        role="user",
        is_active=True,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def _build_services() -> tuple[
    ConversationService,
    KnowledgeBaseService,
]:
    kb_repository = KnowledgeBaseRepository()
    document_repository = DocumentRepository()
    access_policy = KnowledgeBaseAccessPolicy(
        knowledge_base_repository=kb_repository,
        document_repository=document_repository,
    )
    return (
        ConversationService(
            conversation_repository=ConversationRepository(),
            message_repository=ConversationMessageRepository(),
            access_policy=access_policy,
        ),
        KnowledgeBaseService(
            knowledge_base_repository=kb_repository,
            document_repository=document_repository,
            access_policy=access_policy,
        ),
    )


def _waiting_payload(
    *,
    conversation_id: int,
    user_id: int,
    knowledge_base_id: int,
) -> AgentExecutionCheckpointPayload:
    tool_call = LLMToolCall(
        id="approval-call-1",
        name="echo",
        arguments_json='{"text":"sensitive"}',
    )
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
            status=AgentStateStatus.WAITING,
            task="执行审批 Tool",
            messages=(
                ConversationMessagePayload(
                    role=ConversationMessageRole.USER,
                    content="执行审批 Tool",
                ),
            ),
        ),
        pending_tool_calls=(tool_call,),
        last_model_response=LLMToolResponse(tool_calls=[tool_call]),
        pending_approvals=(
            AgentApprovalRequirement(
                call_id=tool_call.id,
                tool_name=tool_call.name,
                reason="需要人工确认",
            ),
        ),
        turn=1,
        tool_call_count=1,
        seen_tool_call_signatures=('echo:{"text":"sensitive"}',),
    )


def _create_waiting_thread(
    db: Session,
    *,
    email: str,
) -> tuple[User, int, int, str, AgentCheckpointService]:
    conversation_service, kb_service = _build_services()
    user = _create_user(db, email)
    kb = kb_service.create(db, user, "HITL KB")
    conversation = conversation_service.create(
        db=db,
        user=user,
        mode=ConversationMode.AGENT,
        knowledge_base_id=kb.id,
    )
    thread_id = f"conversation:{conversation.id}"
    checkpoint_service = AgentCheckpointService()
    checkpoint_service.save_checkpoint(
        db,
        _waiting_payload(
            conversation_id=conversation.id,
            user_id=user.id,
            knowledge_base_id=kb.id,
        ),
    )
    return user, kb.id, conversation.id, thread_id, checkpoint_service


def test_hitl_service_approve_keeps_thread_waiting_until_resume(
    db: Session,
) -> None:
    user, kb_id, _, thread_id, checkpoint_service = _create_waiting_thread(
        db,
        email="hitl-approve@example.com",
    )
    service = AgentHITLService(checkpoint_service)

    approved = service.approve(
        db,
        thread_id=thread_id,
        user_id=user.id,
        knowledge_base_id=kb_id,
    )

    assert approved.agent_state.status is AgentStateStatus.WAITING
    assert approved.approved_call_ids == ("approval-call-1",)
    loaded = service.load_approved_checkpoint(
        db,
        thread_id=thread_id,
        user_id=user.id,
        knowledge_base_id=kb_id,
    )
    assert loaded == approved

    checkpoints = checkpoint_service.list_checkpoints(
        db,
        thread_id=thread_id,
    )
    assert len(checkpoints) == 2
    assert checkpoints[-1].payload["approved_call_ids"] == [
        "approval-call-1"
    ]


def test_hitl_service_reject_transitions_thread_to_cancelled(
    db: Session,
) -> None:
    user, kb_id, _, thread_id, checkpoint_service = _create_waiting_thread(
        db,
        email="hitl-reject@example.com",
    )
    service = AgentHITLService(checkpoint_service)

    rejected = service.reject(
        db,
        thread_id=thread_id,
        user_id=user.id,
        knowledge_base_id=kb_id,
    )

    assert rejected.agent_state.status is AgentStateStatus.CANCELLED
    assert rejected.agent_state.last_error_code == "approval_rejected"
    assert rejected.rejected_call_ids == ("approval-call-1",)

    with pytest.raises(
        AgentApprovalStateError,
        match="not waiting",
    ):
        service.load_approved_checkpoint(
            db,
            thread_id=thread_id,
            user_id=user.id,
            knowledge_base_id=kb_id,
        )


def test_hitl_service_rejects_unknown_selection(
    db: Session,
) -> None:
    user, kb_id, _, thread_id, checkpoint_service = _create_waiting_thread(
        db,
        email="hitl-selection@example.com",
    )
    service = AgentHITLService(checkpoint_service)

    with pytest.raises(
        AgentApprovalStateError,
        match="unknown call_id",
    ):
        service.approve(
            db,
            thread_id=thread_id,
            user_id=user.id,
            knowledge_base_id=kb_id,
            selection=AgentApprovalSelection(
                call_ids=("unknown-call",),
            ),
        )


def test_checkpoint_schema_1_0_remains_readable() -> None:
    tool_call = LLMToolCall(
        id="legacy-call",
        name="echo",
        arguments_json='{"text":"legacy"}',
    )
    payload = AgentExecutionCheckpointPayload.model_validate(
        {
            "checkpoint_schema_version": "1.0",
            "agent_state": {
                "state_schema_version": "1.0",
                "conversation": {
                    "conversation_id": 1,
                    "user_id": 1,
                    "mode": "agent",
                    "knowledge_base_id": 1,
                },
                "thread": {
                    "thread_id": "conversation:1",
                    "conversation_id": 1,
                },
                "status": "running",
                "messages": [],
                "task": "legacy task",
                "retry_count": 0,
                "last_error_code": None,
                "agent_run_id": None,
            },
            "history": [],
            "pending_tool_calls": [tool_call.model_dump(mode="json")],
            "last_model_response": LLMToolResponse(
                tool_calls=[tool_call]
            ).model_dump(mode="json"),
            "tool_observations": [],
            "final_answer": None,
            "turn": 1,
            "tool_call_count": 1,
            "seen_tool_call_signatures": [],
        }
    )

    assert payload.checkpoint_schema_version == "1.0"
    assert payload.pending_approvals == ()
    assert payload.approved_call_ids == ()
    assert payload.rejected_call_ids == ()
