import pytest
from sqlalchemy.orm import Session

from app.agent.checkpoint import AgentExecutionCheckpointPayload
from app.agent.state import AgentState, AgentThreadIdentity
from app.constants.agent_state_status import AgentStateStatus
from app.constants.conversation_message_role import ConversationMessageRole
from app.constants.conversation_mode import ConversationMode
from app.models.database.agent_checkpoint import AgentCheckpoint
from app.models.database.agent_thread import AgentThread
from app.models.database.user import User
from app.repositories.conversation_message_repository import ConversationMessageRepository
from app.repositories.conversation_repository import ConversationRepository
from app.repositories.document_repository import DocumentRepository
from app.repositories.knowledge_base_repository import KnowledgeBaseRepository
from app.schemas.conversation_contract import ConversationMessagePayload, ConversationScope
from app.services.agent_checkpoint_service import (
    AgentCheckpointScopeError,
    AgentCheckpointService,
)
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


def _build_conversation_services() -> tuple[ConversationService, KnowledgeBaseService]:
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


def _payload(
    *,
    conversation_id: int,
    user_id: int,
    knowledge_base_id: int,
    status: AgentStateStatus,
    turn: int,
) -> AgentExecutionCheckpointPayload:
    messages = (
        ConversationMessagePayload(
            role=ConversationMessageRole.USER,
            content="查询发布规则",
        ),
    )
    final_answer = None
    if status is AgentStateStatus.SUCCEEDED:
        messages = (
            *messages,
            ConversationMessagePayload(
                role=ConversationMessageRole.ASSISTANT,
                content="发布规则已找到",
            ),
        )
        final_answer = "发布规则已找到"

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
            task="查询发布规则",
            messages=messages,
        ),
        final_answer=final_answer,
        turn=turn,
    )


def test_checkpoint_service_appends_sequence_and_restores_latest(db: Session) -> None:
    conversation_service, kb_service = _build_conversation_services()
    user = _create_user(db, "checkpoint@example.com")
    kb = kb_service.create(db, user, "Checkpoint KB")
    conversation = conversation_service.create(
        db=db,
        user=user,
        mode=ConversationMode.AGENT,
        knowledge_base_id=kb.id,
    )
    service = AgentCheckpointService()

    first_payload = _payload(
        conversation_id=conversation.id,
        user_id=user.id,
        knowledge_base_id=kb.id,
        status=AgentStateStatus.RUNNING,
        turn=0,
    )
    second_payload = _payload(
        conversation_id=conversation.id,
        user_id=user.id,
        knowledge_base_id=kb.id,
        status=AgentStateStatus.SUCCEEDED,
        turn=1,
    )

    first = service.save_checkpoint(db, first_payload)
    second = service.save_checkpoint(db, second_payload)
    restored = service.load_latest(
        db,
        thread_id=f"conversation:{conversation.id}",
    )

    assert first.sequence == 1
    assert second.sequence == 2
    assert restored == second_payload

    thread = db.query(AgentThread).filter_by(
        conversation_id=conversation.id
    ).one()
    assert thread.status == AgentStateStatus.SUCCEEDED.value
    assert [item.sequence for item in service.list_checkpoints(
        db,
        thread_id=thread.thread_id,
    )] == [1, 2]


def test_checkpoint_scope_rejects_state_claiming_another_kb(db: Session) -> None:
    conversation_service, kb_service = _build_conversation_services()
    user = _create_user(db, "checkpoint-scope@example.com")
    kb = kb_service.create(db, user, "Checkpoint Scope KB")
    other_kb = kb_service.create(db, user, "Other KB")
    conversation = conversation_service.create(
        db=db,
        user=user,
        mode=ConversationMode.AGENT,
        knowledge_base_id=kb.id,
    )

    with pytest.raises(
        AgentCheckpointScopeError,
        match="knowledge base",
    ):
        AgentCheckpointService().save_checkpoint(
            db,
            _payload(
                conversation_id=conversation.id,
                user_id=user.id,
                knowledge_base_id=other_kb.id,
                status=AgentStateStatus.RUNNING,
                turn=0,
            ),
        )


def test_deleting_conversation_cascades_thread_and_checkpoints(db: Session) -> None:
    conversation_service, kb_service = _build_conversation_services()
    user = _create_user(db, "checkpoint-delete@example.com")
    kb = kb_service.create(db, user, "Checkpoint Delete KB")
    conversation = conversation_service.create(
        db=db,
        user=user,
        mode=ConversationMode.AGENT,
        knowledge_base_id=kb.id,
    )
    service = AgentCheckpointService()
    checkpoint = service.save_checkpoint(
        db,
        _payload(
            conversation_id=conversation.id,
            user_id=user.id,
            knowledge_base_id=kb.id,
            status=AgentStateStatus.RUNNING,
            turn=0,
        ),
    )
    thread_id = checkpoint.agent_thread_id
    checkpoint_id = checkpoint.id

    conversation_service.delete(
        db=db,
        user_id=user.id,
        conversation_id=conversation.id,
    )

    assert db.get(AgentThread, thread_id) is None
    assert db.get(AgentCheckpoint, checkpoint_id) is None
