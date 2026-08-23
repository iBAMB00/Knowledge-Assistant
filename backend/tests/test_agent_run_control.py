from __future__ import annotations

import pytest
from sqlalchemy.orm import Session

from app.agent.checkpoint import (
    AgentCheckpointStateTransitionError,
    AgentExecutionCheckpointPayload,
)
from app.agent.hitl import AgentApprovalRequirement, AgentApprovalStateError
from app.agent.model_response import LLMToolCall, LLMToolResponse
from app.agent.run_control import (
    AgentRunCancellationError,
    AgentRunControlNotFoundError,
    AgentRunStateError,
)
from app.agent.state import AgentState, AgentThreadIdentity
from app.constants.agent_state_status import AgentStateStatus
from app.constants.conversation_message_role import ConversationMessageRole
from app.constants.conversation_mode import ConversationMode
from app.models.database.user import User
from app.repositories.conversation_message_repository import (
    ConversationMessageRepository,
)
from app.repositories.conversation_repository import ConversationRepository
from app.repositories.document_repository import DocumentRepository
from app.repositories.knowledge_base_repository import KnowledgeBaseRepository
from app.schemas.conversation_contract import (
    ConversationMessagePayload,
    ConversationScope,
)
from app.services.agent_checkpoint_service import AgentCheckpointService
from app.services.agent_hitl_service import AgentHITLService
from app.services.agent_run_control_service import AgentRunControlService
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


def _build_services() -> tuple[ConversationService, KnowledgeBaseService]:
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
    agent_run_id: int | str | None = None,
) -> AgentExecutionCheckpointPayload:
    base = {
        "agent_state": AgentState(
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
            agent_run_id=agent_run_id,
            status=status,
            task="执行 Stateful Agent",
            messages=(
                ConversationMessagePayload(
                    role=ConversationMessageRole.USER,
                    content="执行 Stateful Agent",
                ),
            ),
        ),
    }

    if status is AgentStateStatus.WAITING:
        tool_call = LLMToolCall(
            id="cancel-call-1",
            name="echo",
            arguments_json='{"text":"sensitive"}',
        )
        return AgentExecutionCheckpointPayload(
            **base,
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
        )

    if status is AgentStateStatus.SUCCEEDED:
        succeeded_state = base["agent_state"].model_copy(
            update={
                "messages": (
                    *base["agent_state"].messages,
                    ConversationMessagePayload(
                        role=ConversationMessageRole.ASSISTANT,
                        content="完成",
                    ),
                )
            }
        )
        return AgentExecutionCheckpointPayload(
            agent_state=succeeded_state,
            final_answer="完成",
            turn=1,
        )

    return AgentExecutionCheckpointPayload(
        **base,
        turn=1 if status is AgentStateStatus.RUNNING else 0,
    )


def _create_thread(
    db: Session,
    *,
    email: str,
    status: AgentStateStatus,
) -> tuple[User, int, int, str, AgentCheckpointService]:
    conversation_service, kb_service = _build_services()
    user = _create_user(db, email)
    kb = kb_service.create(db, user, "Run Control KB")
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
        _payload(
            conversation_id=conversation.id,
            user_id=user.id,
            knowledge_base_id=kb.id,
            status=status,
        ),
    )
    return user, kb.id, conversation.id, thread_id, checkpoint_service


def test_cancel_running_thread_persists_cancelled_checkpoint(
    db: Session,
) -> None:
    user, kb_id, _, thread_id, checkpoint_service = _create_thread(
        db,
        email="cancel-running@example.com",
        status=AgentStateStatus.RUNNING,
    )
    service = AgentRunControlService(checkpoint_service)

    cancelled = service.cancel(
        db,
        thread_id=thread_id,
        user_id=user.id,
        knowledge_base_id=kb_id,
    )

    assert cancelled.agent_state.status is AgentStateStatus.CANCELLED
    assert cancelled.agent_state.last_error_code == "cancelled_by_user"
    assert checkpoint_service.load_latest(
        db,
        thread_id=thread_id,
    ) == cancelled
    assert len(
        checkpoint_service.list_checkpoints(db, thread_id=thread_id)
    ) == 2

    with pytest.raises(AgentRunCancellationError, match="cancelled"):
        service.raise_if_cancelled(
            db,
            thread_id=thread_id,
            user_id=user.id,
            knowledge_base_id=kb_id,
        )


def test_cancel_is_idempotent_and_does_not_append_duplicate_checkpoint(
    db: Session,
) -> None:
    user, kb_id, _, thread_id, checkpoint_service = _create_thread(
        db,
        email="cancel-idempotent@example.com",
        status=AgentStateStatus.RUNNING,
    )
    service = AgentRunControlService(checkpoint_service)

    first = service.cancel(
        db,
        thread_id=thread_id,
        user_id=user.id,
        knowledge_base_id=kb_id,
    )
    second = service.cancel(
        db,
        thread_id=thread_id,
        user_id=user.id,
        knowledge_base_id=kb_id,
    )

    assert second == first
    assert len(
        checkpoint_service.list_checkpoints(db, thread_id=thread_id)
    ) == 2


def test_cancel_waiting_thread_blocks_later_hitl_approval(
    db: Session,
) -> None:
    user, kb_id, _, thread_id, checkpoint_service = _create_thread(
        db,
        email="cancel-waiting@example.com",
        status=AgentStateStatus.WAITING,
    )
    run_control = AgentRunControlService(checkpoint_service)
    hitl = AgentHITLService(checkpoint_service)

    cancelled = run_control.cancel(
        db,
        thread_id=thread_id,
        user_id=user.id,
        knowledge_base_id=kb_id,
    )

    assert cancelled.agent_state.status is AgentStateStatus.CANCELLED
    with pytest.raises(AgentApprovalStateError, match="not waiting"):
        hitl.approve(
            db,
            thread_id=thread_id,
            user_id=user.id,
            knowledge_base_id=kb_id,
        )


def test_cancel_rejects_completed_thread_and_hides_other_scope(
    db: Session,
) -> None:
    user, kb_id, _, thread_id, checkpoint_service = _create_thread(
        db,
        email="cancel-completed@example.com",
        status=AgentStateStatus.SUCCEEDED,
    )
    service = AgentRunControlService(checkpoint_service)

    with pytest.raises(AgentRunStateError, match="not cancellable"):
        service.cancel(
            db,
            thread_id=thread_id,
            user_id=user.id,
            knowledge_base_id=kb_id,
        )

    with pytest.raises(AgentRunControlNotFoundError, match="not found"):
        service.cancel(
            db,
            thread_id=thread_id,
            user_id=user.id + 999,
            knowledge_base_id=kb_id,
        )


def test_cancelled_thread_rejects_stale_running_checkpoint(
    db: Session,
) -> None:
    user, kb_id, conversation_id, thread_id, checkpoint_service = _create_thread(
        db,
        email="cancel-stale@example.com",
        status=AgentStateStatus.RUNNING,
    )
    stale_payload = _payload(
        conversation_id=conversation_id,
        user_id=user.id,
        knowledge_base_id=kb_id,
        status=AgentStateStatus.RUNNING,
    )
    AgentRunControlService(checkpoint_service).cancel(
        db,
        thread_id=thread_id,
        user_id=user.id,
        knowledge_base_id=kb_id,
    )

    with pytest.raises(AgentCheckpointStateTransitionError) as exc_info:
        checkpoint_service.save_checkpoint(db, stale_payload)

    assert exc_info.value.current_status is AgentStateStatus.CANCELLED
    latest = checkpoint_service.load_latest(
        db,
        thread_id=thread_id,
    )
    assert latest is not None
    assert latest.agent_state.status is AgentStateStatus.CANCELLED


def test_cancelled_thread_can_start_new_run_without_reviving_old_run(
    db: Session,
) -> None:
    user, kb_id, conversation_id, thread_id, checkpoint_service = _create_thread(
        db,
        email="cancel-new-run@example.com",
        status=AgentStateStatus.RUNNING,
    )

    # 给首个 attempt 绑定真实 run id，模拟生产 LangGraph Runner。
    first_running = _payload(
        conversation_id=conversation_id,
        user_id=user.id,
        knowledge_base_id=kb_id,
        status=AgentStateStatus.RUNNING,
        agent_run_id=101,
    )
    # _create_thread 已写入一条 legacy-style checkpoint；以显式新 attempt
    # 接管后再测试 Cancel -> 下一轮 fresh run。
    checkpoint_service.save_checkpoint(
        db,
        first_running,
        allowed_previous_statuses={AgentStateStatus.RUNNING},
        new_execution_attempt=True,
    )

    AgentRunControlService(checkpoint_service).cancel(
        db,
        thread_id=thread_id,
        user_id=user.id,
        knowledge_base_id=kb_id,
    )

    second_running = _payload(
        conversation_id=conversation_id,
        user_id=user.id,
        knowledge_base_id=kb_id,
        status=AgentStateStatus.RUNNING,
        agent_run_id=102,
    )
    checkpoint_service.save_checkpoint(
        db,
        second_running,
        allowed_previous_statuses={AgentStateStatus.CANCELLED},
        new_execution_attempt=True,
    )

    latest = checkpoint_service.load_latest(db, thread_id=thread_id)
    assert latest is not None
    assert latest.agent_state.status is AgentStateStatus.RUNNING
    assert latest.agent_state.agent_run_id == 102
    assert latest.agent_state.last_error_code is None

    # 旧 run #101 即使晚到，也不能覆盖新 run #102。
    stale_old_run = _payload(
        conversation_id=conversation_id,
        user_id=user.id,
        knowledge_base_id=kb_id,
        status=AgentStateStatus.RUNNING,
        agent_run_id=101,
    )
    with pytest.raises(AgentCheckpointStateTransitionError) as exc_info:
        checkpoint_service.save_checkpoint(db, stale_old_run)

    assert exc_info.value.current_agent_run_id == 102
    assert exc_info.value.requested_agent_run_id == 101


def test_cancellation_probe_fences_superseded_run_after_new_turn(
    db: Session,
) -> None:
    user, kb_id, conversation_id, thread_id, checkpoint_service = _create_thread(
        db,
        email="cancel-fence@example.com",
        status=AgentStateStatus.RUNNING,
    )

    first_running = _payload(
        conversation_id=conversation_id,
        user_id=user.id,
        knowledge_base_id=kb_id,
        status=AgentStateStatus.RUNNING,
        agent_run_id=201,
    )
    checkpoint_service.save_checkpoint(
        db,
        first_running,
        allowed_previous_statuses={AgentStateStatus.RUNNING},
        new_execution_attempt=True,
    )
    run_control = AgentRunControlService(checkpoint_service)
    run_control.cancel(
        db,
        thread_id=thread_id,
        user_id=user.id,
        knowledge_base_id=kb_id,
    )

    second_running = _payload(
        conversation_id=conversation_id,
        user_id=user.id,
        knowledge_base_id=kb_id,
        status=AgentStateStatus.RUNNING,
        agent_run_id=202,
    )
    checkpoint_service.save_checkpoint(
        db,
        second_running,
        allowed_previous_statuses={AgentStateStatus.CANCELLED},
        new_execution_attempt=True,
    )

    with pytest.raises(AgentRunCancellationError, match="superseded"):
        run_control.raise_if_cancelled(
            db,
            thread_id=thread_id,
            user_id=user.id,
            knowledge_base_id=kb_id,
            agent_run_id=201,
        )

    run_control.raise_if_cancelled(
        db,
        thread_id=thread_id,
        user_id=user.id,
        knowledge_base_id=kb_id,
        agent_run_id=202,
    )
