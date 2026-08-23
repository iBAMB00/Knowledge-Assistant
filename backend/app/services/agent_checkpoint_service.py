from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.agent.checkpoint import (
    AgentCheckpointStateTransitionError,
    AgentExecutionCheckpointPayload,
)
from app.constants.agent_state_status import AgentStateStatus
from app.constants.conversation_mode import ConversationMode
from app.models.database.agent_checkpoint import AgentCheckpoint
from app.models.database.agent_thread import AgentThread
from app.repositories.agent_checkpoint_repository import AgentCheckpointRepository
from app.repositories.agent_thread_repository import AgentThreadRepository
from app.repositories.conversation_repository import ConversationRepository


class AgentCheckpointScopeError(ValueError):
    """Checkpoint 的 Thread / Conversation 安全范围不一致。"""


class AgentCheckpointService:
    """AgentThread 与 durable checkpoint 的事务边界。"""

    def __init__(
        self,
        *,
        thread_repository: AgentThreadRepository | None = None,
        checkpoint_repository: AgentCheckpointRepository | None = None,
        conversation_repository: ConversationRepository | None = None,
    ) -> None:
        self.thread_repository = thread_repository or AgentThreadRepository()
        self.checkpoint_repository = (
            checkpoint_repository or AgentCheckpointRepository()
        )
        self.conversation_repository = (
            conversation_repository or ConversationRepository()
        )

    def save_checkpoint(
        self,
        db: Session,
        payload: AgentExecutionCheckpointPayload,
        *,
        allowed_previous_statuses: set[AgentStateStatus] | None = None,
        new_execution_attempt: bool = False,
    ) -> AgentCheckpoint:
        """确保 Thread 存在，并追加一个不可变顺序 checkpoint。"""

        state = payload.agent_state
        thread_identity = state.thread
        conversation = self.conversation_repository.find_owned_by_id(
            db=db,
            conversation_id=state.conversation.conversation_id,
            user_id=state.conversation.user_id,
        )
        if conversation is None:
            raise AgentCheckpointScopeError("conversation not found for checkpoint")
        if conversation.mode != ConversationMode.AGENT.value:
            raise AgentCheckpointScopeError("checkpoint requires agent conversation")
        if conversation.knowledge_base_id != state.conversation.knowledge_base_id:
            raise AgentCheckpointScopeError(
                "checkpoint knowledge base does not match conversation"
            )

        try:
            thread = self._get_or_create_thread(db, payload)

            # 所有 durable lifecycle 写入先锁 Thread。这样用户 Cancel 与正在
            # 执行的 Graph checkpoint 不会互相覆盖：先提交的一方决定下一状态，
            # 后到的一方必须基于最新状态重新判断。
            locked_thread = self.thread_repository.find_by_thread_id_for_update(
                db,
                thread_identity.thread_id,
            )
            if locked_thread is not None:
                thread = locked_thread

            current_status = AgentStateStatus(thread.status)
            requested_status = state.status

            latest = self.checkpoint_repository.find_latest_by_thread_id(
                db,
                thread.id,
            )
            latest_payload = (
                AgentExecutionCheckpointPayload.model_validate(latest.payload)
                if latest is not None
                else None
            )
            current_agent_run_id = (
                latest_payload.agent_state.agent_run_id
                if latest_payload is not None
                else None
            )
            requested_agent_run_id = state.agent_run_id

            if (
                allowed_previous_statuses is not None
                and latest is not None
                and current_status not in allowed_previous_statuses
            ):
                raise AgentCheckpointStateTransitionError(
                    current_status=current_status,
                    requested_status=requested_status,
                    current_agent_run_id=current_agent_run_id,
                    requested_agent_run_id=requested_agent_run_id,
                )

            if new_execution_attempt:
                # Fresh turn / Resume 都必须由新的 AgentRun attempt 接管 Thread。
                # 只有 Runner 的“执行起始 checkpoint”可以跨 AgentRun 边界；
                # 后续普通 checkpoint 必须继续属于当前 active attempt。
                if (
                    requested_status is not AgentStateStatus.RUNNING
                    or requested_agent_run_id is None
                ):
                    raise AgentCheckpointStateTransitionError(
                        current_status=current_status,
                        requested_status=requested_status,
                        current_agent_run_id=current_agent_run_id,
                        requested_agent_run_id=requested_agent_run_id,
                    )
                if (
                    latest is not None
                    and current_agent_run_id is not None
                    and current_agent_run_id == requested_agent_run_id
                ):
                    raise AgentCheckpointStateTransitionError(
                        current_status=current_status,
                        requested_status=requested_status,
                        current_agent_run_id=current_agent_run_id,
                        requested_agent_run_id=requested_agent_run_id,
                    )
            else:
                # 普通 Graph checkpoint 必须被当前 AgentRun fencing。这样旧 Runner
                # 即使在 Cancel 后又返回，或新问题已经开启下一 AgentRun，也不能
                # 用旧内存状态覆盖新 attempt。
                if (
                    current_agent_run_id is not None
                    and requested_agent_run_id is not None
                    and current_agent_run_id != requested_agent_run_id
                ):
                    raise AgentCheckpointStateTransitionError(
                        current_status=current_status,
                        requested_status=requested_status,
                        current_agent_run_id=current_agent_run_id,
                        requested_agent_run_id=requested_agent_run_id,
                    )

                # CANCELLED 仍是“当前 AgentRun”的 durable terminal state。
                # 旧 attempt 不能复活；只有上面的 new_execution_attempt 分支
                # 才允许同一 Thread 开启下一轮 fresh task。
                if (
                    current_status is AgentStateStatus.CANCELLED
                    and requested_status is not AgentStateStatus.CANCELLED
                ):
                    raise AgentCheckpointStateTransitionError(
                        current_status=current_status,
                        requested_status=requested_status,
                        current_agent_run_id=current_agent_run_id,
                        requested_agent_run_id=requested_agent_run_id,
                    )

            sequence = 1 if latest is None else latest.sequence + 1

            checkpoint = AgentCheckpoint(
                agent_thread_id=thread.id,
                sequence=sequence,
                checkpoint_schema_version=(
                    payload.checkpoint_schema_version
                ),
                state_schema_version=state.state_schema_version,
                payload=payload.model_dump(mode="json"),
            )
            self.checkpoint_repository.create(db, checkpoint)

            thread.status = state.status.value
            thread.state_schema_version = state.state_schema_version
            thread.updated_at = datetime.now(timezone.utc)

            db.commit()
            db.refresh(checkpoint)
            return checkpoint
        except Exception:
            db.rollback()
            raise

    def load_latest(
        self,
        db: Session,
        *,
        thread_id: str,
    ) -> AgentExecutionCheckpointPayload | None:
        """读取并校验 Thread 最新 checkpoint。"""

        thread = self.thread_repository.find_by_thread_id(db, thread_id)
        if thread is None:
            return None
        checkpoint = self.checkpoint_repository.find_latest_by_thread_id(
            db,
            thread.id,
        )
        if checkpoint is None:
            return None
        return AgentExecutionCheckpointPayload.model_validate(
            checkpoint.payload
        )

    def load_latest_for_scope(
        self,
        db: Session,
        *,
        thread_id: str,
        user_id: int,
        knowledge_base_id: int,
    ) -> AgentExecutionCheckpointPayload | None:
        """
        在可信 User / KB Scope 下读取最新 checkpoint。

        找不到 Thread、Conversation 不属于当前用户，或 KB 不匹配时统一
        返回 None，避免把别人的 Thread 是否存在泄漏给调用方。持久化数据
        自身若与数据库 Scope 冲突，则视为状态损坏并显式报错。
        """

        thread = self.thread_repository.find_by_thread_id(db, thread_id)
        if thread is None:
            return None

        conversation = self.conversation_repository.find_owned_by_id(
            db=db,
            conversation_id=thread.conversation_id,
            user_id=user_id,
        )
        if conversation is None:
            return None
        if conversation.mode != ConversationMode.AGENT.value:
            return None
        if conversation.knowledge_base_id != knowledge_base_id:
            return None

        checkpoint = self.checkpoint_repository.find_latest_by_thread_id(
            db,
            thread.id,
        )
        if checkpoint is None:
            return None

        payload = AgentExecutionCheckpointPayload.model_validate(
            checkpoint.payload
        )
        state = payload.agent_state

        if (
            checkpoint.checkpoint_schema_version
            != payload.checkpoint_schema_version
        ):
            raise AgentCheckpointScopeError(
                "checkpoint schema metadata is corrupted"
            )
        if checkpoint.state_schema_version != state.state_schema_version:
            raise AgentCheckpointScopeError(
                "checkpoint state schema metadata is corrupted"
            )
        if thread.state_schema_version != state.state_schema_version:
            raise AgentCheckpointScopeError(
                "thread state schema metadata is corrupted"
            )
        if thread.status != state.status.value:
            raise AgentCheckpointScopeError(
                "thread status does not match latest checkpoint"
            )

        if state.thread.thread_id != thread.thread_id:
            raise AgentCheckpointScopeError(
                "checkpoint thread_id does not match persisted thread"
            )
        if state.thread.conversation_id != thread.conversation_id:
            raise AgentCheckpointScopeError(
                "checkpoint conversation does not match persisted thread"
            )
        if state.conversation.conversation_id != conversation.id:
            raise AgentCheckpointScopeError(
                "checkpoint conversation scope is corrupted"
            )
        if state.conversation.user_id != user_id:
            raise AgentCheckpointScopeError(
                "checkpoint user scope is corrupted"
            )
        if state.conversation.knowledge_base_id != knowledge_base_id:
            raise AgentCheckpointScopeError(
                "checkpoint knowledge base scope is corrupted"
            )

        return payload

    def list_checkpoints(
        self,
        db: Session,
        *,
        thread_id: str,
    ) -> list[AgentCheckpoint]:
        thread = self.thread_repository.find_by_thread_id(db, thread_id)
        if thread is None:
            return []
        return self.checkpoint_repository.list_by_thread_id(db, thread.id)

    def _get_or_create_thread(
        self,
        db: Session,
        payload: AgentExecutionCheckpointPayload,
    ) -> AgentThread:
        state = payload.agent_state
        identity = state.thread

        existing = self.thread_repository.find_by_thread_id(
            db,
            identity.thread_id,
        )
        by_conversation = self.thread_repository.find_by_conversation_id(
            db,
            identity.conversation_id,
        )

        if existing is not None:
            if existing.conversation_id != identity.conversation_id:
                raise AgentCheckpointScopeError(
                    "thread_id already belongs to another conversation"
                )
            return existing

        if by_conversation is not None:
            if by_conversation.thread_id != identity.thread_id:
                raise AgentCheckpointScopeError(
                    "conversation already belongs to another thread"
                )
            return by_conversation

        thread = AgentThread(
            thread_id=identity.thread_id,
            conversation_id=identity.conversation_id,
            state_schema_version=state.state_schema_version,
            status=state.status.value,
        )
        return self.thread_repository.create(db, thread)
