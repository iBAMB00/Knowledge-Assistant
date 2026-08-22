from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.agent.checkpoint import AgentExecutionCheckpointPayload
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
            latest = self.checkpoint_repository.find_latest_by_thread_id(
                db,
                thread.id,
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
