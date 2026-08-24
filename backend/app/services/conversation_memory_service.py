from sqlalchemy.orm import Session

from app.constants.conversation_mode import ConversationMode
from app.constants.memory_status import MemoryStatus
from app.constants.memory_type import MemoryType
from app.models.database.memory_item import MemoryItem
from app.repositories.conversation_message_repository import (
    ConversationMessageRepository,
)
from app.repositories.memory_item_repository import MemoryItemRepository
from app.services.conversation_service import ConversationService


class ConversationMemoryNotFoundError(ValueError):
    """Memory 不存在或不属于当前 Conversation。"""


class ConversationMemorySourceError(ValueError):
    """Memory provenance 不属于目标 Conversation。"""


class ConversationMemoryService:
    """Conversation-derived Memory 的持久化与 Scope 边界。

    B6 只建立持久化事实，不负责“什么值得记住”的 LLM 抽取，也不负责把
    Memory 注入 Agent Context；这两部分分别留给 B7 / B8。
    """

    MAX_CONTENT_LENGTH = 4_000
    MAX_LIST_LIMIT = 200

    def __init__(
        self,
        *,
        repository: MemoryItemRepository,
        message_repository: ConversationMessageRepository,
        conversation_service: ConversationService,
    ) -> None:
        self.repository = repository
        self.message_repository = message_repository
        self.conversation_service = conversation_service

    def create(
        self,
        db: Session,
        *,
        user_id: int,
        knowledge_base_id: int,
        conversation_id: int,
        source_message_start_id: int,
        source_message_end_id: int,
        memory_type: MemoryType,
        content: str,
    ) -> MemoryItem:
        self._ensure_scope(
            db,
            user_id=user_id,
            knowledge_base_id=knowledge_base_id,
            conversation_id=conversation_id,
        )
        self._validate_source_range(
            db,
            conversation_id=conversation_id,
            source_message_start_id=source_message_start_id,
            source_message_end_id=source_message_end_id,
        )

        normalized_content = content.strip()
        if not normalized_content:
            raise ValueError("memory content cannot be empty")
        if len(normalized_content) > self.MAX_CONTENT_LENGTH:
            raise ValueError("memory content is too long")

        memory = MemoryItem(
            source_conversation_id=conversation_id,
            source_message_start_id=source_message_start_id,
            source_message_end_id=source_message_end_id,
            memory_type=memory_type.value,
            content=normalized_content,
            status=MemoryStatus.ACTIVE.value,
        )

        try:
            self.repository.create(db, memory)
            db.commit()
            db.refresh(memory)
            return memory
        except Exception:
            db.rollback()
            raise

    def list_active(
        self,
        db: Session,
        *,
        user_id: int,
        knowledge_base_id: int,
        conversation_id: int,
        limit: int = 100,
    ) -> list[MemoryItem]:
        self._ensure_scope(
            db,
            user_id=user_id,
            knowledge_base_id=knowledge_base_id,
            conversation_id=conversation_id,
        )
        if limit < 1 or limit > self.MAX_LIST_LIMIT:
            raise ValueError(
                f"limit must be between 1 and {self.MAX_LIST_LIMIT}"
            )
        return self.repository.find_active_by_conversation(
            db,
            source_conversation_id=conversation_id,
            limit=limit,
        )

    def forget(
        self,
        db: Session,
        *,
        user_id: int,
        knowledge_base_id: int,
        conversation_id: int,
        memory_id: int,
    ) -> MemoryItem:
        """逻辑忘记一条 Memory；B8 默认只检索 ACTIVE Memory。"""

        self._ensure_scope(
            db,
            user_id=user_id,
            knowledge_base_id=knowledge_base_id,
            conversation_id=conversation_id,
        )
        memory = self.repository.find_by_id_and_conversation(
            db,
            memory_id=memory_id,
            source_conversation_id=conversation_id,
        )
        if memory is None:
            raise ConversationMemoryNotFoundError("memory not found")

        try:
            self.repository.update_status(
                db,
                memory=memory,
                status=MemoryStatus.FORGOTTEN,
            )
            db.commit()
            db.refresh(memory)
            return memory
        except Exception:
            db.rollback()
            raise

    def _ensure_scope(
        self,
        db: Session,
        *,
        user_id: int,
        knowledge_base_id: int,
        conversation_id: int,
    ) -> None:
        self.conversation_service.ensure_chat_scope(
            db=db,
            user_id=user_id,
            conversation_id=conversation_id,
            mode=ConversationMode.AGENT,
            knowledge_base_id=knowledge_base_id,
        )

    def _validate_source_range(
        self,
        db: Session,
        *,
        conversation_id: int,
        source_message_start_id: int,
        source_message_end_id: int,
    ) -> None:
        if source_message_start_id <= 0 or source_message_end_id <= 0:
            raise ConversationMemorySourceError(
                "memory source message ids must be positive"
            )
        if source_message_start_id > source_message_end_id:
            raise ConversationMemorySourceError(
                "memory source message range is invalid"
            )

        source_ids = {
            source_message_start_id,
            source_message_end_id,
        }
        messages = self.message_repository.find_by_ids_in_conversation(
            db=db,
            conversation_id=conversation_id,
            message_ids=source_ids,
        )
        if {message.id for message in messages} != source_ids:
            raise ConversationMemorySourceError(
                "memory source messages do not belong to conversation"
            )
