from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.constants.conversation_message_role import ConversationMessageRole
from app.constants.conversation_mode import ConversationMode
from app.models.database.conversation import Conversation
from app.models.database.conversation_message import ConversationMessage
from app.models.database.user import User
from app.repositories.conversation_message_repository import (
    ConversationMessageRepository,
)
from app.repositories.conversation_repository import ConversationRepository
from app.services.knowledge_base_access_policy import KnowledgeBaseAccessPolicy


class ConversationNotFoundError(ValueError):
    """Conversation 不存在或不属于当前用户。"""


class ConversationService:
    """Conversation / Message 持久化与用户级隔离边界。"""

    AUTO_TITLE_MAX_LENGTH = 60

    def __init__(
        self,
        *,
        conversation_repository: ConversationRepository,
        message_repository: ConversationMessageRepository,
        access_policy: KnowledgeBaseAccessPolicy,
    ) -> None:
        self.conversation_repository = conversation_repository
        self.message_repository = message_repository
        self.access_policy = access_policy

    @staticmethod
    def _normalize_title(title: str | None) -> str | None:
        if title is None:
            return None
        normalized = title.strip()
        return normalized or None

    @staticmethod
    def _validate_limit(limit: int, *, maximum: int) -> int:
        if limit < 1 or limit > maximum:
            raise ValueError(
                f"limit must be between 1 and {maximum}"
            )
        return limit

    def create(
        self,
        db: Session,
        *,
        user: User,
        mode: ConversationMode,
        knowledge_base_id: int,
        title: str | None = None,
    ) -> Conversation:
        # KB 可访问性必须在 Conversation Scope 固化前完成校验。
        self.access_policy.get_accessible_knowledge_base(
            db=db,
            knowledge_base_id=knowledge_base_id,
            user=user,
        )

        conversation = Conversation(
            user_id=user.id,
            knowledge_base_id=knowledge_base_id,
            mode=mode.value,
            title=self._normalize_title(title),
        )

        try:
            self.conversation_repository.create(db, conversation)
            db.commit()
            db.refresh(conversation)
            return conversation
        except Exception:
            db.rollback()
            raise

    def list_owned(
        self,
        db: Session,
        *,
        user_id: int,
        mode: ConversationMode | None = None,
        knowledge_base_id: int | None = None,
        limit: int = 50,
    ) -> list[Conversation]:
        return self.conversation_repository.find_recent_by_user_id(
            db=db,
            user_id=user_id,
            mode=mode,
            knowledge_base_id=knowledge_base_id,
            limit=self._validate_limit(limit, maximum=100),
        )

    def get_owned(
        self,
        db: Session,
        *,
        user_id: int,
        conversation_id: int,
    ) -> Conversation:
        conversation = self.conversation_repository.find_owned_by_id(
            db=db,
            conversation_id=conversation_id,
            user_id=user_id,
        )
        if conversation is None:
            raise ConversationNotFoundError(
                "conversation not found"
            )
        return conversation

    def list_messages(
        self,
        db: Session,
        *,
        user_id: int,
        conversation_id: int,
        limit: int = 200,
    ) -> list[ConversationMessage]:
        conversation = self.get_owned(
            db=db,
            user_id=user_id,
            conversation_id=conversation_id,
        )
        return self.message_repository.find_by_conversation_id(
            db=db,
            conversation_id=conversation.id,
            limit=self._validate_limit(limit, maximum=500),
        )

    def append_message(
        self,
        db: Session,
        *,
        user_id: int,
        conversation_id: int,
        role: ConversationMessageRole,
        content: str,
    ) -> ConversationMessage:
        """
        内部消息写入入口。

        A2 暂不向客户端暴露任意 role 的 Message 写 API；后续 RAG / Agent
        Chat 集成会从服务端调用本方法，避免客户端伪造 assistant 历史。
        """

        conversation = self.get_owned(
            db=db,
            user_id=user_id,
            conversation_id=conversation_id,
        )
        normalized_content = content.strip()
        if not normalized_content:
            raise ValueError("message content cannot be empty")
        if len(normalized_content) > 20000:
            raise ValueError("message content is too long")

        message = ConversationMessage(
            conversation_id=conversation.id,
            role=role.value,
            content=normalized_content,
        )

        if (
            role == ConversationMessageRole.USER
            and conversation.title is None
        ):
            conversation.title = normalized_content[
                : self.AUTO_TITLE_MAX_LENGTH
            ]

        # Message insert 不会自动触发 Conversation.updated_at 的 onupdate，
        # 因此显式 touch，保证历史列表按最近活动排序。
        conversation.updated_at = datetime.now(timezone.utc)

        try:
            self.message_repository.create(db, message)
            db.commit()
            db.refresh(message)
            return message
        except Exception:
            db.rollback()
            raise
