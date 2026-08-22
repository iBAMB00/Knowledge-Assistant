from sqlalchemy.orm import Session

from app.constants.conversation_mode import ConversationMode
from app.models.database.conversation import Conversation


class ConversationRepository:
    """Conversation 数据访问层；只负责查询、add、delete 与 flush。"""

    def create(
        self,
        db: Session,
        conversation: Conversation,
    ) -> Conversation:
        db.add(conversation)
        db.flush()
        return conversation

    def find_owned_by_id(
        self,
        db: Session,
        *,
        conversation_id: int,
        user_id: int,
    ) -> Conversation | None:
        return (
            db.query(Conversation)
            .filter(
                Conversation.id == conversation_id,
                Conversation.user_id == user_id,
            )
            .first()
        )

    def find_recent_by_user_id(
        self,
        db: Session,
        *,
        user_id: int,
        mode: ConversationMode | None = None,
        knowledge_base_id: int | None = None,
        limit: int = 50,
    ) -> list[Conversation]:
        query = db.query(Conversation).filter(
            Conversation.user_id == user_id
        )

        if mode is not None:
            query = query.filter(Conversation.mode == mode.value)
        if knowledge_base_id is not None:
            query = query.filter(
                Conversation.knowledge_base_id == knowledge_base_id
            )

        return (
            query.order_by(
                Conversation.updated_at.desc(),
                Conversation.id.desc(),
            )
            .limit(limit)
            .all()
        )
