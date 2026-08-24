from sqlalchemy.orm import Session

from app.constants.conversation_message_role import ConversationMessageRole

from app.models.database.conversation_message import ConversationMessage


class ConversationMessageRepository:
    """ConversationMessage 数据访问层；只负责查询、add 与 flush。"""

    def create(
        self,
        db: Session,
        message: ConversationMessage,
    ) -> ConversationMessage:
        db.add(message)
        db.flush()
        return message

    def find_by_ids_in_conversation(
        self,
        db: Session,
        *,
        conversation_id: int,
        message_ids: set[int],
    ) -> list[ConversationMessage]:
        if not message_ids:
            return []
        return (
            db.query(ConversationMessage)
            .filter(
                ConversationMessage.conversation_id == conversation_id,
                ConversationMessage.id.in_(message_ids),
            )
            .order_by(ConversationMessage.id.asc())
            .all()
        )

    def find_latest_user_before(
        self,
        db: Session,
        *,
        conversation_id: int,
        before_message_id: int,
    ) -> ConversationMessage | None:
        """查找指定 assistant message 之前最近的用户消息。"""

        return (
            db.query(ConversationMessage)
            .filter(
                ConversationMessage.conversation_id == conversation_id,
                ConversationMessage.id < before_message_id,
                ConversationMessage.role == ConversationMessageRole.USER.value,
            )
            .order_by(ConversationMessage.id.desc())
            .first()
        )

    def find_by_conversation_id(
        self,
        db: Session,
        *,
        conversation_id: int,
        limit: int = 200,
    ) -> list[ConversationMessage]:
        # 先取最近 N 条，再恢复为用户阅读所需的时间正序。
        rows = (
            db.query(ConversationMessage)
            .filter(
                ConversationMessage.conversation_id == conversation_id
            )
            .order_by(ConversationMessage.id.desc())
            .limit(limit)
            .all()
        )
        rows.reverse()
        return rows
