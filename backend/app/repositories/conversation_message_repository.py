from sqlalchemy.orm import Session

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
