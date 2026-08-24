from sqlalchemy.orm import Session

from app.models.database.conversation_summary import ConversationSummary


class ConversationSummaryRepository:
    """ConversationSummary 数据访问层；只负责查询、add 与 flush。"""

    def find_by_conversation_id(
        self,
        db: Session,
        *,
        conversation_id: int,
    ) -> ConversationSummary | None:
        return (
            db.query(ConversationSummary)
            .filter(
                ConversationSummary.conversation_id == conversation_id
            )
            .first()
        )

    def save_or_update(
        self,
        db: Session,
        *,
        conversation_id: int,
        content: str,
        summarized_through_message_id: int,
        prompt_id: str,
        prompt_version: str,
    ) -> ConversationSummary:
        summary = self.find_by_conversation_id(
            db,
            conversation_id=conversation_id,
        )
        if summary is None:
            summary = ConversationSummary(
                conversation_id=conversation_id,
                content=content,
                summarized_through_message_id=(
                    summarized_through_message_id
                ),
                prompt_id=prompt_id,
                prompt_version=prompt_version,
            )
            db.add(summary)
        else:
            summary.content = content
            summary.summarized_through_message_id = (
                summarized_through_message_id
            )
            summary.prompt_id = prompt_id
            summary.prompt_version = prompt_version

        db.flush()
        return summary
