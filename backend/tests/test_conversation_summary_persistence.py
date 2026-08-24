from app.models.database.conversation import Conversation
from app.models.database.conversation_message import ConversationMessage
from app.models.database.conversation_summary import ConversationSummary
from app.models.database.knowledge_base import KnowledgeBase
from app.models.database.user import User
from app.repositories.conversation_summary_repository import (
    ConversationSummaryRepository,
)


def test_conversation_summary_upserts_and_cascades_with_conversation(db) -> None:
    user = User(
        email="b5-summary-persistence@example.com",
        password_hash="test-password-hash",
        role="user",
        is_active=True,
    )
    db.add(user)
    db.flush()

    knowledge_base = KnowledgeBase(
        owner_id=user.id,
        name="B5 Summary KB",
    )
    db.add(knowledge_base)
    db.flush()

    conversation = Conversation(
        user_id=user.id,
        knowledge_base_id=knowledge_base.id,
        mode="agent",
    )
    db.add(conversation)
    db.flush()

    message = ConversationMessage(
        conversation_id=conversation.id,
        role="user",
        content="需要被摘要的旧消息",
    )
    db.add(message)
    db.flush()

    repository = ConversationSummaryRepository()
    created = repository.save_or_update(
        db,
        conversation_id=conversation.id,
        content="第一版摘要",
        summarized_through_message_id=message.id,
        prompt_id="agent.conversation-summary",
        prompt_version="1.0.0",
    )
    db.commit()

    updated = repository.save_or_update(
        db,
        conversation_id=conversation.id,
        content="更新后的摘要",
        summarized_through_message_id=message.id,
        prompt_id="agent.conversation-summary",
        prompt_version="1.0.0",
    )
    db.commit()

    assert updated.id == created.id
    assert updated.content == "更新后的摘要"
    summary_id = updated.id

    db.delete(conversation)
    db.commit()

    assert db.get(ConversationSummary, summary_id) is None
