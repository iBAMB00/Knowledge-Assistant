import pytest
from sqlalchemy.exc import IntegrityError

from app.models.database.conversation import Conversation
from app.models.database.conversation_message import ConversationMessage
from app.models.database.knowledge_base import KnowledgeBase
from app.models.database.memory_item import MemoryItem
from app.models.database.user import User


def _scope(db):
    user = User(
        email="b6-memory-contract@example.com",
        password_hash="test-password-hash",
        role="user",
        is_active=True,
    )
    db.add(user)
    db.flush()
    kb = KnowledgeBase(owner_id=user.id, name="B6 Contract KB")
    db.add(kb)
    db.flush()
    conversation = Conversation(
        user_id=user.id,
        knowledge_base_id=kb.id,
        mode="agent",
    )
    db.add(conversation)
    db.flush()
    message = ConversationMessage(
        conversation_id=conversation.id,
        role="user",
        content="来源",
    )
    db.add(message)
    db.flush()
    return conversation, message


def test_memory_item_database_rejects_unknown_type(db) -> None:
    conversation, message = _scope(db)
    db.add(
        MemoryItem(
            source_conversation_id=conversation.id,
            source_message_start_id=message.id,
            source_message_end_id=message.id,
            memory_type="unknown",
            content="非法类型",
            status="active",
        )
    )

    with pytest.raises(IntegrityError):
        db.flush()
    db.rollback()
