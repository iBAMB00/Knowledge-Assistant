import pytest

from app.constants.memory_status import MemoryStatus
from app.constants.memory_type import MemoryType
from app.models.database.conversation import Conversation
from app.models.database.conversation_message import ConversationMessage
from app.models.database.knowledge_base import KnowledgeBase
from app.models.database.memory_item import MemoryItem
from app.models.database.user import User
from app.repositories.conversation_message_repository import (
    ConversationMessageRepository,
)
from app.repositories.memory_item_repository import MemoryItemRepository
from app.services.conversation_memory_service import (
    ConversationMemoryService,
    ConversationMemorySourceError,
)


class FakeConversationService:
    def __init__(self) -> None:
        self.scope_calls = []

    def ensure_chat_scope(self, db, **kwargs):
        self.scope_calls.append(kwargs)
        return object()


def _create_conversation_with_messages(db, *, email: str, contents: list[str]):
    user = User(
        email=email,
        password_hash="test-password-hash",
        role="user",
        is_active=True,
    )
    db.add(user)
    db.flush()

    knowledge_base = KnowledgeBase(
        owner_id=user.id,
        name=f"Memory KB {email}",
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

    messages = []
    for index, content in enumerate(contents):
        message = ConversationMessage(
            conversation_id=conversation.id,
            role="user" if index % 2 == 0 else "assistant",
            content=content,
        )
        db.add(message)
        db.flush()
        messages.append(message)

    return user, knowledge_base, conversation, messages


def _service(fake_conversation_service: FakeConversationService):
    return ConversationMemoryService(
        repository=MemoryItemRepository(),
        message_repository=ConversationMessageRepository(),
        conversation_service=fake_conversation_service,
    )


def test_memory_create_list_forget_and_conversation_cascade(db) -> None:
    user, kb, conversation, messages = _create_conversation_with_messages(
        db,
        email="b6-memory@example.com",
        contents=["我更喜欢直观代码", "已记录这个偏好"],
    )
    fake_scope = FakeConversationService()
    service = _service(fake_scope)

    memory = service.create(
        db,
        user_id=user.id,
        knowledge_base_id=kb.id,
        conversation_id=conversation.id,
        source_message_start_id=messages[0].id,
        source_message_end_id=messages[1].id,
        memory_type=MemoryType.PREFERENCE,
        content="  用户偏好直观、易读的代码实现。  ",
    )

    assert memory.source_conversation_id == conversation.id
    assert memory.source_message_start_id == messages[0].id
    assert memory.source_message_end_id == messages[1].id
    assert memory.memory_type == MemoryType.PREFERENCE.value
    assert memory.content == "用户偏好直观、易读的代码实现。"
    assert memory.status == MemoryStatus.ACTIVE.value

    active = service.list_active(
        db,
        user_id=user.id,
        knowledge_base_id=kb.id,
        conversation_id=conversation.id,
    )
    assert [item.id for item in active] == [memory.id]

    forgotten = service.forget(
        db,
        user_id=user.id,
        knowledge_base_id=kb.id,
        conversation_id=conversation.id,
        memory_id=memory.id,
    )
    assert forgotten.status == MemoryStatus.FORGOTTEN.value
    assert service.list_active(
        db,
        user_id=user.id,
        knowledge_base_id=kb.id,
        conversation_id=conversation.id,
    ) == []

    memory_id = memory.id
    db.delete(conversation)
    db.commit()
    assert db.get(MemoryItem, memory_id) is None

    assert len(fake_scope.scope_calls) == 4
    assert all(
        call["conversation_id"] == conversation.id
        and call["knowledge_base_id"] == kb.id
        and call["mode"].value == "agent"
        for call in fake_scope.scope_calls
    )


def test_memory_source_range_must_belong_to_same_conversation(db) -> None:
    user, kb, conversation, messages = _create_conversation_with_messages(
        db,
        email="b6-memory-source-a@example.com",
        contents=["来源 A"],
    )
    _, _, other_conversation, other_messages = _create_conversation_with_messages(
        db,
        email="b6-memory-source-b@example.com",
        contents=["来源 B"],
    )
    fake_scope = FakeConversationService()
    service = _service(fake_scope)

    with pytest.raises(ConversationMemorySourceError):
        service.create(
            db,
            user_id=user.id,
            knowledge_base_id=kb.id,
            conversation_id=conversation.id,
            source_message_start_id=messages[0].id,
            source_message_end_id=other_messages[0].id,
            memory_type=MemoryType.FACT,
            content="不能跨 Conversation 形成来源范围",
        )

    assert other_conversation.id != conversation.id
    assert db.query(MemoryItem).count() == 0


def test_memory_rejects_reversed_source_range(db) -> None:
    user, kb, conversation, messages = _create_conversation_with_messages(
        db,
        email="b6-memory-range@example.com",
        contents=["第一条", "第二条"],
    )
    service = _service(FakeConversationService())

    with pytest.raises(ConversationMemorySourceError):
        service.create(
            db,
            user_id=user.id,
            knowledge_base_id=kb.id,
            conversation_id=conversation.id,
            source_message_start_id=messages[1].id,
            source_message_end_id=messages[0].id,
            memory_type=MemoryType.FACT,
            content="非法范围",
        )
