import pytest

from app.constants.memory_type import MemoryType
from app.models.database.conversation import Conversation
from app.models.database.conversation_message import ConversationMessage
from app.models.database.knowledge_base import KnowledgeBase
from app.models.database.user import User
from app.repositories.conversation_message_repository import (
    ConversationMessageRepository,
)
from app.repositories.memory_item_repository import MemoryItemRepository
from app.services.conversation_memory_extraction_service import (
    ConversationMemoryExtractionError,
    ConversationMemoryExtractionService,
    LLMConversationMemoryExtractor,
)
from app.services.conversation_memory_service import ConversationMemoryService


class FakeConversationService:
    def ensure_chat_scope(self, db, **kwargs):
        return object()


class FakeLLM:
    def __init__(self, output: str) -> None:
        self.output = output
        self.calls = []

    def complete_prompt(self, prompt, *, input_message=None, temperature=0.0):
        self.calls.append((prompt, input_message, temperature))
        return self.output


def _create_scope(db):
    user = User(
        email="b7-memory@example.com",
        password_hash="test-password-hash",
        role="user",
        is_active=True,
    )
    db.add(user)
    db.flush()

    kb = KnowledgeBase(owner_id=user.id, name="B7 Memory KB")
    db.add(kb)
    db.flush()

    conversation = Conversation(
        user_id=user.id,
        knowledge_base_id=kb.id,
        mode="agent",
    )
    db.add(conversation)
    db.flush()

    user_message = ConversationMessage(
        conversation_id=conversation.id,
        role="user",
        content="以后代码尽量简单直观，我决定继续使用 FastAPI。",
    )
    db.add(user_message)
    db.flush()

    assistant_message = ConversationMessage(
        conversation_id=conversation.id,
        role="assistant",
        content="好的，后续会优先使用简单直观的代码，并继续基于 FastAPI。",
    )
    db.add(assistant_message)
    db.commit()

    return user, kb, conversation, user_message, assistant_message


def _build_service(output: str):
    messages = ConversationMessageRepository()
    memory_service = ConversationMemoryService(
        repository=MemoryItemRepository(),
        message_repository=messages,
        conversation_service=FakeConversationService(),
    )
    llm = FakeLLM(output)
    return (
        ConversationMemoryExtractionService(
            extractor=LLMConversationMemoryExtractor(llm_service=llm),
            memory_service=memory_service,
            message_repository=messages,
        ),
        memory_service,
        llm,
    )


def test_extract_completed_turn_persists_structured_memories(db) -> None:
    user, kb, conversation, user_message, assistant_message = _create_scope(db)
    service, memory_service, llm = _build_service(
        '''{"memories":[
            {"memory_type":"preference","content":"用户偏好简单直观的代码实现。"},
            {"memory_type":"decision","content":"用户决定继续使用 FastAPI。"}
        ]}'''
    )

    result = service.extract_completed_turn(
        db,
        user_id=user.id,
        knowledge_base_id=kb.id,
        conversation_id=conversation.id,
        user_message_id=user_message.id,
        assistant_message_id=assistant_message.id,
    )

    assert len(result.created) == 2
    assert result.skipped_duplicates == 0
    assert result.prompt_id == "agent.conversation-memory-extraction"
    assert result.prompt_version == "1.0.0"
    assert all(
        item.source_message_start_id == user_message.id
        and item.source_message_end_id == assistant_message.id
        for item in result.created
    )
    assert [item.memory_type for item in result.created] == [
        MemoryType.PREFERENCE.value,
        MemoryType.DECISION.value,
    ]
    assert len(memory_service.list_active(
        db,
        user_id=user.id,
        knowledge_base_id=kb.id,
        conversation_id=conversation.id,
    )) == 2
    assert "当前用户消息" in llm.calls[0][1]
    assert "本轮助手回答" in llm.calls[0][1]


def test_extract_completed_turn_skips_exact_active_duplicate(db) -> None:
    user, kb, conversation, user_message, assistant_message = _create_scope(db)
    service, memory_service, _ = _build_service(
        '{"memories":[{"memory_type":"preference","content":"用户偏好简单直观的代码实现。"}]}'
    )
    memory_service.create(
        db,
        user_id=user.id,
        knowledge_base_id=kb.id,
        conversation_id=conversation.id,
        source_message_start_id=user_message.id,
        source_message_end_id=assistant_message.id,
        memory_type=MemoryType.PREFERENCE,
        content="用户偏好简单直观的代码实现。",
    )

    result = service.extract_completed_turn(
        db,
        user_id=user.id,
        knowledge_base_id=kb.id,
        conversation_id=conversation.id,
        user_message_id=user_message.id,
        assistant_message_id=assistant_message.id,
    )

    assert result.created == ()
    assert result.skipped_duplicates == 1
    assert len(memory_service.list_active(
        db,
        user_id=user.id,
        knowledge_base_id=kb.id,
        conversation_id=conversation.id,
    )) == 1


def test_extract_completed_turn_can_resolve_latest_user_for_resume(db) -> None:
    user, kb, conversation, user_message, assistant_message = _create_scope(db)
    service, _, _ = _build_service('{"memories":[]}')

    result = service.extract_completed_turn(
        db,
        user_id=user.id,
        knowledge_base_id=kb.id,
        conversation_id=conversation.id,
        assistant_message_id=assistant_message.id,
        user_message_id=None,
    )

    assert result.created == ()
    assert result.skipped_duplicates == 0
    assert user_message.id < assistant_message.id


def test_memory_extractor_accepts_json_fence_and_deduplicates_candidates(db) -> None:
    _, _, _, user_message, assistant_message = _create_scope(db)
    llm = FakeLLM(
        '''```json
        {"memories":[
            {"memory_type":"goal","content":"完成 v2.4。"},
            {"memory_type":"goal","content":"完成 v2.4。"}
        ]}
        ```'''
    )
    extractor = LLMConversationMemoryExtractor(llm_service=llm)

    candidates, prompt_id, prompt_version = extractor.extract(
        user_message=user_message,
        assistant_message=assistant_message,
    )

    assert len(candidates) == 1
    assert candidates[0].memory_type is MemoryType.GOAL
    assert prompt_id == "agent.conversation-memory-extraction"
    assert prompt_version == "1.0.0"


def test_memory_extractor_rejects_invalid_structured_output(db) -> None:
    _, _, _, user_message, assistant_message = _create_scope(db)
    extractor = LLMConversationMemoryExtractor(
        llm_service=FakeLLM(
            '{"memories":[{"memory_type":"secret","content":"x"}]}'
        )
    )

    with pytest.raises(ConversationMemoryExtractionError):
        extractor.extract(
            user_message=user_message,
            assistant_message=assistant_message,
        )
