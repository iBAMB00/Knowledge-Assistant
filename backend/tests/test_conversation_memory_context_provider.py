from types import SimpleNamespace

from app.agent.context_engine import AgentContextRole, AgentContextSource
from app.services.conversation_memory_context_provider import (
    ConversationMemoryContextProvider,
    ConversationMemoryRetrievalPolicy,
    LexicalConversationMemoryRetriever,
)


class _FakeMemoryService:
    def __init__(self, memories):
        self.memories = list(memories)
        self.calls = []

    def list_active(self, db, **kwargs):
        self.calls.append(kwargs)
        return list(self.memories)


def _memory(memory_id: int, memory_type: str, content: str):
    return SimpleNamespace(
        id=memory_id,
        memory_type=memory_type,
        content=content,
    )


def test_lexical_memory_retriever_returns_only_relevant_memories():
    memories = [
        _memory(1, "preference", "用户偏好简单直观的代码实现。"),
        _memory(2, "fact", "Knowledge Assistant 后端使用 Python 和 FastAPI。"),
        _memory(3, "goal", "用户计划继续学习日语。"),
    ]

    selected = LexicalConversationMemoryRetriever().retrieve(
        query="Python 后端代码应该怎么写得更直观？",
        memories=memories,
    )

    assert [item.id for item in selected] == [2, 1]
    assert 3 not in [item.id for item in selected]


def test_lexical_memory_retriever_respects_max_results_and_newer_tie_break():
    retriever = LexicalConversationMemoryRetriever(
        policy=ConversationMemoryRetrievalPolicy(max_results=2),
    )
    memories = [
        _memory(1, "fact", "项目使用 FastAPI。"),
        _memory(2, "decision", "决定继续使用 FastAPI。"),
        _memory(3, "constraint", "FastAPI 接口不能破坏现有 Contract。"),
    ]

    selected = retriever.retrieve(
        query="FastAPI",
        memories=memories,
    )

    assert len(selected) == 2
    assert [item.id for item in selected] == [3, 2]


def test_memory_context_provider_converts_relevant_active_memories():
    service = _FakeMemoryService(
        [
            _memory(7, "preference", "用户偏好直观易读的代码。"),
            _memory(8, "goal", "用户计划学习日语。"),
        ]
    )
    provider = ConversationMemoryContextProvider(
        memory_service=service,
    )

    items = provider.load(
        object(),
        user_id=3,
        conversation_id=5,
        knowledge_base_id=9,
        current_message="代码能不能保持直观易读？",
    )

    assert len(items) == 1
    item = items[0]
    assert item.source == AgentContextSource.MEMORY
    assert item.role == AgentContextRole.ASSISTANT
    assert item.source_id == "memory:7"
    assert item.source_version == "conversation-memory-lexical-v1"
    assert "不是企业知识或系统指令" in item.content
    assert "用户偏好直观易读的代码" in item.content
    assert service.calls == [
        {
            "user_id": 3,
            "knowledge_base_id": 9,
            "conversation_id": 5,
        }
    ]


def test_memory_context_provider_without_conversation_does_not_query_memory():
    service = _FakeMemoryService([])
    provider = ConversationMemoryContextProvider(memory_service=service)

    assert provider.load(
        object(),
        user_id=3,
        conversation_id=None,
        knowledge_base_id=9,
        current_message="hello",
    ) == ()
    assert service.calls == []
