from types import SimpleNamespace

from app.agent.context_engine import (
    AgentContextRole,
    AgentContextSource,
)
from app.constants.conversation_message_role import ConversationMessageRole
from app.services.conversation_history_context_provider import (
    ConversationHistoryContextProvider,
)
from app.services.llm_service import LLMService


class _FakeConversationService:
    def __init__(self, messages):
        self.messages = messages
        self.calls = []
        self.scope_calls = []

    def ensure_chat_scope(
        self,
        db,
        *,
        user_id,
        conversation_id,
        mode,
        knowledge_base_id,
    ):
        self.scope_calls.append(
            (user_id, conversation_id, mode.value, knowledge_base_id)
        )
        return object()

    def list_messages(self, db, *, user_id, conversation_id, limit=200):
        self.calls.append((user_id, conversation_id, limit))
        return list(self.messages)


def _message(message_id: int, role: ConversationMessageRole, content: str):
    return SimpleNamespace(
        id=message_id,
        role=role.value,
        content=content,
    )


def test_history_provider_maps_owned_messages_and_excludes_current_turn():
    service = _FakeConversationService([
        _message(10, ConversationMessageRole.USER, "第一个问题"),
        _message(11, ConversationMessageRole.ASSISTANT, "第一个回答"),
        _message(12, ConversationMessageRole.USER, "当前问题"),
    ])
    provider = ConversationHistoryContextProvider(
        conversation_service=service,
    )

    items = provider.load(
        object(),
        user_id=7,
        conversation_id=3,
        knowledge_base_id=9,
        current_message="  当前问题  ",
    )

    assert service.scope_calls == [(7, 3, "agent", 9)]
    assert service.calls == [(7, 3, 200)]
    assert [item.role for item in items] == [
        AgentContextRole.USER,
        AgentContextRole.ASSISTANT,
    ]
    assert [item.source for item in items] == [
        AgentContextSource.CONVERSATION_HISTORY,
        AgentContextSource.CONVERSATION_HISTORY,
    ]
    assert [item.content for item in items] == ["第一个问题", "第一个回答"]
    assert [item.source_id for item in items] == [
        "conversation_message:10",
        "conversation_message:11",
    ]


def test_history_provider_only_removes_last_matching_user_message():
    service = _FakeConversationService([
        _message(1, ConversationMessageRole.USER, "重复问题"),
        _message(2, ConversationMessageRole.ASSISTANT, "旧回答"),
        _message(3, ConversationMessageRole.USER, "重复问题"),
    ])
    provider = ConversationHistoryContextProvider(
        conversation_service=service,
    )

    items = provider.load(
        object(),
        user_id=1,
        conversation_id=2,
        knowledge_base_id=4,
        current_message="重复问题",
    )

    assert [item.content for item in items] == ["重复问题", "旧回答"]


def test_history_provider_keeps_stateless_behavior_without_conversation():
    service = _FakeConversationService([])
    provider = ConversationHistoryContextProvider(
        conversation_service=service,
    )

    assert provider.load(
        object(),
        user_id=1,
        conversation_id=None,
        knowledge_base_id=4,
        current_message="hello",
    ) == ()
    assert service.calls == []
    assert service.scope_calls == []


def test_llm_context_orders_history_before_current_message():
    from app.agent.context_engine import AgentContextItem

    history = (
        AgentContextItem(
            role=AgentContextRole.USER,
            source=AgentContextSource.CONVERSATION_HISTORY,
            content="历史问题",
        ),
        AgentContextItem(
            role=AgentContextRole.ASSISTANT,
            source=AgentContextSource.CONVERSATION_HISTORY,
            content="历史回答",
        ),
    )

    messages = LLMService._build_tool_calling_messages(
        "当前问题",
        supporting_context=history,
    )

    assert [message["role"] for message in messages] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert [message["content"] for message in messages[1:]] == [
        "历史问题",
        "历史回答",
        "当前问题",
    ]
