from types import SimpleNamespace

from app.agent.context_engine import AgentContextRole, AgentContextSource
from app.constants.conversation_message_role import ConversationMessageRole
from app.services.conversation_context_provider import ConversationContextProvider
from app.services.conversation_summary_service import (
    ConversationSummaryGenerationError,
)
from app.services.conversation_history_context_provider import (
    ConversationHistoryContextProvider,
)


class _FakeConversationService:
    def __init__(self, messages):
        self.messages = messages

    def ensure_chat_scope(self, db, **kwargs):
        return object()

    def list_messages(self, db, **kwargs):
        return list(self.messages)


class _FakeSummaryService:
    def __init__(self, summary=None, error=None):
        self.summary = summary
        self.error = error
        self.prepare_calls = []

    def prepare(self, db, *, conversation_id, messages):
        self.prepare_calls.append((conversation_id, list(messages)))
        if self.error is not None:
            raise self.error
        return self.summary

    def get(self, db, *, conversation_id):
        return self.summary


def _message(message_id: int, role: ConversationMessageRole, content: str):
    return SimpleNamespace(
        id=message_id,
        conversation_id=5,
        role=role.value,
        content=content,
    )


def _summary(boundary: int = 2):
    return SimpleNamespace(
        id=30,
        content="用户正在开发 Knowledge Assistant。",
        summarized_through_message_id=boundary,
        prompt_version="1.0.0",
    )


def test_context_provider_injects_summary_then_only_unsummarized_history():
    messages = [
        _message(1, ConversationMessageRole.USER, "旧问题"),
        _message(2, ConversationMessageRole.ASSISTANT, "旧回答"),
        _message(3, ConversationMessageRole.USER, "最近问题"),
        _message(4, ConversationMessageRole.ASSISTANT, "最近回答"),
        _message(5, ConversationMessageRole.USER, "当前问题"),
    ]
    history_provider = ConversationHistoryContextProvider(
        conversation_service=_FakeConversationService(messages),
    )
    provider = ConversationContextProvider(
        history_provider=history_provider,
        summary_service=_FakeSummaryService(_summary(boundary=2)),
    )

    items = provider.load(
        object(),
        user_id=7,
        conversation_id=5,
        knowledge_base_id=9,
        current_message="当前问题",
    )

    assert [item.source for item in items] == [
        AgentContextSource.CONVERSATION_SUMMARY,
        AgentContextSource.CONVERSATION_HISTORY,
        AgentContextSource.CONVERSATION_HISTORY,
    ]
    assert items[0].role == AgentContextRole.ASSISTANT
    assert items[0].source_id == "conversation_summary:30"
    assert [item.content for item in items[1:]] == ["最近问题", "最近回答"]


def test_context_provider_fails_open_to_existing_summary_when_refresh_fails():
    messages = [
        _message(1, ConversationMessageRole.USER, "旧问题"),
        _message(2, ConversationMessageRole.ASSISTANT, "旧回答"),
        _message(3, ConversationMessageRole.USER, "最近问题"),
        _message(4, ConversationMessageRole.USER, "当前问题"),
    ]
    summary_service = _FakeSummaryService(
        _summary(boundary=2),
        error=ConversationSummaryGenerationError("provider unavailable"),
    )
    provider = ConversationContextProvider(
        history_provider=ConversationHistoryContextProvider(
            conversation_service=_FakeConversationService(messages),
        ),
        summary_service=summary_service,
    )

    items = provider.load(
        object(),
        user_id=7,
        conversation_id=5,
        knowledge_base_id=9,
        current_message="当前问题",
    )

    assert items[0].source == AgentContextSource.CONVERSATION_SUMMARY
    assert [item.content for item in items[1:]] == ["最近问题"]


def test_context_provider_without_summary_keeps_all_previous_raw_history():
    messages = [
        _message(1, ConversationMessageRole.USER, "问题"),
        _message(2, ConversationMessageRole.ASSISTANT, "回答"),
        _message(3, ConversationMessageRole.USER, "当前问题"),
    ]
    provider = ConversationContextProvider(
        history_provider=ConversationHistoryContextProvider(
            conversation_service=_FakeConversationService(messages),
        ),
        summary_service=_FakeSummaryService(None),
    )

    items = provider.load(
        object(),
        user_id=7,
        conversation_id=5,
        knowledge_base_id=9,
        current_message="当前问题",
    )

    assert [item.source for item in items] == [
        AgentContextSource.CONVERSATION_HISTORY,
        AgentContextSource.CONVERSATION_HISTORY,
    ]
    assert [item.content for item in items] == ["问题", "回答"]
