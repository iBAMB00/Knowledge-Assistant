from types import SimpleNamespace

from app.agent.agent_prompt import render_agent_tool_calling_system_prompt
from app.agent.context_engine import (
    AgentContextBudgetPolicy,
    AgentContextBuilder,
    AgentContextItem,
    AgentContextRole,
    AgentContextSource,
)
from app.constants.conversation_message_role import ConversationMessageRole
from app.agent.prompts import RenderedPrompt
from app.services.conversation_context_provider import ConversationContextProvider
from app.services.conversation_history_context_provider import (
    ConversationHistoryContextProvider,
)
from app.services.conversation_memory_context_provider import (
    ConversationMemoryContextProvider,
    LexicalConversationMemoryRetriever,
)
from app.services.conversation_summary_service import (
    ConversationSummaryPolicy,
    ConversationSummaryService,
    GeneratedConversationSummary,
)


class _FakeConversationService:
    def __init__(self, messages):
        self.messages = list(messages)

    def ensure_chat_scope(self, db, **kwargs):
        return object()

    def list_messages(self, db, **kwargs):
        return list(self.messages)


class _FakeSummaryService:
    def __init__(self, summary=None):
        self.summary = summary

    def prepare(self, db, *, conversation_id, messages):
        return self.summary

    def get(self, db, *, conversation_id):
        return self.summary


class _FakeMemoryService:
    def __init__(self, memories):
        self.memories = list(memories)
        self.calls = []

    def list_active(self, db, **kwargs):
        self.calls.append(kwargs)
        return list(self.memories)


class _FakeSummaryRepository:
    def __init__(self, existing=None):
        self.existing = existing
        self.saved = None

    def find_by_conversation_id(self, db, *, conversation_id):
        return self.existing

    def save_or_update(
        self,
        db,
        *,
        conversation_id,
        content,
        summarized_through_message_id,
        prompt_id,
        prompt_version,
    ):
        self.saved = SimpleNamespace(
            id=31,
            conversation_id=conversation_id,
            content=content,
            summarized_through_message_id=summarized_through_message_id,
            prompt_id=prompt_id,
            prompt_version=prompt_version,
        )
        self.existing = self.saved
        return self.saved


class _FakeSummaryGenerator:
    def __init__(self):
        self.calls = []

    def generate(self, *, previous_summary, messages):
        self.calls.append((previous_summary, list(messages)))
        return GeneratedConversationSummary(
            content="更新后的稳定摘要",
            prompt_id="agent.conversation-summary",
            prompt_version="1.0.0",
        )


class _FakeDB:
    def commit(self):
        return None

    def rollback(self):
        return None

    def refresh(self, obj):
        return None


def _message(message_id, role, content, *, conversation_id=5):
    return SimpleNamespace(
        id=message_id,
        conversation_id=conversation_id,
        role=role.value,
        content=content,
    )


def _history(content, role):
    return AgentContextItem(
        role=role,
        source=AgentContextSource.CONVERSATION_HISTORY,
        content=content,
    )


def _gate_prompt():
    return RenderedPrompt(
        prompt_id="test.release-gate",
        version="1.0.0",
        content="sys",
    )


def _memory(memory_id, memory_type, content):
    return SimpleNamespace(
        id=memory_id,
        memory_type=memory_type,
        content=content,
    )


def test_release_short_conversation_has_no_context_regression():
    supporting = (
        _history("上一轮问题", AgentContextRole.USER),
        _history("上一轮回答", AgentContextRole.ASSISTANT),
    )
    context = AgentContextBuilder().build(
        system_prompt=render_agent_tool_calling_system_prompt(),
        current_message="当前问题",
        supporting_items=supporting,
    )

    assert context.items[1:3] == supporting
    assert context.items[-1].content == "当前问题"
    assert context.budget is not None
    assert context.budget.truncated is False


def test_release_long_conversation_preserves_summary_memory_and_recent_history():
    messages = [
        _message(1, ConversationMessageRole.USER, "旧问题"),
        _message(2, ConversationMessageRole.ASSISTANT, "旧回答"),
        _message(3, ConversationMessageRole.USER, "最近问题"),
        _message(4, ConversationMessageRole.ASSISTANT, "最近回答"),
        _message(5, ConversationMessageRole.USER, "当前 FastAPI 问题"),
    ]
    summary = SimpleNamespace(
        id=10,
        content="此前已确认项目继续使用 FastAPI。",
        summarized_through_message_id=2,
        prompt_version="1.0.0",
    )
    memory_provider = ConversationMemoryContextProvider(
        memory_service=_FakeMemoryService(
            [_memory(7, "preference", "用户偏好 FastAPI 代码保持简洁。")]
        )
    )
    provider = ConversationContextProvider(
        history_provider=ConversationHistoryContextProvider(
            conversation_service=_FakeConversationService(messages)
        ),
        summary_service=_FakeSummaryService(summary),
        memory_context_provider=memory_provider,
    )

    items = provider.load(
        object(),
        user_id=3,
        conversation_id=5,
        knowledge_base_id=9,
        current_message="当前 FastAPI 问题",
    )

    assert [item.source for item in items] == [
        AgentContextSource.CONVERSATION_SUMMARY,
        AgentContextSource.MEMORY,
        AgentContextSource.CONVERSATION_HISTORY,
        AgentContextSource.CONVERSATION_HISTORY,
    ]
    assert [item.content for item in items[-2:]] == ["最近问题", "最近回答"]


def test_release_history_truncation_keeps_recent_messages():
    supporting = tuple(
        _history(
            f"history-{index}-" + "x" * 32,
            AgentContextRole.USER if index % 2 else AgentContextRole.ASSISTANT,
        )
        for index in range(1, 13)
    )
    context = AgentContextBuilder(
        budget_policy=AgentContextBudgetPolicy(
            max_tokens=120,
            keep_recent_history_items=4,
        )
    ).build(
        system_prompt=_gate_prompt(),
        current_message="now",
        supporting_items=supporting,
    )

    selected_history = context.items_from(AgentContextSource.CONVERSATION_HISTORY)
    assert context.budget is not None and context.budget.truncated is True
    assert selected_history
    assert selected_history[-1].content.startswith("history-12-")
    assert not any(item.content.startswith("history-1-") for item in selected_history)


def test_release_summary_continuity_is_incremental_and_keeps_recent_raw_messages():
    existing = SimpleNamespace(
        id=20,
        content="已有摘要",
        summarized_through_message_id=4,
        prompt_id="agent.conversation-summary",
        prompt_version="1.0.0",
    )
    repository = _FakeSummaryRepository(existing)
    generator = _FakeSummaryGenerator()
    service = ConversationSummaryService(
        repository=repository,
        generator=generator,
        policy=ConversationSummaryPolicy(
            trigger_history_tokens=1,
            refresh_candidate_tokens=1,
            keep_recent_messages=2,
            max_batch_messages=80,
        ),
    )
    messages = [
        _message(index, ConversationMessageRole.USER, f"消息{index}")
        for index in range(1, 11)
    ]

    summary = service.prepare(_FakeDB(), conversation_id=5, messages=messages)

    assert summary.summarized_through_message_id == 8
    previous_summary, batch = generator.calls[0]
    assert previous_summary == "已有摘要"
    assert [message.id for message in batch] == [5, 6, 7, 8]


def test_release_memory_save_recall_baseline_returns_relevant_active_memory():
    retriever = LexicalConversationMemoryRetriever()
    relevant = _memory(1, "preference", "用户偏好简单直观的 Python 代码。")
    unrelated = _memory(2, "goal", "用户计划学习日语。")

    selected = retriever.retrieve(
        query="Python 代码怎么保持简单直观？",
        memories=[relevant, unrelated],
    )

    assert [memory.id for memory in selected] == [1]


def test_release_irrelevant_memory_is_not_recalled_for_unrelated_query():
    retriever = LexicalConversationMemoryRetriever()
    wrong = _memory(1, "fact", "部署端口是 9999。")

    selected = retriever.retrieve(
        query="请解释 Python 装饰器",
        memories=[wrong],
    )

    assert selected == ()


def test_release_enterprise_knowledge_outranks_memory_under_budget_and_prompt_trust():
    memory = AgentContextItem(
        role=AgentContextRole.ASSISTANT,
        source=AgentContextSource.MEMORY,
        content="memory-" + "m" * 20,
    )
    knowledge = AgentContextItem(
        role=AgentContextRole.SYSTEM,
        source=AgentContextSource.KNOWLEDGE,
        content="knowled-" + "k" * 20,
    )
    context = AgentContextBuilder(
        budget_policy=AgentContextBudgetPolicy(
            max_tokens=24,
            keep_recent_history_items=0,
        )
    ).build(
        system_prompt=_gate_prompt(),
        current_message="now",
        supporting_items=(memory, knowledge),
    )

    selected_sources = {item.source for item in context.items}
    assert AgentContextSource.KNOWLEDGE in selected_sources
    assert AgentContextSource.MEMORY not in selected_sources

    system_prompt = render_agent_tool_calling_system_prompt().content
    assert "Memory" in system_prompt
    assert "企业知识" in system_prompt
    assert "冲突" in system_prompt


def test_release_memory_provider_keeps_user_and_conversation_scope_explicit():
    service = _FakeMemoryService(
        [_memory(3, "decision", "项目决定继续使用 FastAPI。")]
    )
    provider = ConversationMemoryContextProvider(memory_service=service)

    provider.load(
        object(),
        user_id=11,
        conversation_id=22,
        knowledge_base_id=33,
        current_message="FastAPI 方案",
    )

    assert service.calls == [
        {
            "user_id": 11,
            "knowledge_base_id": 33,
            "conversation_id": 22,
        }
    ]


def test_release_runtime_parity_uses_framework_neutral_context_contract():
    context = AgentContextBuilder().build(
        system_prompt=render_agent_tool_calling_system_prompt(),
        current_message="当前问题",
        supporting_items=(
            AgentContextItem(
                role=AgentContextRole.ASSISTANT,
                source=AgentContextSource.CONVERSATION_SUMMARY,
                content="摘要",
            ),
            AgentContextItem(
                role=AgentContextRole.ASSISTANT,
                source=AgentContextSource.MEMORY,
                content="Memory",
            ),
            _history("历史", AgentContextRole.USER),
        ),
    )

    # B2 的 AgentContext 是三套 Runtime 的共同输入事实；B10 Gate 固定其稳定顺序。
    assert [item.source for item in context.items] == [
        AgentContextSource.SYSTEM_PROMPT,
        AgentContextSource.CONVERSATION_SUMMARY,
        AgentContextSource.MEMORY,
        AgentContextSource.CONVERSATION_HISTORY,
        AgentContextSource.CURRENT_MESSAGE,
    ]


def test_release_context_growth_stays_bounded_when_history_grows():
    policy = AgentContextBudgetPolicy(
        max_tokens=1_000,
        keep_recent_history_items=8,
    )
    builder = AgentContextBuilder(budget_policy=policy)

    supporting = tuple(
        _history(
            f"turn-{index}-" + "x" * 160,
            AgentContextRole.USER if index % 2 else AgentContextRole.ASSISTANT,
        )
        for index in range(200)
    )
    context = builder.build(
        system_prompt=render_agent_tool_calling_system_prompt(),
        current_message="当前问题",
        supporting_items=supporting,
    )

    assert context.budget is not None
    assert context.budget.input_supporting_items == 200
    assert context.budget.selected_supporting_items < 200
    assert context.budget.estimated_tokens <= context.budget.max_tokens
    assert context.budget.truncated is True
