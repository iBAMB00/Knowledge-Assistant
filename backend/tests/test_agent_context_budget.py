from app.agent.context_engine import (
    AgentContextBudgetPolicy,
    AgentContextBuilder,
    AgentContextItem,
    AgentContextRole,
    AgentContextSource,
    ApproximateTokenEstimator,
)
from app.agent.prompts import RenderedPrompt


def _prompt(content: str = "sys") -> RenderedPrompt:
    return RenderedPrompt(
        prompt_id="test.context-budget",
        version="1.0.0",
        content=content,
    )


def _history(content: str, *, role: AgentContextRole) -> AgentContextItem:
    return AgentContextItem(
        role=role,
        source=AgentContextSource.CONVERSATION_HISTORY,
        content=content,
    )


def test_token_estimator_is_stable_and_provider_neutral():
    estimator = ApproximateTokenEstimator()

    assert estimator.estimate_text("abcd") == 1
    assert estimator.estimate_text("abcdefgh") == 2
    assert estimator.estimate_text("你好") == 2
    assert estimator.estimate_text("   ") == 0
    assert estimator.VERSION == "utf8-bytes-v1"


def test_context_budget_keeps_all_items_when_under_budget():
    supporting = [
        _history("old-user", role=AgentContextRole.USER),
        _history("old-answer", role=AgentContextRole.ASSISTANT),
    ]

    context = AgentContextBuilder(
        budget_policy=AgentContextBudgetPolicy(max_tokens=100),
    ).build(
        system_prompt=_prompt(),
        current_message="now",
        supporting_items=supporting,
    )

    assert context.items[1:3] == tuple(supporting)
    assert context.budget is not None
    assert context.budget.truncated is False
    assert context.budget.dropped_supporting_items == 0
    assert context.budget.selected_supporting_items == 2
    assert context.budget.estimated_tokens <= context.budget.max_tokens


def test_context_budget_drops_old_history_and_keeps_recent_history_in_order():
    supporting = [
        _history("history1", role=AgentContextRole.USER),
        _history("history2", role=AgentContextRole.ASSISTANT),
        _history("history3", role=AgentContextRole.USER),
        _history("history4", role=AgentContextRole.ASSISTANT),
    ]

    # system/current 各约 5 token，剩余 12；每条 history 约 6 token。
    context = AgentContextBuilder(
        budget_policy=AgentContextBudgetPolicy(
            max_tokens=22,
            keep_recent_history_items=2,
        ),
    ).build(
        system_prompt=_prompt(),
        current_message="now",
        supporting_items=supporting,
    )

    assert [item.content for item in context.items] == [
        "sys",
        "history3",
        "history4",
        "now",
    ]
    assert context.budget is not None
    assert context.budget.truncated is True
    assert context.budget.input_supporting_items == 4
    assert context.budget.selected_supporting_items == 2
    assert context.budget.dropped_supporting_items == 2
    assert context.budget.estimated_tokens <= context.budget.max_tokens


def test_context_budget_uses_source_priority_before_old_raw_history():
    supporting = [
        _history("history1", role=AgentContextRole.USER),
        _history("history2", role=AgentContextRole.ASSISTANT),
        AgentContextItem(
            role=AgentContextRole.SYSTEM,
            source=AgentContextSource.CONVERSATION_SUMMARY,
            content="summary1",
        ),
        _history("history4", role=AgentContextRole.ASSISTANT),
    ]

    context = AgentContextBuilder(
        budget_policy=AgentContextBudgetPolicy(
            max_tokens=22,
            keep_recent_history_items=1,
        ),
    ).build(
        system_prompt=_prompt(),
        current_message="now",
        supporting_items=supporting,
    )

    assert [item.content for item in context.items] == [
        "sys",
        "summary1",
        "history4",
        "now",
    ]


def test_context_budget_never_silently_truncates_system_or_current_message():
    supporting = [
        _history("history", role=AgentContextRole.USER),
    ]

    context = AgentContextBuilder(
        budget_policy=AgentContextBudgetPolicy(max_tokens=5),
    ).build(
        system_prompt=_prompt("system-required"),
        current_message="current-required",
        supporting_items=supporting,
    )

    assert [item.source for item in context.items] == [
        AgentContextSource.SYSTEM_PROMPT,
        AgentContextSource.CURRENT_MESSAGE,
    ]
    assert context.budget is not None
    assert context.budget.required_over_budget is True
    assert context.budget.truncated is True
    assert context.budget.dropped_supporting_items == 1
    assert context.budget.estimated_tokens > context.budget.max_tokens


def test_zero_recent_history_setting_does_not_promote_all_history():
    supporting = [
        _history("history1", role=AgentContextRole.USER),
        AgentContextItem(
            role=AgentContextRole.SYSTEM,
            source=AgentContextSource.MEMORY,
            content="memory11",
        ),
        _history("history3", role=AgentContextRole.ASSISTANT),
    ]

    context = AgentContextBuilder(
        budget_policy=AgentContextBudgetPolicy(
            max_tokens=16,
            keep_recent_history_items=0,
        ),
    ).build(
        system_prompt=_prompt(),
        current_message="now",
        supporting_items=supporting,
    )

    # 只够一个 supporting item 时，Memory 应优先于 old raw history。
    assert [item.content for item in context.items] == [
        "sys",
        "memory11",
        "now",
    ]


def test_context_budget_keeps_enterprise_knowledge_before_memory():
    supporting = [
        AgentContextItem(
            role=AgentContextRole.ASSISTANT,
            source=AgentContextSource.MEMORY,
            content="memoryxx",
        ),
        AgentContextItem(
            role=AgentContextRole.SYSTEM,
            source=AgentContextSource.KNOWLEDGE,
            content="knowledg",
        ),
    ]

    context = AgentContextBuilder(
        budget_policy=AgentContextBudgetPolicy(
            max_tokens=16,
            keep_recent_history_items=0,
        ),
    ).build(
        system_prompt=_prompt(),
        current_message="now",
        supporting_items=supporting,
    )

    assert [item.content for item in context.items] == [
        "sys",
        "knowledg",
        "now",
    ]
