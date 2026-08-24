from app.agent.context_engine import (
    AgentContextItem,
    AgentContextRole,
    AgentContextSource,
    AgentContextUsage,
)
from app.agent.context_engine.usage import resolve_agent_context_usage
from app.schemas.agent_chat_response import AgentChatResponse


def test_context_usage_reports_only_selected_safe_sources() -> None:
    usage = resolve_agent_context_usage(
        current_message="当前问题",
        supporting_items=(
            AgentContextItem(
                role=AgentContextRole.ASSISTANT,
                source=AgentContextSource.CONVERSATION_SUMMARY,
                content="此前摘要",
            ),
            AgentContextItem(
                role=AgentContextRole.USER,
                source=AgentContextSource.CONVERSATION_HISTORY,
                content="最近历史",
            ),
            AgentContextItem(
                role=AgentContextRole.ASSISTANT,
                source=AgentContextSource.MEMORY,
                content="用户偏好简洁回答",
            ),
        ),
    )

    assert usage == AgentContextUsage(
        history=True,
        summary=True,
        memory=True,
    )


def test_context_usage_does_not_claim_item_dropped_by_budget() -> None:
    usage = resolve_agent_context_usage(
        current_message="当前问题",
        supporting_items=(
            AgentContextItem(
                role=AgentContextRole.USER,
                source=AgentContextSource.CONVERSATION_HISTORY,
                content="旧历史" * 30000,
            ),
            AgentContextItem(
                role=AgentContextRole.ASSISTANT,
                source=AgentContextSource.CONVERSATION_SUMMARY,
                content="精简摘要",
            ),
            AgentContextItem(
                role=AgentContextRole.ASSISTANT,
                source=AgentContextSource.MEMORY,
                content="用户偏好简洁回答",
            ),
        ),
    )

    assert usage.history is False
    assert usage.summary is True
    assert usage.memory is True


def test_public_chat_response_exposes_only_usage_booleans() -> None:
    response = AgentChatResponse(
        answer="回答",
        context_usage=AgentContextUsage(
            history=True,
            summary=False,
            memory=True,
        ),
    )

    assert response.model_dump() == {
        "answer": "回答",
        "context_usage": {
            "history": True,
            "summary": False,
            "memory": True,
        },
    }
