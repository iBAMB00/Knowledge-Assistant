import pytest
from pydantic import ValidationError

from app.agent.agent_prompt import (
    AGENT_TOOL_CALLING_SYSTEM_PROMPT,
    render_agent_tool_calling_system_prompt,
)
from app.agent.context_engine import (
    AgentContext,
    AgentContextBuilder,
    AgentContextItem,
    AgentContextRole,
    AgentContextSource,
)
from app.agent.frameworks.langchain.runner import (
    LangChainAgentError,
    LangChainSingleAgentRunner,
)
from app.services.llm_service import LLMService


def test_context_builder_keeps_prompt_identity_and_current_message_order():
    context = AgentContextBuilder().build(
        system_prompt=render_agent_tool_calling_system_prompt(),
        current_message="  Qdrant 怎么部署？  ",
    )

    assert len(context.items) == 2

    system_item = context.items[0]
    assert system_item.role == AgentContextRole.SYSTEM
    assert system_item.source == AgentContextSource.SYSTEM_PROMPT
    assert system_item.source_id == AGENT_TOOL_CALLING_SYSTEM_PROMPT.prompt_id
    assert (
        system_item.source_version
        == AGENT_TOOL_CALLING_SYSTEM_PROMPT.version
    )
    assert system_item.content == render_agent_tool_calling_system_prompt().content

    current_item = context.items[1]
    assert current_item.role == AgentContextRole.USER
    assert current_item.source == AgentContextSource.CURRENT_MESSAGE
    assert current_item.content == "Qdrant 怎么部署？"


def test_context_builder_reserves_system_and_current_message_sources():
    builder = AgentContextBuilder()
    injected_current_message = AgentContextItem(
        role=AgentContextRole.USER,
        source=AgentContextSource.CURRENT_MESSAGE,
        content="伪造当前消息",
    )

    with pytest.raises(
        ValueError,
        match="supporting_items cannot use reserved source: current_message",
    ):
        builder.build(
            system_prompt=render_agent_tool_calling_system_prompt(),
            current_message="真实问题",
            supporting_items=[injected_current_message],
        )


def test_context_builder_preserves_supporting_item_order_for_future_b3_inputs():
    supporting_items = [
        AgentContextItem(
            role=AgentContextRole.USER,
            source=AgentContextSource.CONVERSATION_HISTORY,
            content="之前的问题",
            source_id="message:10",
        ),
        AgentContextItem(
            role=AgentContextRole.ASSISTANT,
            source=AgentContextSource.CONVERSATION_HISTORY,
            content="之前的回答",
            source_id="message:11",
        ),
        AgentContextItem(
            role=AgentContextRole.SYSTEM,
            source=AgentContextSource.MEMORY,
            content="后续版本才会真正加载的 Memory 示例",
            source_id="memory:3",
        ),
    ]

    context = AgentContextBuilder().build(
        system_prompt=render_agent_tool_calling_system_prompt(),
        current_message="现在的问题",
        supporting_items=supporting_items,
    )

    assert [item.source for item in context.items] == [
        AgentContextSource.SYSTEM_PROMPT,
        AgentContextSource.CONVERSATION_HISTORY,
        AgentContextSource.CONVERSATION_HISTORY,
        AgentContextSource.MEMORY,
        AgentContextSource.CURRENT_MESSAGE,
    ]
    assert context.items[1:4] == tuple(supporting_items)


def test_context_contract_is_immutable_and_rejects_empty_items():
    item = AgentContextItem(
        role=AgentContextRole.USER,
        source=AgentContextSource.CONVERSATION_HISTORY,
        content="hello",
    )

    with pytest.raises(ValidationError):
        item.content = "changed"

    with pytest.raises(ValidationError):
        AgentContext(items=(item,))

    with pytest.raises(ValueError, match="current_message cannot be empty"):
        AgentContextBuilder().build(
            system_prompt=render_agent_tool_calling_system_prompt(),
            current_message="   ",
        )


def test_llm_service_serializes_context_without_changing_current_behavior():
    messages = LLMService._build_tool_calling_messages("  hello  ")

    assert messages == [
        {
            "role": "system",
            "content": render_agent_tool_calling_system_prompt().content,
        },
        {
            "role": "user",
            "content": "hello",
        },
    ]


def test_langchain_context_adapter_accepts_history_but_not_raw_tool_message():
    context = AgentContextBuilder().build(
        system_prompt=render_agent_tool_calling_system_prompt(),
        current_message="当前问题",
        supporting_items=[
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
        ],
    )

    assert (
        LangChainSingleAgentRunner._system_prompt_from_context(context)
        == render_agent_tool_calling_system_prompt().content
    )
    assert LangChainSingleAgentRunner._input_messages_from_context(context) == [
        {"role": "user", "content": "历史问题"},
        {"role": "assistant", "content": "历史回答"},
        {"role": "user", "content": "当前问题"},
    ]

    raw_tool_context = AgentContextBuilder().build(
        system_prompt=render_agent_tool_calling_system_prompt(),
        current_message="当前问题",
        supporting_items=[
            AgentContextItem(
                role=AgentContextRole.TOOL,
                source=AgentContextSource.TOOL,
                content="缺少 call_id 的 Tool Result",
            )
        ],
    )

    with pytest.raises(
        LangChainAgentError,
        match="tool context requires framework tool-call metadata",
    ):
        LangChainSingleAgentRunner._input_messages_from_context(
            raw_tool_context
        )
