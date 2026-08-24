"""从 B4 最终预算选择结果生成可安全暴露的 Context Usage。"""

from collections.abc import Sequence

from app.agent.agent_prompt import render_agent_tool_calling_system_prompt
from app.agent.context_engine.builder import AgentContextBuilder
from app.agent.context_engine.contracts import (
    AgentContextItem,
    AgentContextSource,
    AgentContextUsage,
)


_CONTEXT_BUILDER = AgentContextBuilder()


def resolve_agent_context_usage(
    *,
    current_message: str,
    supporting_items: Sequence[AgentContextItem] = (),
) -> AgentContextUsage:
    """按与生产 Runtime 相同的 B4 Budget 规则确认真正进入模型的来源。"""

    context = _CONTEXT_BUILDER.build(
        system_prompt=render_agent_tool_calling_system_prompt(),
        current_message=current_message,
        supporting_items=supporting_items,
    )
    selected_sources = {item.source for item in context.items}
    return AgentContextUsage(
        history=AgentContextSource.CONVERSATION_HISTORY in selected_sources,
        summary=AgentContextSource.CONVERSATION_SUMMARY in selected_sources,
        memory=AgentContextSource.MEMORY in selected_sources,
    )
