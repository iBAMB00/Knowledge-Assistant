"""统一 Agent Context Builder。"""

from collections.abc import Sequence

from app.agent.context_engine.contracts import (
    AgentContext,
    AgentContextItem,
    AgentContextRole,
    AgentContextSource,
)
from app.agent.prompts import RenderedPrompt


class AgentContextBuilder:
    """把 Prompt、支持上下文和当前消息组装成稳定的有序上下文。"""

    _RESERVED_SOURCES = {
        AgentContextSource.SYSTEM_PROMPT,
        AgentContextSource.CURRENT_MESSAGE,
    }

    def build(
        self,
        *,
        system_prompt: RenderedPrompt,
        current_message: str,
        supporting_items: Sequence[AgentContextItem] = (),
    ) -> AgentContext:
        """构建一次模型调用的基础上下文。

        B2 只真正接入 System Prompt + Current Message；supporting_items
        是 B3 以后 Conversation History / Summary / Memory / Knowledge 的
        统一扩展口，不在本阶段主动加载任何持久化数据。
        """

        normalized_message = current_message.strip()
        if not normalized_message:
            raise ValueError("current_message cannot be empty")

        normalized_supporting_items = tuple(supporting_items)
        for item in normalized_supporting_items:
            if item.source in self._RESERVED_SOURCES:
                raise ValueError(
                    "supporting_items cannot use reserved source: "
                    f"{item.source.value}"
                )

        system_item = AgentContextItem(
            role=AgentContextRole.SYSTEM,
            source=AgentContextSource.SYSTEM_PROMPT,
            content=system_prompt.content,
            source_id=system_prompt.prompt_id,
            source_version=system_prompt.version,
        )
        current_item = AgentContextItem(
            role=AgentContextRole.USER,
            source=AgentContextSource.CURRENT_MESSAGE,
            content=normalized_message,
        )

        return AgentContext(
            items=(
                system_item,
                *normalized_supporting_items,
                current_item,
            )
        )
