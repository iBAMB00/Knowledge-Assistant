"""统一 Agent Context Builder。"""

import logging
from collections.abc import Sequence

from app.agent.context_engine.budget import (
    AgentContextBudgetManager,
    AgentContextBudgetPolicy,
    ApproximateTokenEstimator,
)
from app.agent.context_engine.contracts import (
    AgentContext,
    AgentContextItem,
    AgentContextRole,
    AgentContextSource,
)
from app.agent.prompts import RenderedPrompt


logger = logging.getLogger(__name__)


class AgentContextBuilder:
    """把 Prompt、支持上下文和当前消息组装成稳定的有序上下文。"""

    _RESERVED_SOURCES = {
        AgentContextSource.SYSTEM_PROMPT,
        AgentContextSource.CURRENT_MESSAGE,
    }

    def __init__(
        self,
        *,
        budget_policy: AgentContextBudgetPolicy | None = None,
        token_estimator: ApproximateTokenEstimator | None = None,
    ) -> None:
        self._budget_manager = AgentContextBudgetManager(
            policy=budget_policy,
            estimator=token_estimator,
        )

    def build(
        self,
        *,
        system_prompt: RenderedPrompt,
        current_message: str,
        supporting_items: Sequence[AgentContextItem] = (),
    ) -> AgentContext:
        """构建一次模型调用的预算受控上下文。

        System Prompt 与 Current Message 是 Builder 保留且必须保留的边界；
        Conversation History / Summary / Memory / Knowledge 等 supporting context
        统一进入 B4 Budget Manager。预算不足时优先淘汰旧 Raw History，并保持
        最终被选中内容的原始顺序。
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

        selected_supporting_items, budget = self._budget_manager.apply(
            required_items=(system_item, current_item),
            supporting_items=normalized_supporting_items,
        )

        if budget.truncated or budget.required_over_budget:
            logger.info(
                "Agent context budget applied: max_tokens=%d "
                "estimated_tokens=%d input_supporting_items=%d "
                "selected_supporting_items=%d dropped_supporting_items=%d "
                "required_over_budget=%s estimator=%s",
                budget.max_tokens,
                budget.estimated_tokens,
                budget.input_supporting_items,
                budget.selected_supporting_items,
                budget.dropped_supporting_items,
                budget.required_over_budget,
                budget.estimator_version,
            )

        return AgentContext(
            items=(
                system_item,
                *selected_supporting_items,
                current_item,
            ),
            budget=budget,
        )
