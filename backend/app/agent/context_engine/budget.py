"""Agent Context 的轻量预算与裁剪策略。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import ceil

from app.agent.context_engine.contracts import (
    AgentContextBudget,
    AgentContextItem,
    AgentContextSource,
)


@dataclass(frozen=True)
class AgentContextBudgetPolicy:
    """B4 第一版预算策略。

    ``max_tokens`` 是应用侧输入上下文软预算。System Prompt 和当前用户消息
    属于必须上下文，不会为了满足预算而静默截断；当二者本身已经超预算时，
    supporting context 会全部淘汰，并在 budget metadata 中标记。
    """

    max_tokens: int = 12_000
    keep_recent_history_items: int = 8

    def __post_init__(self) -> None:
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be greater than 0")
        if self.keep_recent_history_items < 0:
            raise ValueError("keep_recent_history_items cannot be negative")


class ApproximateTokenEstimator:
    """不绑定特定模型 tokenizer 的稳定近似估算器。

    第一版使用 UTF-8 byte / 4 近似文本 token，并为每条 message 计入固定
    framing overhead。它的目标是提供稳定、可测试的预算边界，不冒充 provider
    的精确 tokenizer；未来真实模型差异出现时可替换 estimator，而不用改 Builder。
    """

    VERSION = "utf8-bytes-v1"
    MESSAGE_OVERHEAD_TOKENS = 4

    def estimate_text(self, content: str) -> int:
        normalized = content.strip()
        if not normalized:
            return 0
        return max(1, ceil(len(normalized.encode("utf-8")) / 4))

    def estimate_item(self, item: AgentContextItem) -> int:
        return self.estimate_text(item.content) + self.MESSAGE_OVERHEAD_TOKENS


@dataclass(frozen=True)
class _BudgetCandidate:
    index: int
    item: AgentContextItem
    tokens: int
    priority: int


class AgentContextBudgetManager:
    """在不改变 Context Source 语义的前提下选择 supporting context。"""

    _SOURCE_PRIORITY = {
        AgentContextSource.CONVERSATION_SUMMARY: 600,
        AgentContextSource.KNOWLEDGE: 450,
        AgentContextSource.MEMORY: 400,
        AgentContextSource.TOOL: 300,
        AgentContextSource.CONVERSATION_HISTORY: 100,
    }
    _RECENT_HISTORY_PRIORITY = 500

    def __init__(
        self,
        *,
        policy: AgentContextBudgetPolicy | None = None,
        estimator: ApproximateTokenEstimator | None = None,
    ) -> None:
        self.policy = policy or AgentContextBudgetPolicy()
        self.estimator = estimator or ApproximateTokenEstimator()

    def apply(
        self,
        *,
        required_items: Sequence[AgentContextItem],
        supporting_items: Sequence[AgentContextItem],
    ) -> tuple[tuple[AgentContextItem, ...], AgentContextBudget]:
        """返回预算内 supporting items，并生成可解释预算元数据。"""

        normalized_required = tuple(required_items)
        normalized_supporting = tuple(supporting_items)

        required_tokens = sum(
            self.estimator.estimate_item(item)
            for item in normalized_required
        )
        input_supporting_tokens = sum(
            self.estimator.estimate_item(item)
            for item in normalized_supporting
        )

        available_tokens = max(0, self.policy.max_tokens - required_tokens)

        if input_supporting_tokens <= available_tokens:
            selected = normalized_supporting
        else:
            selected = self._select_supporting_items(
                normalized_supporting,
                available_tokens=available_tokens,
            )

        selected_supporting_tokens = sum(
            self.estimator.estimate_item(item)
            for item in selected
        )
        dropped_count = len(normalized_supporting) - len(selected)
        estimated_tokens = required_tokens + selected_supporting_tokens

        budget = AgentContextBudget(
            max_tokens=self.policy.max_tokens,
            estimated_tokens=estimated_tokens,
            required_tokens=required_tokens,
            input_supporting_tokens=input_supporting_tokens,
            selected_supporting_tokens=selected_supporting_tokens,
            input_supporting_items=len(normalized_supporting),
            selected_supporting_items=len(selected),
            dropped_supporting_items=dropped_count,
            truncated=dropped_count > 0,
            required_over_budget=required_tokens > self.policy.max_tokens,
            estimator_version=self.estimator.VERSION,
        )
        return selected, budget

    def _select_supporting_items(
        self,
        items: tuple[AgentContextItem, ...],
        *,
        available_tokens: int,
    ) -> tuple[AgentContextItem, ...]:
        if available_tokens <= 0 or not items:
            return ()

        history_indices = [
            index
            for index, item in enumerate(items)
            if item.source == AgentContextSource.CONVERSATION_HISTORY
        ]
        if self.policy.keep_recent_history_items == 0:
            recent_history_indices: set[int] = set()
        else:
            recent_history_indices = set(
                history_indices[-self.policy.keep_recent_history_items :]
            )

        candidates: list[_BudgetCandidate] = []
        for index, item in enumerate(items):
            priority = self._SOURCE_PRIORITY.get(item.source, 0)
            if index in recent_history_indices:
                priority = self._RECENT_HISTORY_PRIORITY

            candidates.append(
                _BudgetCandidate(
                    index=index,
                    item=item,
                    tokens=self.estimator.estimate_item(item),
                    priority=priority,
                )
            )

        # 高优先级先选；同为 Conversation History 时优先更近的消息。
        candidates.sort(
            key=lambda candidate: (
                -candidate.priority,
                -candidate.index
                if candidate.item.source
                == AgentContextSource.CONVERSATION_HISTORY
                else candidate.index,
            )
        )

        selected_indices: set[int] = set()
        remaining_tokens = available_tokens
        for candidate in candidates:
            if candidate.tokens > remaining_tokens:
                continue
            selected_indices.add(candidate.index)
            remaining_tokens -= candidate.tokens

        # 最终模型输入仍保持 Provider 给出的原始时间/语义顺序。
        return tuple(
            item
            for index, item in enumerate(items)
            if index in selected_indices
        )
