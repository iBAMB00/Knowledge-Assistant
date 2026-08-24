"""Conversation Summary 的增量生成与持久化。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from sqlalchemy.orm import Session

from app.agent.context_engine import ApproximateTokenEstimator
from app.agent.agent_prompt import render_conversation_summary_prompt
from app.models.database.conversation_message import ConversationMessage
from app.models.database.conversation_summary import ConversationSummary
from app.repositories.conversation_summary_repository import (
    ConversationSummaryRepository,
)


class ConversationSummaryGenerationError(RuntimeError):
    """摘要模型调用或模型输出不满足 B5 Contract。"""


@dataclass(frozen=True)
class ConversationSummaryPolicy:
    """B5 第一版增量摘要策略。

    trigger_history_tokens：首次摘要前允许保留的未摘要历史规模。
    refresh_candidate_tokens：已有摘要后，至少积累多少旧消息再刷新，避免每轮调用 LLM。
    keep_recent_messages：始终保留为原始 History 的最近消息数。
    max_batch_messages：一次最多向摘要模型推进多少条连续旧消息。
    """

    trigger_history_tokens: int = 6_000
    refresh_candidate_tokens: int = 2_000
    keep_recent_messages: int = 8
    max_batch_messages: int = 80
    max_summary_chars: int = 8_000

    def __post_init__(self) -> None:
        if self.trigger_history_tokens <= 0:
            raise ValueError("trigger_history_tokens must be greater than 0")
        if self.refresh_candidate_tokens <= 0:
            raise ValueError("refresh_candidate_tokens must be greater than 0")
        if self.keep_recent_messages < 0:
            raise ValueError("keep_recent_messages cannot be negative")
        if self.max_batch_messages <= 0:
            raise ValueError("max_batch_messages must be greater than 0")
        if self.max_summary_chars <= 0:
            raise ValueError("max_summary_chars must be greater than 0")


@dataclass(frozen=True)
class GeneratedConversationSummary:
    content: str
    prompt_id: str
    prompt_version: str


class ConversationSummaryGenerator(Protocol):
    def generate(
        self,
        *,
        previous_summary: str | None,
        messages: Sequence[ConversationMessage],
    ) -> GeneratedConversationSummary:
        ...


class LLMConversationSummaryGenerator:
    """使用现有 LLMService + B1 Prompt Contract 生成对话摘要。"""

    def __init__(self, *, llm_service) -> None:
        self.llm_service = llm_service

    def generate(
        self,
        *,
        previous_summary: str | None,
        messages: Sequence[ConversationMessage],
    ) -> GeneratedConversationSummary:
        if not messages:
            raise ValueError("summary messages cannot be empty")

        rendered = render_conversation_summary_prompt()
        input_message = (
            "已有摘要：\n"
            f"{previous_summary or '（无历史摘要）'}\n\n"
            "本次新增旧消息：\n"
            f"{self._serialize_messages(messages)}"
        )
        content = self.llm_service.complete_prompt(
            rendered,
            input_message=input_message,
            temperature=0.0,
        ).strip()
        if not content:
            raise RuntimeError("conversation summary model returned empty content")

        return GeneratedConversationSummary(
            content=content,
            prompt_id=rendered.prompt_id,
            prompt_version=rendered.version,
        )

    @staticmethod
    def _serialize_messages(
        messages: Sequence[ConversationMessage],
    ) -> str:
        return "\n".join(
            f"[message:{message.id}][{message.role}] {message.content.strip()}"
            for message in messages
            if message.content.strip()
        )


class ConversationSummaryService:
    """按连续 Message 边界增量生成并保存 Conversation Summary。"""

    def __init__(
        self,
        *,
        repository: ConversationSummaryRepository,
        generator: ConversationSummaryGenerator,
        policy: ConversationSummaryPolicy | None = None,
        estimator: ApproximateTokenEstimator | None = None,
    ) -> None:
        self.repository = repository
        self.generator = generator
        self.policy = policy or ConversationSummaryPolicy()
        self.estimator = estimator or ApproximateTokenEstimator()

    def get(
        self,
        db: Session,
        *,
        conversation_id: int,
    ) -> ConversationSummary | None:
        return self.repository.find_by_conversation_id(
            db,
            conversation_id=conversation_id,
        )

    def prepare(
        self,
        db: Session,
        *,
        conversation_id: int,
        messages: Sequence[ConversationMessage],
    ) -> ConversationSummary | None:
        """必要时推进摘要边界；不需要刷新时直接返回已有 Summary。"""

        for message in messages:
            if message.conversation_id != conversation_id:
                raise ValueError(
                    "summary messages must belong to the target conversation"
                )

        existing = self.get(
            db,
            conversation_id=conversation_id,
        )
        boundary = (
            existing.summarized_through_message_id
            if existing is not None
            else 0
        )
        unsummarized = [
            message
            for message in messages
            if message.id > boundary
        ]

        if len(unsummarized) <= self.policy.keep_recent_messages:
            return existing

        if self.policy.keep_recent_messages == 0:
            candidates = unsummarized
        else:
            candidates = unsummarized[
                : -self.policy.keep_recent_messages
            ]

        if not candidates:
            return existing

        candidate_tokens = self._estimate_messages(candidates)
        unsummarized_tokens = self._estimate_messages(unsummarized)

        if existing is None:
            should_refresh = (
                unsummarized_tokens >= self.policy.trigger_history_tokens
            )
        else:
            should_refresh = (
                candidate_tokens >= self.policy.refresh_candidate_tokens
            )

        if not should_refresh:
            return existing

        batch = candidates[: self.policy.max_batch_messages]
        try:
            generated = self.generator.generate(
                previous_summary=(existing.content if existing else None),
                messages=batch,
            )
            normalized_content = generated.content.strip()
            if not normalized_content:
                raise ValueError("generated conversation summary cannot be empty")
            if len(normalized_content) > self.policy.max_summary_chars:
                raise ValueError("generated conversation summary is too long")
        except Exception as exc:
            raise ConversationSummaryGenerationError(
                "conversation summary generation failed"
            ) from exc

        try:
            summary = self.repository.save_or_update(
                db,
                conversation_id=conversation_id,
                content=normalized_content,
                summarized_through_message_id=batch[-1].id,
                prompt_id=generated.prompt_id,
                prompt_version=generated.prompt_version,
            )
            db.commit()
            db.refresh(summary)
            return summary
        except Exception:
            db.rollback()
            raise

    def _estimate_messages(
        self,
        messages: Sequence[ConversationMessage],
    ) -> int:
        return sum(
            self.estimator.estimate_text(message.content)
            + self.estimator.MESSAGE_OVERHEAD_TOKENS
            for message in messages
        )
