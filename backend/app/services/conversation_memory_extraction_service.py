"""B7 Conversation-derived Memory Extraction。"""

import json
from dataclasses import dataclass
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from sqlalchemy.orm import Session

from app.agent.agent_prompt import render_conversation_memory_extraction_prompt
from app.constants.conversation_message_role import ConversationMessageRole
from app.constants.memory_type import MemoryType
from app.models.database.conversation_message import ConversationMessage
from app.models.database.memory_item import MemoryItem
from app.repositories.conversation_message_repository import (
    ConversationMessageRepository,
)
from app.services.conversation_memory_service import ConversationMemoryService


class ConversationMemoryExtractionError(RuntimeError):
    """Memory 抽取模型输出无效或抽取调用失败。"""


class MemoryExtractionLLM(Protocol):
    def complete_prompt(
        self,
        prompt,
        *,
        input_message: str | None = None,
        temperature: float = 0.0,
    ) -> str: ...


class MemoryExtractionCandidate(BaseModel):
    """LLM 返回的一条结构化 Memory Candidate。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    memory_type: MemoryType
    content: str = Field(min_length=1, max_length=500)


class MemoryExtractionPayload(BaseModel):
    """Memory Extraction 的稳定结构化输出。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    memories: tuple[MemoryExtractionCandidate, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )


@dataclass(frozen=True)
class MemoryExtractionResult:
    """一次已完成 Conversation turn 的抽取结果。"""

    created: tuple[MemoryItem, ...]
    skipped_duplicates: int
    prompt_id: str
    prompt_version: str


class LLMConversationMemoryExtractor:
    """把一轮已完成 Conversation 转成结构化 Memory Candidates。"""

    def __init__(self, *, llm_service: MemoryExtractionLLM) -> None:
        self.llm_service = llm_service

    def extract(
        self,
        *,
        user_message: ConversationMessage,
        assistant_message: ConversationMessage,
    ) -> tuple[tuple[MemoryExtractionCandidate, ...], str, str]:
        prompt = render_conversation_memory_extraction_prompt()
        input_message = (
            "当前用户消息（Memory 必须由这里明确支持）：\n"
            f"{user_message.content}\n\n"
            "本轮助手回答（仅用于理解上下文，不可单独作为 Memory 事实来源）：\n"
            f"{assistant_message.content}"
        )

        try:
            raw = self.llm_service.complete_prompt(
                prompt,
                input_message=input_message,
                temperature=0.0,
            )
            payload = MemoryExtractionPayload.model_validate(
                json.loads(self._strip_json_fence(raw))
            )
        except (json.JSONDecodeError, ValidationError, ValueError, RuntimeError) as exc:
            raise ConversationMemoryExtractionError(
                "memory extraction returned invalid structured output"
            ) from exc
        except Exception as exc:
            raise ConversationMemoryExtractionError(
                "memory extraction model call failed"
            ) from exc

        normalized: list[MemoryExtractionCandidate] = []
        seen: set[tuple[str, str]] = set()
        for candidate in payload.memories:
            content = candidate.content.strip()
            if not content:
                continue
            key = (candidate.memory_type.value, content.casefold())
            if key in seen:
                continue
            seen.add(key)
            normalized.append(
                MemoryExtractionCandidate(
                    memory_type=candidate.memory_type,
                    content=content,
                )
            )

        return tuple(normalized), prompt.prompt_id, prompt.version

    @staticmethod
    def _strip_json_fence(raw: str) -> str:
        normalized = raw.strip()
        if normalized.startswith("```") and normalized.endswith("```"):
            lines = normalized.splitlines()
            if len(lines) >= 3:
                return "\n".join(lines[1:-1]).strip()
        return normalized


class ConversationMemoryExtractionService:
    """B7 编排：已完成 turn -> Candidate -> B6 Memory Persistence。"""

    def __init__(
        self,
        *,
        extractor: LLMConversationMemoryExtractor,
        memory_service: ConversationMemoryService,
        message_repository: ConversationMessageRepository,
    ) -> None:
        self.extractor = extractor
        self.memory_service = memory_service
        self.message_repository = message_repository

    def extract_completed_turn(
        self,
        db: Session,
        *,
        user_id: int,
        knowledge_base_id: int,
        conversation_id: int,
        assistant_message_id: int,
        user_message_id: int | None = None,
    ) -> MemoryExtractionResult:
        assistant_message = self._get_message(
            db,
            conversation_id=conversation_id,
            message_id=assistant_message_id,
        )
        if assistant_message.role != ConversationMessageRole.ASSISTANT.value:
            raise ConversationMemoryExtractionError(
                "memory extraction source must end with assistant message"
            )

        user_message = (
            self._get_message(
                db,
                conversation_id=conversation_id,
                message_id=user_message_id,
            )
            if user_message_id is not None
            else self.message_repository.find_latest_user_before(
                db,
                conversation_id=conversation_id,
                before_message_id=assistant_message.id,
            )
        )
        if user_message is None or user_message.role != ConversationMessageRole.USER.value:
            raise ConversationMemoryExtractionError(
                "completed turn user message was not found"
            )
        if user_message.id >= assistant_message.id:
            raise ConversationMemoryExtractionError(
                "memory extraction source message order is invalid"
            )

        candidates, prompt_id, prompt_version = self.extractor.extract(
            user_message=user_message,
            assistant_message=assistant_message,
        )

        active = self.memory_service.list_active(
            db,
            user_id=user_id,
            knowledge_base_id=knowledge_base_id,
            conversation_id=conversation_id,
        )
        existing_keys = {
            (item.memory_type, item.content.strip().casefold())
            for item in active
        }

        created: list[MemoryItem] = []
        skipped_duplicates = 0
        for candidate in candidates:
            key = (
                candidate.memory_type.value,
                candidate.content.strip().casefold(),
            )
            if key in existing_keys:
                skipped_duplicates += 1
                continue

            memory = self.memory_service.create(
                db,
                user_id=user_id,
                knowledge_base_id=knowledge_base_id,
                conversation_id=conversation_id,
                source_message_start_id=user_message.id,
                source_message_end_id=assistant_message.id,
                memory_type=candidate.memory_type,
                content=candidate.content,
            )
            created.append(memory)
            existing_keys.add(key)

        return MemoryExtractionResult(
            created=tuple(created),
            skipped_duplicates=skipped_duplicates,
            prompt_id=prompt_id,
            prompt_version=prompt_version,
        )

    def _get_message(
        self,
        db: Session,
        *,
        conversation_id: int,
        message_id: int | None,
    ) -> ConversationMessage:
        if message_id is None:
            raise ConversationMemoryExtractionError(
                "memory extraction message id is required"
            )
        rows = self.message_repository.find_by_ids_in_conversation(
            db=db,
            conversation_id=conversation_id,
            message_ids={message_id},
        )
        if len(rows) != 1:
            raise ConversationMemoryExtractionError(
                "memory extraction source message was not found"
            )
        return rows[0]
