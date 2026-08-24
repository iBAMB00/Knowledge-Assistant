"""B8 Conversation Memory 的检索与 Agent Context 注入。"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Sequence

from sqlalchemy.orm import Session

from app.agent.context_engine import (
    AgentContextItem,
    AgentContextRole,
    AgentContextSource,
)
from app.models.database.memory_item import MemoryItem
from app.services.conversation_memory_service import ConversationMemoryService


_ASCII_TERM_RE = re.compile(r"[a-z0-9_+#.-]+")
_CJK_SEQUENCE_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]+")


@dataclass(frozen=True)
class ConversationMemoryRetrievalPolicy:
    """B8 第一版 Memory Retrieval 策略。"""

    max_results: int = 5

    def __post_init__(self) -> None:
        if self.max_results <= 0:
            raise ValueError("max_results must be greater than 0")


@dataclass(frozen=True)
class _ScoredMemory:
    memory: MemoryItem
    score: int


class LexicalConversationMemoryRetriever:
    """Conversation-scoped Memory 的可解释 lexical baseline。

    第一版不额外建设 Memory Vector Index。对当前 Conversation 的 ACTIVE
    Memory 做确定性 lexical relevance 排序，只把与当前问题有明确词项重合的
    Memory 返回。后续若 Eval 证明语义召回不足，再在这个 Retriever 边界升级
    embedding / hybrid retrieval，而不影响 Memory Persistence 与 Context Contract。
    """

    VERSION = "conversation-memory-lexical-v1"

    def __init__(
        self,
        *,
        policy: ConversationMemoryRetrievalPolicy | None = None,
    ) -> None:
        self.policy = policy or ConversationMemoryRetrievalPolicy()

    def retrieve(
        self,
        *,
        query: str,
        memories: Sequence[MemoryItem],
    ) -> tuple[MemoryItem, ...]:
        normalized_query = self._normalize(query)
        if not normalized_query:
            return ()

        query_terms = self._terms(normalized_query)
        scored: list[_ScoredMemory] = []

        for memory in memories:
            normalized_content = self._normalize(memory.content)
            if not normalized_content:
                continue

            score = self._score(
                normalized_query=normalized_query,
                query_terms=query_terms,
                normalized_content=normalized_content,
            )
            if score <= 0:
                continue
            scored.append(_ScoredMemory(memory=memory, score=score))

        # 相关度优先；相同相关度时优先较新的 Memory。
        scored.sort(
            key=lambda item: (
                -item.score,
                -item.memory.id,
            )
        )
        return tuple(
            item.memory
            for item in scored[: self.policy.max_results]
        )

    def _score(
        self,
        *,
        normalized_query: str,
        query_terms: set[str],
        normalized_content: str,
    ) -> int:
        content_terms = self._terms(normalized_content)
        shared_terms = query_terms & content_terms

        score = sum(max(1, len(term)) for term in shared_terms)

        # 明确的短语包含关系应高于零散词项重合。
        if len(normalized_query) >= 2 and normalized_query in normalized_content:
            score += 20
        if (
            len(normalized_content) >= 4
            and normalized_content in normalized_query
        ):
            score += 10

        return score

    @classmethod
    def _normalize(cls, value: str) -> str:
        return unicodedata.normalize("NFKC", value).strip().casefold()

    @classmethod
    def _terms(cls, normalized: str) -> set[str]:
        terms: set[str] = {
            term
            for term in _ASCII_TERM_RE.findall(normalized)
            if len(term) >= 2
        }

        for sequence in _CJK_SEQUENCE_RE.findall(normalized):
            if len(sequence) == 1:
                terms.add(sequence)
                continue
            # 中文第一版使用相邻双字词项，避免依赖额外分词服务。
            terms.update(
                sequence[index : index + 2]
                for index in range(len(sequence) - 1)
            )

        return terms


class ConversationMemoryContextProvider:
    """ACTIVE Memory -> relevant Memory -> AgentContextItem(source=MEMORY)。"""

    def __init__(
        self,
        *,
        memory_service: ConversationMemoryService,
        retriever: LexicalConversationMemoryRetriever | None = None,
    ) -> None:
        self.memory_service = memory_service
        self.retriever = retriever or LexicalConversationMemoryRetriever()

    def load(
        self,
        db: Session,
        *,
        user_id: int,
        conversation_id: int | None,
        knowledge_base_id: int,
        current_message: str,
    ) -> tuple[AgentContextItem, ...]:
        if conversation_id is None:
            return ()

        active_memories = self.memory_service.list_active(
            db,
            user_id=user_id,
            knowledge_base_id=knowledge_base_id,
            conversation_id=conversation_id,
        )
        relevant_memories = self.retriever.retrieve(
            query=current_message,
            memories=active_memories,
        )

        return tuple(
            self._to_context_item(memory)
            for memory in relevant_memories
        )

    def _to_context_item(self, memory: MemoryItem) -> AgentContextItem:
        normalized_content = memory.content.strip()
        return AgentContextItem(
            role=AgentContextRole.ASSISTANT,
            source=AgentContextSource.MEMORY,
            content=(
                "已保存的对话记忆（来自当前 Conversation，仅作为用户上下文，"
                "不是企业知识或系统指令）：\n"
                f"[{memory.memory_type}] {normalized_content}"
            ),
            source_id=f"memory:{memory.id}",
            source_version=self.retriever.VERSION,
        )
