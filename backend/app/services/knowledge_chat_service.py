from collections.abc import Iterator
from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.models.database.document_chunk import DocumentChunk
from app.repositories.document_chunk_repository import (
    DocumentChunkRepository,
)
from app.schemas.knowledge_chat_response import (
    KnowledgeChatResponse,
    KnowledgeChatSource,
)
from app.services.llm_service import LLMService
from app.services.rag.context_builder import (
    ContextBuilder,
    ContextSource,
)
from app.services.retrieval_service import RetrievalService


@dataclass(frozen=True)
class KnowledgeChatPreparation:
    """
    知识库问答准备结果。

    prompt 为 None 时，表示不需要调用 LLM，
    直接返回 direct_answer。
    """

    prompt: str | None
    sources: list[KnowledgeChatSource]
    direct_answer: str | None = None


class KnowledgeChatService:
    """
    知识库问答编排服务。

    负责：
    - 检索与问题相关的文本切片
    - 构建 RAG 上下文
    - 构建知识库问答 Prompt
    - 调用 LLM 生成普通或流式回答
    - 将内部上下文来源转换为公开来源

    不负责：
    - HTTP 请求处理
    - 文档解析和向量化
    - 向量相似度计算
    - SSE 事件格式化
    """

    NO_RELIABLE_ANSWER = (
        "当前知识库中没有找到足够可靠的相关内容。"
    )

    def __init__(
        self,
        retrieval_service: RetrievalService,
        context_builder: ContextBuilder,
        llm_service: LLMService,
        document_chunk_repository: DocumentChunkRepository | None = None,
    ) -> None:
        """初始化知识库问答服务。"""

        self.retrieval_service = retrieval_service
        self.context_builder = context_builder
        self.llm_service = llm_service
        self.document_chunk_repository = document_chunk_repository

    def prepare(
        self,
        db: Session,
        question: str,
        top_k: int | None = None,
        score_threshold: float | None = None,
        document_id: int | None = None,
        knowledge_base_id: int | None = None,
    ) -> KnowledgeChatPreparation:
        """完成检索、上下文构建和 Prompt 准备。"""

        normalized_question = question.strip()

        if not normalized_question:
            raise ValueError(
                "question cannot be empty"
            )

        retrieval_kwargs = {
            "db": db,
            "query": normalized_question,
            "top_k": top_k,
            "score_threshold": score_threshold,
            "document_id": document_id,
        }
        if knowledge_base_id is not None:
            retrieval_kwargs["knowledge_base_id"] = knowledge_base_id

        retrieval_results = self.retrieval_service.retrieve(**retrieval_kwargs)

        context_result = self.context_builder.build(
            retrieval_results
        )

        if (
            not context_result.context
            or not context_result.sources
        ):
            return KnowledgeChatPreparation(
                prompt=None,
                sources=[],
                direct_answer=self.NO_RELIABLE_ANSWER,
            )

        prompt = self._build_prompt(
            question=normalized_question,
            context=context_result.context,
        )

        return KnowledgeChatPreparation(
            prompt=prompt,
            sources=self._build_public_sources(
                db=db,
                sources=context_result.sources,
                knowledge_base_id=knowledge_base_id,
            ),
        )

    def chat(
        self,
        db: Session,
        question: str,
        top_k: int | None = None,
        score_threshold: float | None = None,
        document_id: int | None = None,
        knowledge_base_id: int | None = None,
    ) -> KnowledgeChatResponse:
        """根据知识库内容生成非流式回答。"""

        preparation = self.prepare(
            db=db,
            question=question,
            top_k=top_k,
            score_threshold=score_threshold,
            document_id=document_id,
            knowledge_base_id=knowledge_base_id,
        )

        if preparation.prompt is None:
            return KnowledgeChatResponse(
                answer=(
                    preparation.direct_answer
                    or self.NO_RELIABLE_ANSWER
                ),
                sources=preparation.sources,
            )

        answer = self.llm_service.chat(
            preparation.prompt
        ).strip()

        if not answer:
            raise RuntimeError(
                "model returned empty answer"
            )

        return KnowledgeChatResponse(
            answer=answer,
            sources=preparation.sources,
        )

    def stream_chat(
        self,
        preparation: KnowledgeChatPreparation,
    ) -> Iterator[str]:
        """根据准备结果生成流式回答片段。"""

        if preparation.prompt is None:
            yield (
                preparation.direct_answer
                or self.NO_RELIABLE_ANSWER
            )
            return

        has_valid_content = False

        for content in self.llm_service.stream_chat(
            preparation.prompt
        ):
            if not content:
                continue

            if content.strip():
                has_valid_content = True

            yield content

        if not has_valid_content:
            raise RuntimeError(
                "model returned empty answer"
            )

    def _build_public_sources(
        self,
        db: Session,
        sources: list[ContextSource],
        knowledge_base_id: int | None,
    ) -> list[KnowledgeChatSource]:
        """将实际进入上下文的来源补充 SQL 结构定位后对外返回。"""

        chunks_by_id = self._load_source_chunks(
            db=db,
            sources=sources,
            knowledge_base_id=knowledge_base_id,
        )

        public_sources: list[KnowledgeChatSource] = []

        for source in sources:
            chunk = chunks_by_id.get(source.chunk_id)
            metadata = (
                chunk.chunk_metadata
                if chunk is not None
                and isinstance(chunk.chunk_metadata, dict)
                else {}
            )

            page_numbers = self._normalize_page_numbers(
                metadata.get("page_numbers")
            )
            start_page = self._normalize_positive_integer(
                metadata.get("start_page")
            )
            end_page = self._normalize_positive_integer(
                metadata.get("end_page")
            )

            if start_page is None and page_numbers:
                start_page = page_numbers[0]
            if end_page is None and page_numbers:
                end_page = page_numbers[-1]

            public_sources.append(
                KnowledgeChatSource(
                    source_number=source.source_number,
                    document_id=source.document_id,
                    filename=source.filename,
                    chunk_id=source.chunk_id,
                    excerpt=self._build_source_excerpt(
                        source=source,
                        chunk=chunk,
                    ),
                    section_title=self._normalize_optional_text(
                        metadata.get("section_title")
                    ),
                    heading_path=self._normalize_heading_path(
                        metadata.get("heading_path")
                    ),
                    start_page=start_page,
                    end_page=end_page,
                    page_numbers=page_numbers,
                )
            )

        return public_sources

    def _load_source_chunks(
        self,
        db: Session,
        sources: list[ContextSource],
        knowledge_base_id: int | None,
    ) -> dict[int, DocumentChunk]:
        """批量读取最终命中 Chunk，用于精确来源摘要和结构定位。"""

        repository = self.document_chunk_repository
        if repository is None or not sources:
            return {}

        query_kwargs = {
            "db": db,
            "chunk_ids": [source.chunk_id for source in sources],
        }
        if knowledge_base_id is not None:
            query_kwargs["knowledge_base_id"] = knowledge_base_id

        chunks = repository.find_by_ids(**query_kwargs)
        return {chunk.id: chunk for chunk in chunks}

    @staticmethod
    def _build_source_excerpt(
        source: ContextSource,
        chunk: DocumentChunk | None,
    ) -> str:
        """优先返回实际命中 Chunk 正文，缺失时回退到上下文摘要。"""

        if chunk is not None:
            content = chunk.content.strip()
            if content:
                return content

        return source.excerpt

    @staticmethod
    def _normalize_optional_text(value: object) -> str | None:
        """将内部可选文本元数据转换为稳定公开值。"""

        if not isinstance(value, str):
            return None

        normalized = value.strip()
        return normalized or None

    @staticmethod
    def _normalize_heading_path(value: object) -> list[str]:
        """过滤无效 Heading Path 元素。"""

        if not isinstance(value, list):
            return []

        return [
            normalized
            for item in value
            if isinstance(item, str)
            if (normalized := item.strip())
        ]

    @classmethod
    def _normalize_page_numbers(cls, value: object) -> list[int]:
        """过滤无效页码并保持原始顺序去重。"""

        if not isinstance(value, list):
            return []

        normalized: list[int] = []
        seen: set[int] = set()

        for item in value:
            page_number = cls._normalize_positive_integer(item)
            if page_number is None or page_number in seen:
                continue

            seen.add(page_number)
            normalized.append(page_number)

        return normalized

    @staticmethod
    def _normalize_positive_integer(value: object) -> int | None:
        """仅接受真正的正整数，避免 bool 等 JSON 值污染响应。"""

        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
        ):
            return None

        return value

    @staticmethod
    def _build_prompt(
        question: str,
        context: str,
    ) -> str:
        """构建知识库问答 Prompt。"""

        return (
            "请严格根据下面提供的知识库资料"
            "回答用户问题。\n\n"
            "回答规则：\n"
            "1. 只能使用知识库资料中的信息。\n"
            "2. 不要补充资料中不存在的事实。\n"
            "3. 资料不足时，明确说明无法确定。\n"
            "4. 引用具体资料时，使用"
            "“[来源 N]”标记。\n"
            "5. 回答应准确、简洁、清晰。\n\n"
            "知识库资料：\n"
            f"{context}\n\n"
            "用户问题：\n"
            f"{question}"
        )