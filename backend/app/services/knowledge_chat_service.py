from collections.abc import Iterator
from dataclasses import dataclass

from sqlalchemy.orm import Session

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
    ) -> None:
        """初始化知识库问答服务。"""

        self.retrieval_service = retrieval_service
        self.context_builder = context_builder
        self.llm_service = llm_service

    def prepare(
        self,
        db: Session,
        question: str,
        top_k: int | None = None,
        score_threshold: float | None = None,
        document_id: int | None = None,
    ) -> KnowledgeChatPreparation:
        """完成检索、上下文构建和 Prompt 准备。"""

        normalized_question = question.strip()

        if not normalized_question:
            raise ValueError(
                "question cannot be empty"
            )

        retrieval_results = (
            self.retrieval_service.retrieve(
                db=db,
                query=normalized_question,
                top_k=top_k,
                score_threshold=score_threshold,
                document_id=document_id,
            )
        )

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
                context_result.sources
            ),
        )

    def chat(
        self,
        db: Session,
        question: str,
        top_k: int | None = None,
        score_threshold: float | None = None,
        document_id: int | None = None,
    ) -> KnowledgeChatResponse:
        """根据知识库内容生成非流式回答。"""

        preparation = self.prepare(
            db=db,
            question=question,
            top_k=top_k,
            score_threshold=score_threshold,
            document_id=document_id,
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

    @staticmethod
    def _build_public_sources(
        sources: list[ContextSource],
    ) -> list[KnowledgeChatSource]:
        """将内部上下文来源转换为公开响应来源。"""

        return [
            KnowledgeChatSource(
                source_number=source.source_number,
                document_id=source.document_id,
                filename=source.filename,
                excerpt=source.excerpt,
            )
            for source in sources
        ]

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