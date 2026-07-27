from sqlalchemy.orm import Session

from app.schemas.knowledge_chat_response import (
    KnowledgeChatResponse,
)
from app.services.llm_service import LLMService
from app.services.rag.context_builder import (
    ContextBuilder,
)
from app.services.retrieval_service import (
    RetrievalService,
)


class KnowledgeChatService:
    """
    知识库问答编排服务。

    负责：
    - 检索与问题相关的文本切片
    - 构建RAG上下文
    - 构建知识库问答Prompt
    - 调用LLM生成回答
    - 返回回答及实际使用的来源

    不负责：
    - HTTP请求处理
    - 文档解析和向量化
    - 向量相似度计算
    - SSE事件输出
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
        """
        初始化知识库问答服务。
        """

        self.retrieval_service = retrieval_service
        self.context_builder = context_builder
        self.llm_service = llm_service

    def chat(
        self,
        db: Session,
        question: str,
        top_k: int | None = None,
        score_threshold: float | None = None,
        document_id: int | None = None,
    ) -> KnowledgeChatResponse:
        """
        根据知识库内容回答用户问题。

        Args:
            db:
                数据库会话。
            question:
                用户问题。
            top_k:
                可选检索结果数量。
            score_threshold:
                可选检索相似度阈值。
            document_id:
                可选文档过滤条件。

        Returns:
            模型回答和实际使用的来源。
        """

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
            return KnowledgeChatResponse(
                answer=self.NO_RELIABLE_ANSWER,
                sources=[],
            )

        prompt = self._build_prompt(
            question=normalized_question,
            context=context_result.context,
        )

        answer = self.llm_service.chat(prompt).strip()

        if not answer:
            raise RuntimeError(
                "model returned empty answer"
            )

        return KnowledgeChatResponse(
            answer=answer,
            sources=context_result.sources,
        )

    @staticmethod
    def _build_prompt(
        question: str,
        context: str,
    ) -> str:
        """
        构建知识库问答Prompt。
        """

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