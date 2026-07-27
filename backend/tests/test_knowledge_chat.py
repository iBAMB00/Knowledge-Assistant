from sqlalchemy.orm import Session

import pytest

from app.schemas.vector_search_result import (
    VectorSearchResult,
)
from app.services.knowledge_chat_service import (
    KnowledgeChatService,
)
from app.services.rag.context_builder import (
    ContextBuilder,
)


class FakeRetrievalService:
    """
    KnowledgeChatService测试使用的检索服务。
    """

    def __init__(
        self,
        results: list[VectorSearchResult],
    ) -> None:
        self.results = results

        self.received_query: str | None = None
        self.received_top_k: int | None = None
        self.received_score_threshold: float | None = None
        self.received_document_id: int | None = None

    def retrieve(
        self,
        db: Session,
        query: str,
        top_k: int | None = None,
        score_threshold: float | None = None,
        document_id: int | None = None,
    ) -> list[VectorSearchResult]:
        """
        返回预设检索结果。
        """

        self.received_query = query
        self.received_top_k = top_k
        self.received_score_threshold = score_threshold
        self.received_document_id = document_id

        return self.results


class FakeLLMService:
    """
    KnowledgeChatService测试使用的LLM服务。
    """

    def __init__(
        self,
        answer: str = "这是知识库生成的回答。",
    ) -> None:
        self.answer = answer
        self.received_message: str | None = None
        self.call_count = 0

    def chat(self, message: str) -> str:
        """
        返回预设模型回答。
        """

        self.received_message = message
        self.call_count += 1

        return self.answer


def build_search_result(
    document_id: int,
    chunk_id: int,
    chunk_index: int,
    content: str,
    score: float,
) -> VectorSearchResult:
    """
    创建测试使用的向量检索结果。
    """

    return VectorSearchResult(
        document_id=document_id,
        chunk_id=chunk_id,
        chunk_index=chunk_index,
        content=content,
        score=score,
    )


def test_knowledge_chat_returns_answer_and_sources(
    db: Session,
) -> None:
    """
    验证知识库问答返回模型回答和实际来源。
    """

    retrieval_service = FakeRetrievalService(
        results=[
            build_search_result(
                document_id=1,
                chunk_id=10,
                chunk_index=0,
                content=(
                    "管理员可以在系统设置中"
                    "重置用户密码。"
                ),
                score=0.95,
            ),
            build_search_result(
                document_id=2,
                chunk_id=20,
                chunk_index=1,
                content="这是相关度较低的补充内容。",
                score=0.70,
            ),
        ]
    )

    llm_service = FakeLLMService(
        answer="管理员可在系统设置中重置密码。"
    )

    service = KnowledgeChatService(
        retrieval_service=retrieval_service,
        context_builder=ContextBuilder(
            default_max_chunks=1,
        ),
        llm_service=llm_service,
    )

    response = service.chat(
        db=db,
        question="  如何重置用户密码？  ",
        top_k=5,
        score_threshold=0.60,
        document_id=1,
    )

    assert response.answer == (
        "管理员可在系统设置中重置密码。"
    )

    assert len(response.sources) == 1
    assert response.sources[0].chunk_id == 10

    assert retrieval_service.received_query == (
        "如何重置用户密码？"
    )

    assert retrieval_service.received_top_k == 5
    assert (
        retrieval_service.received_score_threshold
        == 0.60
    )
    assert retrieval_service.received_document_id == 1

    assert llm_service.call_count == 1
    assert llm_service.received_message is not None

    assert (
        "管理员可以在系统设置中"
        in llm_service.received_message
    )

    assert (
        "如何重置用户密码？"
        in llm_service.received_message
    )

    assert (
        "[来源 1]"
        in llm_service.received_message
    )

    assert (
        "这是相关度较低的补充内容"
        not in llm_service.received_message
    )


def test_knowledge_chat_does_not_call_llm_when_no_results(
    db: Session,
) -> None:
    """
    验证没有检索结果时不调用LLM。
    """

    llm_service = FakeLLMService()

    service = KnowledgeChatService(
        retrieval_service=FakeRetrievalService(
            results=[]
        ),
        context_builder=ContextBuilder(),
        llm_service=llm_service,
    )

    response = service.chat(
        db=db,
        question="知识库中不存在的问题",
    )

    assert response.answer == (
        KnowledgeChatService.NO_RELIABLE_ANSWER
    )

    assert response.sources == []
    assert llm_service.call_count == 0


def test_knowledge_chat_does_not_call_llm_when_context_empty(
    db: Session,
) -> None:
    """
    验证检索内容为空时不调用LLM。
    """

    llm_service = FakeLLMService()

    service = KnowledgeChatService(
        retrieval_service=FakeRetrievalService(
            results=[
                build_search_result(
                    document_id=1,
                    chunk_id=1,
                    chunk_index=0,
                    content="   ",
                    score=0.95,
                )
            ]
        ),
        context_builder=ContextBuilder(),
        llm_service=llm_service,
    )

    response = service.chat(
        db=db,
        question="测试问题",
    )

    assert response.answer == (
        KnowledgeChatService.NO_RELIABLE_ANSWER
    )

    assert response.sources == []
    assert llm_service.call_count == 0


def test_knowledge_chat_rejects_empty_question(
    db: Session,
) -> None:
    """
    验证空问题被拒绝。
    """

    service = KnowledgeChatService(
        retrieval_service=FakeRetrievalService(
            results=[]
        ),
        context_builder=ContextBuilder(),
        llm_service=FakeLLMService(),
    )

    with pytest.raises(
        ValueError,
        match="question cannot be empty",
    ):
        service.chat(
            db=db,
            question="   ",
        )


def test_knowledge_chat_rejects_empty_model_answer(
    db: Session,
) -> None:
    """
    验证模型返回空白内容时抛出异常。
    """

    service = KnowledgeChatService(
        retrieval_service=FakeRetrievalService(
            results=[
                build_search_result(
                    document_id=1,
                    chunk_id=1,
                    chunk_index=0,
                    content="有效的知识库内容",
                    score=0.95,
                )
            ]
        ),
        context_builder=ContextBuilder(),
        llm_service=FakeLLMService(
            answer="   "
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="model returned empty answer",
    ):
        service.chat(
            db=db,
            question="测试问题",
        )