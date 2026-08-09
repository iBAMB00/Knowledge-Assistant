from types import SimpleNamespace

import pytest
from sqlalchemy.orm import Session

from app.schemas.vector_search_result import VectorSearchResult
from app.services.knowledge_chat_service import KnowledgeChatService
from app.services.rag.context_builder import ContextBuilder


class FakeRetrievalService:
    """
    KnowledgeChatService 测试使用的检索服务。
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
        self.received_knowledge_base_id: int | None = None

    def retrieve(
        self,
        db: Session,
        query: str,
        top_k: int | None = None,
        score_threshold: float | None = None,
        document_id: int | None = None,
        knowledge_base_id: int | None = None,
    ) -> list[VectorSearchResult]:
        """
        返回预设检索结果。
        """

        self.received_query = query
        self.received_top_k = top_k
        self.received_score_threshold = score_threshold
        self.received_document_id = document_id
        self.received_knowledge_base_id = knowledge_base_id

        return self.results


class FakeDocumentChunkRepository:
    """知识问答引用定位测试使用的 Chunk Repository。"""

    def __init__(
        self,
        metadata_by_chunk_id: dict[int, dict],
        content_by_chunk_id: dict[int, str] | None = None,
    ) -> None:
        self.metadata_by_chunk_id = metadata_by_chunk_id
        self.content_by_chunk_id = content_by_chunk_id or {}
        self.received_chunk_ids: list[int] = []
        self.received_knowledge_base_id: int | None = None

    def find_by_ids(
        self,
        db: Session,
        chunk_ids: list[int],
        knowledge_base_id: int | None = None,
    ) -> list[SimpleNamespace]:
        self.received_chunk_ids = chunk_ids
        self.received_knowledge_base_id = knowledge_base_id

        return [
            SimpleNamespace(
                id=chunk_id,
                content=self.content_by_chunk_id.get(
                    chunk_id,
                    f"child-{chunk_id}",
                ),
                chunk_metadata=self.metadata_by_chunk_id.get(chunk_id),
            )
            for chunk_id in chunk_ids
            if chunk_id in self.metadata_by_chunk_id
        ]


class FakeLLMService:
    """
    KnowledgeChatService 测试使用的 LLM 服务。
    """

    def __init__(
        self,
        answer: str = "这是知识库生成的回答。",
    ) -> None:
        self.answer = answer
        self.received_message: str | None = None
        self.call_count = 0

    def chat(
        self,
        message: str,
    ) -> str:
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
    filename: str = "test-document.txt",
) -> VectorSearchResult:
    """
    创建测试使用的向量检索结果。
    """

    return VectorSearchResult(
        document_id=document_id,
        filename=filename,
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
                filename="test-document.txt",
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
                filename="test-document.txt",
                chunk_id=20,
                chunk_index=1,
                content=(
                    "这是相关度较低的补充内容。"
                ),
                score=0.70,
            ),
        ]
    )

    llm_service = FakeLLMService(
        answer=(
            "管理员可在系统设置中重置密码。"
        )
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

    assert (
        response.sources[0].source_number
        == 1
    )

    assert response.sources[0].document_id == 1
    assert response.sources[0].chunk_id == 10
    assert response.sources[0].section_title is None
    assert response.sources[0].heading_path == []
    assert response.sources[0].page_numbers == []

    assert response.sources[0].excerpt == (
        "管理员可以在系统设置中"
        "重置用户密码。"
    )

    assert retrieval_service.received_query == (
        "如何重置用户密码？"
    )

    assert retrieval_service.received_top_k == 5

    assert (
        retrieval_service.received_score_threshold
        == 0.60
    )

    assert (
        retrieval_service.received_document_id
        == 1
    )

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


def test_knowledge_chat_exposes_traceable_source_but_not_ranking_fields(
    db: Session,
) -> None:
    """验证来源暴露可追踪 Chunk ID，但不泄漏检索排序内部字段。"""

    service = KnowledgeChatService(
        retrieval_service=FakeRetrievalService(
            results=[
                build_search_result(
                    document_id=1,
                    filename="test-document.txt",
                    chunk_id=99,
                    chunk_index=3,
                    content="有效知识内容",
                    score=0.95,
                )
            ]
        ),
        context_builder=ContextBuilder(),
        llm_service=FakeLLMService(),
    )

    response = service.chat(
        db=db,
        question="测试问题",
    )

    source_payload = (
        response.sources[0].model_dump()
    )

    assert source_payload == {
        "filename": "test-document.txt",
        "source_number": 1,
        "document_id": 1,
        "chunk_id": 99,
        "excerpt": "有效知识内容",
        "section_title": None,
        "heading_path": [],
        "start_page": None,
        "end_page": None,
        "page_numbers": [],
    }

    assert "chunk_index" not in source_payload
    assert "score" not in source_payload
    assert "content" not in source_payload


def test_knowledge_chat_enriches_source_with_chunk_structure_metadata(
    db: Session,
) -> None:
    """验证最终来源通过命中 Chunk ID 回 SQL 补充章节和页码。"""

    repository = FakeDocumentChunkRepository(
        metadata_by_chunk_id={
            10: {
                "section_title": " PostgreSQL ",
                "heading_path": [
                    "部署指南",
                    " PostgreSQL ",
                ],
                "start_page": 3,
                "end_page": 4,
                "page_numbers": [3, 4],
            }
        },
        content_by_chunk_id={
            10: "数据库连接地址为 postgres:5432。",
        },
    )
    retrieval_service = FakeRetrievalService(
        results=[
            build_search_result(
                document_id=1,
                filename="deployment.pdf",
                chunk_id=10,
                chunk_index=0,
                content=(
                    "PostgreSQL 部署章节完整 Parent 上下文，"
                    "包含连接池和故障处理说明。"
                ),
                score=0.98,
            )
        ]
    )

    service = KnowledgeChatService(
        retrieval_service=retrieval_service,
        context_builder=ContextBuilder(),
        llm_service=FakeLLMService(),
        document_chunk_repository=repository,
    )

    response = service.chat(
        db=db,
        question="数据库地址是什么？",
        knowledge_base_id=7,
    )

    source = response.sources[0]

    assert source.chunk_id == 10
    assert source.excerpt == "数据库连接地址为 postgres:5432。"
    assert source.section_title == "PostgreSQL"
    assert source.heading_path == [
        "部署指南",
        "PostgreSQL",
    ]
    assert source.start_page == 3
    assert source.end_page == 4
    assert source.page_numbers == [3, 4]

    assert repository.received_chunk_ids == [10]
    assert repository.received_knowledge_base_id == 7
    assert retrieval_service.received_knowledge_base_id == 7


def test_knowledge_chat_normalizes_invalid_source_metadata(
    db: Session,
) -> None:
    """验证脏结构元数据不会污染公开 Source DTO。"""

    repository = FakeDocumentChunkRepository(
        metadata_by_chunk_id={
            10: {
                "section_title": "   ",
                "heading_path": ["部署", " ", 123],
                "start_page": True,
                "page_numbers": [2, 2, -1, True, 3],
            }
        }
    )

    service = KnowledgeChatService(
        retrieval_service=FakeRetrievalService(
            results=[
                build_search_result(
                    document_id=1,
                    chunk_id=10,
                    chunk_index=0,
                    content="有效知识内容",
                    score=0.95,
                )
            ]
        ),
        context_builder=ContextBuilder(),
        llm_service=FakeLLMService(),
        document_chunk_repository=repository,
    )

    source = service.chat(
        db=db,
        question="测试",
    ).sources[0]

    assert source.section_title is None
    assert source.heading_path == ["部署"]
    assert source.start_page == 2
    assert source.end_page == 3
    assert source.page_numbers == [2, 3]


def test_knowledge_chat_does_not_call_llm_when_no_results(
    db: Session,
) -> None:
    """
    验证没有检索结果时不调用 LLM。
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
    验证检索内容为空时不调用 LLM。
    """

    llm_service = FakeLLMService()

    service = KnowledgeChatService(
        retrieval_service=FakeRetrievalService(
            results=[
                build_search_result(
                    document_id=1,
                    filename="test-document.txt",
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
                    filename="test-document.txt",
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