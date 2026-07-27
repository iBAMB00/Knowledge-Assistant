from collections.abc import Sequence

import pytest
from sqlalchemy.orm import Session

from app.schemas.vector_search_result import VectorSearchResult
from app.services.embedding.base import EmbeddingProvider
from app.services.retrieval_service import RetrievalService
from app.services.vector_store.base import VectorStore


class FakeEmbeddingProvider(EmbeddingProvider):
    """
    RetrievalService测试使用的Embedding Provider。
    """

    def __init__(self) -> None:
        self.received_query: str | None = None

    @property
    def model_name(self) -> str:
        """
        返回测试模型名称。
        """

        return "test-query-model"

    def embed_documents(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        """
        当前测试不使用文档向量化。
        """

        return [
            [1.0, 0.0]
            for _ in texts
        ]

    def embed_query(
        self,
        text: str,
    ) -> list[float]:
        """
        返回固定查询向量。
        """

        self.received_query = text
        return [1.0, 0.0]


class FakeVectorStore(VectorStore):
    """
    RetrievalService测试使用的向量存储。
    """

    def __init__(self) -> None:
        self.received_query_vector: list[float] | None = None
        self.received_embedding_model: str | None = None
        self.received_top_k: int | None = None
        self.received_document_id: int | None = None

    def search(
        self,
        db: Session,
        query_vector: Sequence[float],
        embedding_model: str,
        top_k: int = 5,
        document_id: int | None = None,
    ) -> list[VectorSearchResult]:
        """
        返回固定检索结果。
        """

        self.received_query_vector = list(query_vector)
        self.received_embedding_model = embedding_model
        self.received_top_k = top_k
        self.received_document_id = document_id

        results = [
            VectorSearchResult(
                document_id=1,
                chunk_id=1,
                chunk_index=0,
                content="第一条高相关文本",
                score=0.90,
            ),
            VectorSearchResult(
                document_id=1,
                chunk_id=2,
                chunk_index=1,
                content="第二条中等相关文本",
                score=0.65,
            ),
            VectorSearchResult(
                document_id=2,
                chunk_id=3,
                chunk_index=0,
                content="第三条低相关文本",
                score=0.20,
            ),
        ]

        return results[:top_k]


def test_retrieve_generates_query_vector_and_filters_results(
    db: Session,
) -> None:
    """
    验证查询向量生成、参数传递和阈值过滤。
    """

    embedding_provider = FakeEmbeddingProvider()
    vector_store = FakeVectorStore()

    retrieval_service = RetrievalService(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        default_top_k=3,
        default_score_threshold=0.60,
    )

    results = retrieval_service.retrieve(
        db=db,
        query="  企业知识库如何检索？  ",
        document_id=1,
    )

    assert embedding_provider.received_query == (
        "企业知识库如何检索？"
    )

    assert vector_store.received_query_vector == [
        1.0,
        0.0,
    ]

    assert (
        vector_store.received_embedding_model
        == "test-query-model"
    )

    assert vector_store.received_top_k == 3
    assert vector_store.received_document_id == 1

    assert len(results) == 2
    assert results[0].chunk_id == 1
    assert results[1].chunk_id == 2

    assert all(
        result.score >= 0.60
        for result in results
    )


def test_retrieve_allows_overriding_default_parameters(
    db: Session,
) -> None:
    """
    验证单次检索可以覆盖默认参数。
    """

    retrieval_service = RetrievalService(
        embedding_provider=FakeEmbeddingProvider(),
        vector_store=FakeVectorStore(),
        default_top_k=1,
        default_score_threshold=0.80,
    )

    results = retrieval_service.retrieve(
        db=db,
        query="测试问题",
        top_k=3,
        score_threshold=0.10,
    )

    assert len(results) == 3


def test_retrieve_rejects_empty_query(
    db: Session,
) -> None:
    """
    验证空查询被拒绝。
    """

    retrieval_service = RetrievalService(
        embedding_provider=FakeEmbeddingProvider(),
        vector_store=FakeVectorStore(),
    )

    with pytest.raises(
        ValueError,
        match="query cannot be empty",
    ):
        retrieval_service.retrieve(
            db=db,
            query="   ",
        )


@pytest.mark.parametrize(
    "top_k",
    [
        0,
        -1,
        True,
        1.5,
    ],
)
def test_retrieve_rejects_invalid_top_k(
    db: Session,
    top_k: object,
) -> None:
    """
    验证非法Top-K参数被拒绝。
    """

    retrieval_service = RetrievalService(
        embedding_provider=FakeEmbeddingProvider(),
        vector_store=FakeVectorStore(),
    )

    with pytest.raises(
        ValueError,
        match="top_k must be a positive integer",
    ):
        retrieval_service.retrieve(
            db=db,
            query="测试问题",
            top_k=top_k,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "score_threshold",
    [
        -1.1,
        1.1,
        True,
        "0.5",
    ],
)
def test_retrieve_rejects_invalid_score_threshold(
    db: Session,
    score_threshold: object,
) -> None:
    """
    验证非法相似度阈值被拒绝。
    """

    retrieval_service = RetrievalService(
        embedding_provider=FakeEmbeddingProvider(),
        vector_store=FakeVectorStore(),
    )

    with pytest.raises(
        ValueError,
        match="score_threshold",
    ):
        retrieval_service.retrieve(
            db=db,
            query="测试问题",
            score_threshold=score_threshold,  # type: ignore[arg-type]
        )