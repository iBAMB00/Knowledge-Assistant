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
        default_candidate_k=3,
        default_score_threshold=0.60,
    )

    results = retrieval_service.retrieve(
        db=db,
        query="  企业知识库如何检索？  ",
        document_id=1,
        candidate_k=3,
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
        candidate_k=3,
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
            candidate_k=3,
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

class MultiDocumentVectorStore(VectorStore):
    """
    多文档召回测试使用的向量存储。
    """

    def __init__(
        self,
        results: list[VectorSearchResult],
    ) -> None:
        self.results = results
        self.received_top_k: int | None = None

    def search(
        self,
        db: Session,
        query_vector: Sequence[float],
        embedding_model: str,
        top_k: int = 5,
        document_id: int | None = None,
    ) -> list[VectorSearchResult]:
        """
        返回预设候选结果。
        """

        self.received_top_k = top_k

        return self.results[:top_k]


def build_search_result(
    document_id: int,
    chunk_id: int,
    content: str,
    score: float,
) -> VectorSearchResult:
    """
    创建检索结果。
    """

    return VectorSearchResult(
        document_id=document_id,
        chunk_id=chunk_id,
        chunk_index=chunk_id,
        content=content,
        score=score,
    )


def test_retrieve_uses_candidate_k_for_vector_search(
    db: Session,
) -> None:
    """
    验证VectorStore接收的是candidate_k。
    """

    vector_store = MultiDocumentVectorStore(
        results=[
            build_search_result(
                document_id=1,
                chunk_id=1,
                content="测试内容",
                score=0.90,
            )
        ]
    )

    service = RetrievalService(
        embedding_provider=FakeEmbeddingProvider(),
        vector_store=vector_store,
    )

    service.retrieve(
        db=db,
        query="测试问题",
        top_k=1,
        candidate_k=10,
    )

    assert vector_store.received_top_k == 10


def test_retrieve_balances_multiple_documents(
    db: Session,
) -> None:
    """
    验证多个文档不会被单一文档占满。
    """

    vector_store = MultiDocumentVectorStore(
        results=[
            build_search_result(
                document_id=1,
                chunk_id=1,
                content="文档一第一条",
                score=0.99,
            ),
            build_search_result(
                document_id=1,
                chunk_id=2,
                content="文档一第二条",
                score=0.98,
            ),
            build_search_result(
                document_id=1,
                chunk_id=3,
                content="文档一第三条",
                score=0.97,
            ),
            build_search_result(
                document_id=2,
                chunk_id=4,
                content="文档二内容",
                score=0.96,
            ),
            build_search_result(
                document_id=3,
                chunk_id=5,
                content="文档三内容",
                score=0.95,
            ),
        ]
    )

    service = RetrievalService(
        embedding_provider=FakeEmbeddingProvider(),
        vector_store=vector_store,
    )

    results = service.retrieve(
        db=db,
        query="多文档问题",
        top_k=3,
        candidate_k=5,
        per_document_limit=1,
    )

    assert [
        result.chunk_id
        for result in results
    ] == [
        1,
        4,
        5,
    ]

    assert {
        result.document_id
        for result in results
    } == {
        1,
        2,
        3,
    }


def test_retrieve_backfills_results_when_documents_are_few(
    db: Session,
) -> None:
    """
    验证文档数量不足时使用高分结果补足Top-K。
    """

    vector_store = MultiDocumentVectorStore(
        results=[
            build_search_result(
                document_id=1,
                chunk_id=1,
                content="第一条内容",
                score=0.99,
            ),
            build_search_result(
                document_id=1,
                chunk_id=2,
                content="第二条内容",
                score=0.98,
            ),
            build_search_result(
                document_id=1,
                chunk_id=3,
                content="第三条内容",
                score=0.97,
            ),
        ]
    )

    service = RetrievalService(
        embedding_provider=FakeEmbeddingProvider(),
        vector_store=vector_store,
    )

    results = service.retrieve(
        db=db,
        query="单文档问题",
        top_k=3,
        candidate_k=3,
        per_document_limit=1,
    )

    assert [
        result.chunk_id
        for result in results
    ] == [
        1,
        2,
        3,
    ]


def test_retrieve_removes_duplicate_content_in_same_document(
    db: Session,
) -> None:
    """
    验证同一文档中的完全重复内容被去除。
    """

    vector_store = MultiDocumentVectorStore(
        results=[
            build_search_result(
                document_id=1,
                chunk_id=1,
                content="管理员可以重置密码。",
                score=0.99,
            ),
            build_search_result(
                document_id=1,
                chunk_id=2,
                content=" 管理员可以重置密码。 ",
                score=0.98,
            ),
            build_search_result(
                document_id=2,
                chunk_id=3,
                content="密码重置需要管理员权限。",
                score=0.90,
            ),
        ]
    )

    service = RetrievalService(
        embedding_provider=FakeEmbeddingProvider(),
        vector_store=vector_store,
    )

    results = service.retrieve(
        db=db,
        query="如何重置密码",
        top_k=3,
        candidate_k=3,
        per_document_limit=2,
    )

    assert [
        result.chunk_id
        for result in results
    ] == [
        1,
        3,
    ]


@pytest.mark.parametrize(
    ("top_k", "candidate_k"),
    [
        (5, 4),
        (2, 1),
    ],
)
def test_retrieve_rejects_candidate_k_smaller_than_top_k(
    db: Session,
    top_k: int,
    candidate_k: int,
) -> None:
    """
    验证candidate_k不能小于top_k。
    """

    service = RetrievalService(
        embedding_provider=FakeEmbeddingProvider(),
        vector_store=FakeVectorStore(),
    )

    with pytest.raises(
        ValueError,
        match="candidate_k",
    ):
        service.retrieve(
            db=db,
            query="测试问题",
            top_k=top_k,
            candidate_k=candidate_k,
        )

@pytest.mark.parametrize(
    "candidate_k",
    [
        0,
        -1,
        True,
        1.5,
    ],
)
def test_retrieve_rejects_invalid_candidate_k(
    db: Session,
    candidate_k: object,
) -> None:
    """
    验证非法候选数量被拒绝。
    """

    service = RetrievalService(
        embedding_provider=FakeEmbeddingProvider(),
        vector_store=FakeVectorStore(),
    )

    with pytest.raises(
        ValueError,
        match="candidate_k must be a positive integer",
    ):
        service.retrieve(
            db=db,
            query="测试问题",
            candidate_k=candidate_k,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "per_document_limit",
    [
        0,
        -1,
        True,
        1.5,
    ],
)
def test_retrieve_rejects_invalid_per_document_limit(
    db: Session,
    per_document_limit: object,
) -> None:
    """
    验证非法单文档数量限制被拒绝。
    """

    service = RetrievalService(
        embedding_provider=FakeEmbeddingProvider(),
        vector_store=FakeVectorStore(),
    )

    with pytest.raises(
        ValueError,
        match="per_document_limit must be a positive integer",
    ):
        service.retrieve(
            db=db,
            query="测试问题",
            per_document_limit=per_document_limit,  # type: ignore[arg-type]
        )



def test_retrieve_baseline_reproduces_original_top_k(
    db: Session,
) -> None:
    """
    验证Baseline只执行原始Top-K召回。

    Baseline不执行重复内容过滤，
    也不执行多文档平衡。
    """

    vector_store = MultiDocumentVectorStore(
        results=[
            build_search_result(
                document_id=1,
                chunk_id=1,
                content="管理员可以重置密码。",
                score=0.99,
            ),
            build_search_result(
                document_id=1,
                chunk_id=2,
                content=" 管理员可以重置密码。 ",
                score=0.98,
            ),
            build_search_result(
                document_id=1,
                chunk_id=3,
                content="密码策略由管理员配置。",
                score=0.97,
            ),
            build_search_result(
                document_id=2,
                chunk_id=4,
                content="普通用户不能修改密码策略。",
                score=0.96,
            ),
        ]
    )

    service = RetrievalService(
        embedding_provider=FakeEmbeddingProvider(),
        vector_store=vector_store,
    )

    results = service.retrieve(
        db=db,
        query="如何重置密码？",
        top_k=3,
        candidate_k=4,
        per_document_limit=1,
        retrieval_mode="baseline",
    )

    # Baseline向VectorStore请求的是Top-K，
    # 而不是Candidate-K。
    assert vector_store.received_top_k == 3

    # 重复内容仍然保留。
    assert [
        result.chunk_id
        for result in results
    ] == [
        1,
        2,
        3,
    ]

    # 单个文档可以占满全部结果。
    assert {
        result.document_id
        for result in results
    } == {
        1,
    }


def test_retrieve_baseline_only_filters_by_score(
    db: Session,
) -> None:
    """
    验证Baseline只额外执行相似度阈值过滤。
    """

    vector_store = MultiDocumentVectorStore(
        results=[
            build_search_result(
                document_id=1,
                chunk_id=1,
                content="高相关内容",
                score=0.90,
            ),
            build_search_result(
                document_id=1,
                chunk_id=2,
                content="低相关内容",
                score=0.40,
            ),
            build_search_result(
                document_id=2,
                chunk_id=3,
                content="中等相关内容",
                score=0.70,
            ),
        ]
    )

    service = RetrievalService(
        embedding_provider=FakeEmbeddingProvider(),
        vector_store=vector_store,
    )

    results = service.retrieve(
        db=db,
        query="测试问题",
        top_k=3,
        score_threshold=0.60,
        retrieval_mode="baseline",
    )

    assert [
        result.chunk_id
        for result in results
    ] == [
        1,
        3,
    ]


def test_retrieve_modes_produce_different_results(
    db: Session,
) -> None:
    """
    验证相同候选数据下两种模式产生不同结果。
    """

    search_results = [
        build_search_result(
            document_id=1,
            chunk_id=1,
            content="文档一第一条",
            score=0.99,
        ),
        build_search_result(
            document_id=1,
            chunk_id=2,
            content="文档一第二条",
            score=0.98,
        ),
        build_search_result(
            document_id=1,
            chunk_id=3,
            content="文档一第三条",
            score=0.97,
        ),
        build_search_result(
            document_id=2,
            chunk_id=4,
            content="文档二内容",
            score=0.96,
        ),
        build_search_result(
            document_id=3,
            chunk_id=5,
            content="文档三内容",
            score=0.95,
        ),
    ]

    vector_store = MultiDocumentVectorStore(
        results=search_results
    )

    service = RetrievalService(
        embedding_provider=FakeEmbeddingProvider(),
        vector_store=vector_store,
    )

    baseline_results = service.retrieve(
        db=db,
        query="多文档问题",
        top_k=3,
        candidate_k=5,
        per_document_limit=1,
        retrieval_mode="baseline",
    )

    optimized_results = service.retrieve(
        db=db,
        query="多文档问题",
        top_k=3,
        candidate_k=5,
        per_document_limit=1,
        retrieval_mode="optimized",
    )

    assert [
        result.chunk_id
        for result in baseline_results
    ] == [
        1,
        2,
        3,
    ]

    assert [
        result.chunk_id
        for result in optimized_results
    ] == [
        1,
        4,
        5,
    ]


def test_retrieve_uses_optimized_mode_by_default(
    db: Session,
) -> None:
    """
    验证正常业务调用默认使用优化模式。
    """

    vector_store = MultiDocumentVectorStore(
        results=[
            build_search_result(
                document_id=1,
                chunk_id=1,
                content="文档一第一条",
                score=0.99,
            ),
            build_search_result(
                document_id=1,
                chunk_id=2,
                content="文档一第二条",
                score=0.98,
            ),
            build_search_result(
                document_id=2,
                chunk_id=3,
                content="文档二内容",
                score=0.97,
            ),
        ]
    )

    service = RetrievalService(
        embedding_provider=FakeEmbeddingProvider(),
        vector_store=vector_store,
    )

    results = service.retrieve(
        db=db,
        query="测试默认检索模式",
        top_k=2,
        candidate_k=3,
        per_document_limit=1,
    )

    assert [
        result.chunk_id
        for result in results
    ] == [
        1,
        3,
    ]


def test_retrieve_rejects_invalid_mode(
    db: Session,
) -> None:
    """
    验证非法检索模式被拒绝。
    """

    service = RetrievalService(
        embedding_provider=FakeEmbeddingProvider(),
        vector_store=FakeVectorStore(),
    )

    with pytest.raises(
        ValueError,
        match="retrieval_mode",
    ):
        service.retrieve(
            db=db,
            query="测试问题",
            retrieval_mode="unknown",  # type: ignore[arg-type]
        )