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
                filename="test-document.txt",
                chunk_id=1,
                chunk_index=0,
                content="第一条高相关文本",
                score=0.90,
            ),
            VectorSearchResult(
                document_id=1,
                filename="test-document.txt",
                chunk_id=2,
                chunk_index=1,
                content="第二条中等相关文本",
                score=0.65,
            ),
            VectorSearchResult(
                document_id=2,
                filename="test-document.txt",
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
    filename: str = "test-document.txt",
) -> VectorSearchResult:
    """
    创建检索结果。
    """

    return VectorSearchResult(
        document_id=document_id,
        filename=filename,
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

class ParentChildVectorStore(VectorStore):
    """Parent-Child检索测试使用的向量存储。"""

    def __init__(
        self,
        child_results: list[VectorSearchResult],
        parent_results: list[VectorSearchResult],
    ) -> None:
        self.child_results = child_results
        self.parent_results = parent_results
        self.received_roles: list[str | None] = []

    def search(
        self,
        db: Session,
        query_vector: Sequence[float],
        embedding_model: str,
        top_k: int = 5,
        document_id: int | None = None,
        chunk_role: str | None = None,
    ) -> list[VectorSearchResult]:
        del db, query_vector, embedding_model, document_id
        self.received_roles.append(chunk_role)

        if chunk_role == "child":
            return self.child_results[:top_k]
        return self.parent_results[:top_k]


def test_parent_child_retrieval_uses_child_and_returns_parent_context(
    db: Session,
) -> None:
    """验证Child负责召回，同一Parent只返回一次并扩展Parent正文。"""

    from app.models.database.document import Document
    from app.models.database.document_content import DocumentContent
    from app.models.database.document_chunk import DocumentChunk
    from app.repositories.document_chunk_repository import DocumentChunkRepository

    document = Document(
        filename="parent-child.txt",
        stored_name="parent-child-stored.txt",
        path="uploads/parent-child-stored.txt",
        size=100,
        status="completed",
    )
    db.add(document)
    db.flush()

    document_content = DocumentContent(
        document_id=document.id,
        content="完整正文",
        parser_type="txt",
        parser_version="1.0",
    )
    db.add(document_content)
    db.flush()

    parent_one = DocumentChunk(
        document_content_id=document_content.id,
        chunk_index=0,
        content="父块一：这里包含更完整的上下文。",
        chunk_strategy="recursive_character",
    )
    parent_two = DocumentChunk(
        document_content_id=document_content.id,
        chunk_index=1,
        content="父块二：这是另一个完整上下文。",
        chunk_strategy="recursive_character",
    )
    db.add_all([parent_one, parent_two])
    db.flush()

    child_one = DocumentChunk(
        document_content_id=document_content.id,
        chunk_index=2,
        content="细粒度命中一",
        chunk_strategy="recursive_character",
        parent_chunk_id=parent_one.id,
    )
    child_two = DocumentChunk(
        document_content_id=document_content.id,
        chunk_index=3,
        content="细粒度命中二",
        chunk_strategy="recursive_character",
        parent_chunk_id=parent_one.id,
    )
    child_three = DocumentChunk(
        document_content_id=document_content.id,
        chunk_index=4,
        content="另一个父块的命中",
        chunk_strategy="recursive_character",
        parent_chunk_id=parent_two.id,
    )
    db.add_all([child_one, child_two, child_three])
    db.commit()

    vector_store = ParentChildVectorStore(
        child_results=[
            VectorSearchResult(
                document_id=document.id,
                filename=document.filename,
                chunk_id=child_one.id,
                chunk_index=child_one.chunk_index,
                content=child_one.content,
                score=0.95,
                parent_chunk_id=parent_one.id,
            ),
            VectorSearchResult(
                document_id=document.id,
                filename=document.filename,
                chunk_id=child_two.id,
                chunk_index=child_two.chunk_index,
                content=child_two.content,
                score=0.90,
                parent_chunk_id=parent_one.id,
            ),
            VectorSearchResult(
                document_id=document.id,
                filename=document.filename,
                chunk_id=child_three.id,
                chunk_index=child_three.chunk_index,
                content=child_three.content,
                score=0.85,
                parent_chunk_id=parent_two.id,
            ),
        ],
        parent_results=[],
    )

    service = RetrievalService(
        embedding_provider=FakeEmbeddingProvider(),
        vector_store=vector_store,
        document_chunk_repository=DocumentChunkRepository(),
        parent_child_enabled=True,
        default_top_k=2,
        default_candidate_k=5,
    )

    results = service.retrieve(
        db=db,
        query="测试Parent-Child",
        top_k=2,
        candidate_k=5,
        document_id=document.id,
    )

    assert vector_store.received_roles == ["child"]
    assert [result.chunk_id for result in results] == [
        child_one.id,
        child_three.id,
    ]
    assert [result.parent_chunk_id for result in results] == [
        parent_one.id,
        parent_two.id,
    ]
    assert [result.content for result in results] == [
        parent_one.content,
        parent_two.content,
    ]


class FakeBM25Retriever:
    """Hybrid 检索测试使用的 BM25 替身。"""

    def __init__(self, results: list[VectorSearchResult]) -> None:
        self.results = results
        self.received_query: str | None = None
        self.received_role: str | None = None

    def search(
        self,
        db: Session,
        query: str,
        top_k: int = 20,
        document_id: int | None = None,
        chunk_role: str | None = None,
    ) -> list[VectorSearchResult]:
        del db, document_id
        self.received_query = query
        self.received_role = chunk_role
        return self.results[:top_k]


def test_rrf_fusion_rewards_candidates_found_by_both_retrievers() -> None:
    """验证 Dense 和 BM25 都命中的 Chunk 在 RRF 中优先。"""

    from app.services.rrf_fusion_service import RRFFusionService

    dense = [
        build_search_result(1, 1, "dense first", 0.95),
        build_search_result(1, 2, "shared", 0.90),
    ]
    lexical = [
        build_search_result(1, 2, "shared", 8.0),
        build_search_result(1, 3, "lexical only", 6.0),
    ]

    results = RRFFusionService(rank_constant=60).fuse(
        rankings=[dense, lexical],
        top_k=3,
    )

    assert [result.chunk_id for result in results] == [2, 1, 3]
    assert 0 < results[0].score <= 1


def test_bm25_retrieval_ranks_exact_chinese_keyword_first(
    db: Session,
) -> None:
    """验证轻量 BM25 能优先召回包含中文专有关键词的 Child。"""

    from app.constants.embedding_status import EmbeddingStatus
    from app.models.database.document import Document
    from app.models.database.document_content import DocumentContent
    from app.models.database.document_chunk import DocumentChunk
    from app.repositories.document_chunk_repository import DocumentChunkRepository
    from app.services.bm25_retrieval_service import BM25RetrievalService

    document = Document(
        filename="bm25.txt",
        stored_name="bm25-stored.txt",
        path="uploads/bm25-stored.txt",
        size=100,
        status="completed",
    )
    db.add(document)
    db.flush()

    content = DocumentContent(
        document_id=document.id,
        content="完整正文",
        parser_type="txt",
        parser_version="1.0",
    )
    db.add(content)
    db.flush()

    parent = DocumentChunk(
        document_content_id=content.id,
        chunk_index=0,
        content="父块",
        chunk_strategy="recursive_character",
        embedding_status=EmbeddingStatus.COMPLETED.value,
    )
    db.add(parent)
    db.flush()

    exact = DocumentChunk(
        document_content_id=content.id,
        chunk_index=1,
        content="API 接口需要实现限流、幂等和鉴权。",
        chunk_strategy="recursive_character",
        embedding_status=EmbeddingStatus.COMPLETED.value,
        parent_chunk_id=parent.id,
    )
    unrelated = DocumentChunk(
        document_content_id=content.id,
        chunk_index=2,
        content="系统支持日志导出和归档。",
        chunk_strategy="recursive_character",
        embedding_status=EmbeddingStatus.COMPLETED.value,
        parent_chunk_id=parent.id,
    )
    db.add_all([exact, unrelated])
    db.commit()

    service = BM25RetrievalService(DocumentChunkRepository())
    results = service.search(
        db=db,
        query="API 幂等",
        top_k=2,
        document_id=document.id,
        chunk_role="child",
    )

    assert results
    assert results[0].chunk_id == exact.id
    assert results[0].score > 0
    assert all(result.parent_chunk_id == parent.id for result in results)


def test_hybrid_parent_child_retrieval_fuses_before_parent_expansion(
    db: Session,
) -> None:
    """验证 Hybrid 先融合 Child 排名，再扩展为 Parent 上下文。"""

    from app.models.database.document import Document
    from app.models.database.document_content import DocumentContent
    from app.models.database.document_chunk import DocumentChunk
    from app.repositories.document_chunk_repository import DocumentChunkRepository
    from app.services.rrf_fusion_service import RRFFusionService

    document = Document(
        filename="hybrid.txt",
        stored_name="hybrid-stored.txt",
        path="uploads/hybrid-stored.txt",
        size=100,
        status="completed",
    )
    db.add(document)
    db.flush()

    content = DocumentContent(
        document_id=document.id,
        content="完整正文",
        parser_type="txt",
        parser_version="1.0",
    )
    db.add(content)
    db.flush()

    parent_one = DocumentChunk(
        document_content_id=content.id,
        chunk_index=0,
        content="父块一完整上下文",
        chunk_strategy="recursive_character",
    )
    parent_two = DocumentChunk(
        document_content_id=content.id,
        chunk_index=1,
        content="父块二完整上下文",
        chunk_strategy="recursive_character",
    )
    db.add_all([parent_one, parent_two])
    db.flush()

    child_dense = DocumentChunk(
        document_content_id=content.id,
        chunk_index=2,
        content="语义召回内容",
        chunk_strategy="recursive_character",
        parent_chunk_id=parent_one.id,
    )
    child_shared = DocumentChunk(
        document_content_id=content.id,
        chunk_index=3,
        content="API 幂等",
        chunk_strategy="recursive_character",
        parent_chunk_id=parent_two.id,
    )
    db.add_all([child_dense, child_shared])
    db.commit()

    vector_store = ParentChildVectorStore(
        child_results=[
            VectorSearchResult(
                document_id=document.id,
                filename=document.filename,
                chunk_id=child_dense.id,
                chunk_index=child_dense.chunk_index,
                content=child_dense.content,
                score=0.95,
                parent_chunk_id=parent_one.id,
            ),
            VectorSearchResult(
                document_id=document.id,
                filename=document.filename,
                chunk_id=child_shared.id,
                chunk_index=child_shared.chunk_index,
                content=child_shared.content,
                score=0.90,
                parent_chunk_id=parent_two.id,
            ),
        ],
        parent_results=[],
    )
    bm25 = FakeBM25Retriever(
        results=[
            VectorSearchResult(
                document_id=document.id,
                filename=document.filename,
                chunk_id=child_shared.id,
                chunk_index=child_shared.chunk_index,
                content=child_shared.content,
                score=9.0,
                parent_chunk_id=parent_two.id,
            )
        ]
    )

    service = RetrievalService(
        embedding_provider=FakeEmbeddingProvider(),
        vector_store=vector_store,
        document_chunk_repository=DocumentChunkRepository(),
        parent_child_enabled=True,
        bm25_retriever=bm25,  # type: ignore[arg-type]
        rrf_fusion_service=RRFFusionService(),
        hybrid_enabled=True,
        default_top_k=2,
        default_candidate_k=5,
    )

    results = service.retrieve(
        db=db,
        query="API 幂等",
        top_k=2,
        candidate_k=5,
        document_id=document.id,
    )

    assert bm25.received_query == "API 幂等"
    assert bm25.received_role == "child"
    assert [result.chunk_id for result in results] == [
        child_shared.id,
        child_dense.id,
    ]
    assert [result.content for result in results] == [
        parent_two.content,
        parent_one.content,
    ]


def test_retrieve_by_vector_keeps_dense_behavior_without_query_text(
    db: Session,
) -> None:
    """评估器只有共享 Query Vector 时暂时保持 Dense 路径。"""

    from app.services.rrf_fusion_service import RRFFusionService

    vector_store = MultiDocumentVectorStore(
        results=[
            build_search_result(1, 1, "dense", 0.9),
        ]
    )
    bm25 = FakeBM25Retriever(
        results=[build_search_result(1, 2, "lexical", 9.0)]
    )

    service = RetrievalService(
        embedding_provider=FakeEmbeddingProvider(),
        vector_store=vector_store,
        bm25_retriever=bm25,  # type: ignore[arg-type]
        rrf_fusion_service=RRFFusionService(),
        hybrid_enabled=True,
    )

    results = service.retrieve_by_vector(
        db=db,
        query_vector=[1.0, 0.0],
        top_k=1,
        candidate_k=2,
    )

    assert [result.chunk_id for result in results] == [1]
    assert bm25.received_query is None

from app.services.reranker.base import RerankItem, RerankerProvider


class FakeReranker(RerankerProvider):
    """RetrievalService 重排序测试使用的 Fake Provider。"""

    def __init__(self, indexes: list[int]) -> None:
        self.indexes = indexes
        self.received_query: str | None = None
        self.received_documents: list[str] = []
        self.received_top_n: int | None = None

    @property
    def model_name(self) -> str:
        return "fake-reranker"

    def rerank(
        self,
        query: str,
        documents: Sequence[str],
        top_n: int,
    ) -> list[RerankItem]:
        self.received_query = query
        self.received_documents = list(documents)
        self.received_top_n = top_n
        return [
            RerankItem(index=index, score=1.0 - rank * 0.1)
            for rank, index in enumerate(self.indexes)
        ][:top_n]


class FailingReranker(RerankerProvider):
    """模拟外部重排序服务异常。"""

    @property
    def model_name(self) -> str:
        return "failing-reranker"

    def rerank(
        self,
        query: str,
        documents: Sequence[str],
        top_n: int,
    ) -> list[RerankItem]:
        raise RuntimeError("reranker unavailable")


def test_retrieve_reranks_candidates_before_final_top_k(
    db: Session,
) -> None:
    """验证 Reranker 能改变候选排序，并在最终 Top-K 前生效。"""

    vector_store = MultiDocumentVectorStore(
        results=[
            build_search_result(1, 1, "第一候选", 0.90),
            build_search_result(1, 2, "第二候选", 0.80),
            build_search_result(1, 3, "第三候选", 0.70),
        ]
    )
    reranker = FakeReranker(indexes=[2, 0, 1])

    service = RetrievalService(
        embedding_provider=FakeEmbeddingProvider(),
        vector_store=vector_store,
        default_top_k=2,
        default_candidate_k=3,
        reranker=reranker,
        reranker_enabled=True,
    )

    results = service.retrieve(
        db=db,
        query="哪个候选最相关？",
        top_k=2,
        candidate_k=3,
        document_id=1,
    )

    assert [result.chunk_id for result in results] == [3, 1]
    assert [result.score for result in results] == pytest.approx([1.0, 0.9])
    assert reranker.received_query == "哪个候选最相关？"
    assert reranker.received_documents == ["第一候选", "第二候选", "第三候选"]
    assert reranker.received_top_n == 3


def test_retrieve_reranker_fail_open_keeps_original_ranking(
    db: Session,
) -> None:
    """验证 Reranker 暂时不可用时可回退到原检索排序。"""

    vector_store = MultiDocumentVectorStore(
        results=[
            build_search_result(1, 1, "第一候选", 0.90),
            build_search_result(1, 2, "第二候选", 0.80),
        ]
    )

    service = RetrievalService(
        embedding_provider=FakeEmbeddingProvider(),
        vector_store=vector_store,
        default_top_k=2,
        default_candidate_k=2,
        reranker=FailingReranker(),
        reranker_enabled=True,
        reranker_fail_open=True,
    )

    results = service.retrieve(
        db=db,
        query="测试问题",
        document_id=1,
    )

    assert [result.chunk_id for result in results] == [1, 2]


def test_retrieve_by_vector_without_query_text_skips_reranker(
    db: Session,
) -> None:
    """验证旧评估路径没有 query 文本时不会错误调用 Reranker。"""

    vector_store = MultiDocumentVectorStore(
        results=[build_search_result(1, 1, "候选", 0.90)]
    )
    reranker = FakeReranker(indexes=[0])

    service = RetrievalService(
        embedding_provider=FakeEmbeddingProvider(),
        vector_store=vector_store,
        default_top_k=1,
        default_candidate_k=1,
        reranker=reranker,
        reranker_enabled=True,
    )

    results = service.retrieve_by_vector(
        db=db,
        query_vector=[1.0, 0.0],
        document_id=1,
    )

    assert [result.chunk_id for result in results] == [1]
    assert reranker.received_query is None


def test_retrieve_logs_stage_timings_without_query_content(
    db: Session,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """验证 Optimized 检索输出分阶段耗时且不记录用户问题正文。"""

    import logging

    from app.services.rrf_fusion_service import RRFFusionService

    secret_query = "企业内部敏感问题XYZ"
    vector_store = MultiDocumentVectorStore(
        results=[
            build_search_result(1, 1, "第一候选", 0.90),
            build_search_result(2, 2, "第二候选", 0.80),
        ]
    )
    bm25 = FakeBM25Retriever(
        results=[
            build_search_result(2, 2, "第二候选", 9.0),
        ]
    )
    reranker = FakeReranker(indexes=[1, 0])

    service = RetrievalService(
        embedding_provider=FakeEmbeddingProvider(),
        vector_store=vector_store,
        default_top_k=2,
        default_candidate_k=2,
        bm25_retriever=bm25,  # type: ignore[arg-type]
        rrf_fusion_service=RRFFusionService(),
        hybrid_enabled=True,
        reranker=reranker,
        reranker_enabled=True,
    )

    caplog.set_level(
        logging.INFO,
        logger="app.services.retrieval_service",
    )

    results = service.retrieve(
        db=db,
        query=secret_query,
        top_k=2,
        candidate_k=2,
    )

    assert [result.chunk_id for result in results] == [1, 2]
    assert "retrieval completed: mode=optimized" in caplog.text
    assert "dense_ms=" in caplog.text
    assert "bm25_ms=" in caplog.text
    assert "rrf_ms=" in caplog.text
    assert "reranker_ms=" in caplog.text
    assert "parent_expand_ms=" in caplog.text
    assert "balance_ms=" in caplog.text
    assert "total_ms=" in caplog.text
    assert "hybrid_applied=True" in caplog.text
    assert "reranker_attempted=True" in caplog.text
    assert secret_query not in caplog.text


def test_database_vector_store_filters_candidates_by_knowledge_base(
    db: Session,
) -> None:
    """验证关系数据库 Dense Baseline 不会跨 KnowledgeBase 召回。"""

    from app.constants.embedding_status import EmbeddingStatus
    from app.models.database.chunk_embedding import ChunkEmbedding
    from app.models.database.document import Document
    from app.models.database.document_chunk import DocumentChunk
    from app.models.database.document_content import DocumentContent
    from app.models.database.knowledge_base import KnowledgeBase
    from app.models.database.user import User
    from app.repositories.chunk_embedding_repository import ChunkEmbeddingRepository
    from app.services.vector_store.database import DatabaseVectorStore

    user = User(
        email="dense-isolation@example.com",
        password_hash="hash",
        role="user",
        is_active=True,
    )
    db.add(user)
    db.flush()
    kb_one = KnowledgeBase(owner_id=user.id, name="Dense KB 1")
    kb_two = KnowledgeBase(owner_id=user.id, name="Dense KB 2")
    db.add_all([kb_one, kb_two])
    db.flush()

    for knowledge_base, filename, vector in [
        (kb_one, "kb-one.txt", [1.0, 0.0]),
        (kb_two, "kb-two.txt", [1.0, 0.0]),
    ]:
        document = Document(
            knowledge_base_id=knowledge_base.id,
            filename=filename,
            stored_name=f"stored-{filename}",
            path=f"uploads/{filename}",
            size=10,
            status="completed",
        )
        db.add(document)
        db.flush()
        content = DocumentContent(
            document_id=document.id,
            content=filename,
            parser_type="txt",
            parser_version="1.0",
        )
        db.add(content)
        db.flush()
        chunk = DocumentChunk(
            document_content_id=content.id,
            chunk_index=0,
            content=filename,
            token_count=2,
            chunk_strategy="recursive_character",
            embedding_status=EmbeddingStatus.COMPLETED.value,
        )
        db.add(chunk)
        db.flush()
        db.add(
            ChunkEmbedding(
                document_chunk_id=chunk.id,
                vector=vector,
                embedding_model="test-model",
                embedding_dimension=2,
            )
        )

    db.commit()

    vector_store = DatabaseVectorStore(ChunkEmbeddingRepository())
    results = vector_store.search(
        db=db,
        query_vector=[1.0, 0.0],
        embedding_model="test-model",
        top_k=10,
        knowledge_base_id=kb_one.id,
    )

    assert len(results) == 1
    assert results[0].filename == "kb-one.txt"


def test_bm25_filters_candidates_by_knowledge_base(db: Session) -> None:
    """验证 Hybrid 的 BM25 分支与 Dense 使用相同知识库边界。"""

    from app.constants.embedding_status import EmbeddingStatus
    from app.models.database.document import Document
    from app.models.database.document_chunk import DocumentChunk
    from app.models.database.document_content import DocumentContent
    from app.models.database.knowledge_base import KnowledgeBase
    from app.models.database.user import User
    from app.repositories.document_chunk_repository import DocumentChunkRepository
    from app.services.bm25_retrieval_service import BM25RetrievalService

    user = User(
        email="bm25-isolation@example.com",
        password_hash="hash",
        role="user",
        is_active=True,
    )
    db.add(user)
    db.flush()
    kb_one = KnowledgeBase(owner_id=user.id, name="BM25 KB 1")
    kb_two = KnowledgeBase(owner_id=user.id, name="BM25 KB 2")
    db.add_all([kb_one, kb_two])
    db.flush()

    for knowledge_base, filename in [
        (kb_one, "owned.txt"),
        (kb_two, "other.txt"),
    ]:
        document = Document(
            knowledge_base_id=knowledge_base.id,
            filename=filename,
            stored_name=f"stored-{filename}",
            path=f"uploads/{filename}",
            size=20,
            status="completed",
        )
        db.add(document)
        db.flush()
        content = DocumentContent(
            document_id=document.id,
            content="统一专有术语 alphaomega",
            parser_type="txt",
            parser_version="1.0",
        )
        db.add(content)
        db.flush()
        db.add(
            DocumentChunk(
                document_content_id=content.id,
                chunk_index=0,
                content="统一专有术语 alphaomega",
                token_count=4,
                chunk_strategy="recursive_character",
                embedding_status=EmbeddingStatus.COMPLETED.value,
            )
        )

    db.commit()

    results = BM25RetrievalService(DocumentChunkRepository()).search(
        db=db,
        query="alphaomega",
        top_k=10,
        knowledge_base_id=kb_one.id,
    )

    assert len(results) == 1
    assert results[0].filename == "owned.txt"
