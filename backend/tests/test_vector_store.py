from collections.abc import Sequence
from typing import Literal

import pytest
from qdrant_client import QdrantClient, models
from sqlalchemy.orm import Session

from app.constants.document_status import DocumentStatus
from app.constants.embedding_status import EmbeddingStatus
from app.core.config import Settings
from app.models.database.chunk_embedding import ChunkEmbedding
from app.models.database.document import Document
from app.models.database.document_chunk import DocumentChunk
from app.models.database.document_content import DocumentContent
from app.repositories.chunk_embedding_repository import ChunkEmbeddingRepository
from app.repositories.document_repository import DocumentRepository
from app.services.vector_index_service import VectorIndexService
from app.services.vector_store.base import VectorIndex, VectorIndexRecord
from app.services.vector_store.database import DatabaseVectorStore
from app.services.vector_store.factory import VectorStoreFactory
from app.services.vector_store.qdrant import QdrantVectorStore

class FakeVectorIndex(VectorIndex):
    """记录待写入数据的测试向量索引。"""

    def __init__(self) -> None:
        self.records: list[VectorIndexRecord] = []

    def ensure_collection(self) -> None:
        pass

    def upsert(self, records: Sequence[VectorIndexRecord]) -> None:
        self.records = list(records)

    def delete_by_document_id(self, document_id: int) -> None:
        pass

def build_vector_store_settings(
    backend: Literal["database", "qdrant"],
    embedding_dimension: int = 2,
) -> Settings:
    """创建向量存储Factory测试配置。"""

    return Settings(
        model_provider="test",
        model_base_url="http://test-llm",
        model_name="test-model",
        model_api_key="test-key",
        embedding_provider="test",
        embedding_base_url="http://test-embedding",
        embedding_model="test-embedding-model",
        embedding_api_key="test-key",
        embedding_dimension=embedding_dimension,
        vector_store_backend=backend,
    )

def test_database_vector_store_returns_top_k(
    db: Session,
) -> None:
    """
    验证向量检索按照余弦相似度返回Top-K结果。
    """

    document = Document(
        filename="vector-search.txt",
        stored_name="vector-search-stored.txt",
        path="uploads/vector-search-stored.txt",
        size=100,
        status=DocumentStatus.COMPLETED.value,
    )

    db.add(document)
    db.flush()

    document_content = DocumentContent(
        document_id=document.id,
        content="向量检索测试全文",
        parser_type="txt_parser",
        parser_version="1.0",
    )

    db.add(document_content)
    db.flush()

    chunks = [
        DocumentChunk(
            document_content_id=document_content.id,
            chunk_index=0,
            content="企业知识库可以帮助员工快速检索内部资料。",
            token_count=20,
            chunk_strategy="recursive_character",
            embedding_status=(
                EmbeddingStatus.COMPLETED.value
            ),
        ),
        DocumentChunk(
            document_content_id=document_content.id,
            chunk_index=1,
            content="RAG通过召回相关文本辅助大模型回答。",
            token_count=18,
            chunk_strategy="recursive_character",
            embedding_status=(
                EmbeddingStatus.COMPLETED.value
            ),
        ),
        DocumentChunk(
            document_content_id=document_content.id,
            chunk_index=2,
            content="今天的天气非常晴朗。",
            token_count=10,
            chunk_strategy="recursive_character",
            embedding_status=(
                EmbeddingStatus.COMPLETED.value
            ),
        ),
    ]

    db.add_all(chunks)
    db.flush()

    embeddings = [
        ChunkEmbedding(
            document_chunk_id=chunks[0].id,
            vector=[1.0, 0.0],
            embedding_model="test-model",
            embedding_dimension=2,
            embedding_metadata=None,
        ),
        ChunkEmbedding(
            document_chunk_id=chunks[1].id,
            vector=[0.8, 0.2],
            embedding_model="test-model",
            embedding_dimension=2,
            embedding_metadata=None,
        ),
        ChunkEmbedding(
            document_chunk_id=chunks[2].id,
            vector=[0.0, 1.0],
            embedding_model="test-model",
            embedding_dimension=2,
            embedding_metadata=None,
        ),
    ]

    db.add_all(embeddings)
    db.commit()

    vector_store = DatabaseVectorStore(
        chunk_embedding_repository=(
            ChunkEmbeddingRepository()
        )
    )

    results = vector_store.search(
        db=db,
        query_vector=[1.0, 0.0],
        embedding_model="test-model",
        top_k=2,
    )

    assert len(results) == 2

    assert results[0].chunk_id == chunks[0].id
    assert results[1].chunk_id == chunks[1].id

    assert results[0].score == pytest.approx(
        1.0
    )

    assert (
        results[0].score
        > results[1].score
    )

    assert (
        results[0].document_id
        == document.id
    )


def test_database_vector_store_rejects_dimension_mismatch(
    db: Session,
) -> None:
    """
    验证查询向量与文档向量维度不一致时抛出异常。
    """

    document = Document(
        filename="dimension-test.txt",
        stored_name="dimension-test-stored.txt",
        path="uploads/dimension-test-stored.txt",
        size=100,
        status=DocumentStatus.COMPLETED.value,
    )

    db.add(document)
    db.flush()

    document_content = DocumentContent(
        document_id=document.id,
        content="维度检查测试全文",
        parser_type="txt_parser",
        parser_version="1.0",
    )

    db.add(document_content)
    db.flush()

    chunk = DocumentChunk(
        document_content_id=document_content.id,
        chunk_index=0,
        content="这是一个二维向量对应的文本。",
        token_count=10,
        chunk_strategy="recursive_character",
        embedding_status=(
            EmbeddingStatus.COMPLETED.value
        ),
    )

    db.add(chunk)
    db.flush()

    embedding = ChunkEmbedding(
        document_chunk_id=chunk.id,
        vector=[1.0, 0.0],
        embedding_model="test-model",
        embedding_dimension=2,
        embedding_metadata=None,
    )

    db.add(embedding)
    db.commit()

    vector_store = DatabaseVectorStore(
        chunk_embedding_repository=(
            ChunkEmbeddingRepository()
        )
    )

    with pytest.raises(
        RuntimeError,
        match="dimension does not match",
    ):
        vector_store.search(
            db=db,
            query_vector=[
                1.0,
                0.0,
                0.0,
            ],
            embedding_model="test-model",
            top_k=1,
        )

@pytest.mark.filterwarnings(
    "ignore:Payload indexes have no effect "
    "in the local Qdrant.*:UserWarning"
)
def test_qdrant_vector_store_manages_points(
    db: Session,
) -> None:
    """
    验证Qdrant写入、过滤、幂等更新和删除。
    """

    client = QdrantClient(":memory:")

    try:
        vector_store = QdrantVectorStore(
            client=client,
            collection_name="test_chunks",
            vector_size=2,
        )

        vector_store.ensure_collection()

        vector_store.upsert(
            [
                VectorIndexRecord(
                    chunk_id=1,
                    document_id=1,
                    knowledge_base_id=10,
                    chunk_index=0,
                    filename="document-1.txt",
                    content="知识库文档一",
                    embedding_model="test-model",
                    vector=[1.0, 0.0],
                ),
                VectorIndexRecord(
                    chunk_id=2,
                    document_id=2,
                    knowledge_base_id=20,
                    chunk_index=0,
                    filename="document-2.txt",
                    content="知识库文档二",
                    embedding_model="test-model",
                    vector=[0.0, 1.0],
                ),
            ]
        )

        results = vector_store.search(
            db=db,
            query_vector=[1.0, 0.0],
            embedding_model="test-model",
            top_k=5,
            document_id=1,
        )

        assert len(results) == 1
        assert results[0].chunk_id == 1
        assert results[0].document_id == 1
        assert results[0].filename == (
            "document-1.txt"
        )

        kb_results = vector_store.search(
            db=db,
            query_vector=[1.0, 0.0],
            embedding_model="test-model",
            top_k=5,
            knowledge_base_id=20,
        )
        assert len(kb_results) == 1
        assert kb_results[0].document_id == 2

        # 相同Chunk ID再次Upsert，不应产生重复Point。
        vector_store.upsert(
            [
                VectorIndexRecord(
                    chunk_id=1,
                    document_id=1,
                    knowledge_base_id=10,
                    chunk_index=0,
                    filename="document-1.txt",
                    content="更新后的知识库文档一",
                    embedding_model="test-model",
                    vector=[0.8, 0.2],
                )
            ]
        )

        point_count = client.count(
            collection_name="test_chunks",
            exact=True,
        )

        assert point_count.count == 2

        vector_store.delete_by_document_id(
            document_id=1,
        )

        deleted_results = vector_store.search(
            db=db,
            query_vector=[1.0, 0.0],
            embedding_model="test-model",
            top_k=5,
            document_id=1,
        )

        assert deleted_results == []

        remaining_results = vector_store.search(
            db=db,
            query_vector=[0.0, 1.0],
            embedding_model="test-model",
            top_k=5,
        )

        assert len(remaining_results) == 1
        assert (
            remaining_results[0].document_id
            == 2
        )

    finally:
        client.close()


def test_qdrant_vector_store_rejects_dimension_mismatch(
) -> None:
    """
    验证已有Collection维度错误时拒绝启动。
    """

    client = QdrantClient(":memory:")

    try:
        client.create_collection(
            collection_name="wrong_dimension",
            vectors_config=models.VectorParams(
                size=3,
                distance=models.Distance.COSINE,
            ),
        )

        vector_store = QdrantVectorStore(
            client=client,
            collection_name="wrong_dimension",
            vector_size=2,
        )

        with pytest.raises(
            RuntimeError,
            match="dimension",
        ):
            vector_store.ensure_collection()

    finally:
        client.close()

def test_vector_index_service_builds_records_from_sql(db: Session) -> None:
    """验证服务从SQL读取向量并构造索引记录。"""

    document = Document(
        filename="index-test.txt",
        stored_name="index-test-stored.txt",
        path="uploads/index-test-stored.txt",
        size=100,
        status=DocumentStatus.COMPLETED.value,
    )

    db.add(document)
    db.flush()

    document_content = DocumentContent(
        document_id=document.id,
        content="向量索引测试全文",
        parser_type="txt_parser",
        parser_version="1.0",
    )

    db.add(document_content)
    db.flush()

    chunk = DocumentChunk(
        document_content_id=document_content.id,
        chunk_index=0,
        content="管理员可以在系统设置中重置密码。",
        token_count=18,
        chunk_strategy="recursive_character",
        embedding_status=EmbeddingStatus.COMPLETED.value,
    )

    db.add(chunk)
    db.flush()

    embedding = ChunkEmbedding(
        document_chunk_id=chunk.id,
        vector=[1.0, 0.0],
        embedding_model="test-model",
        embedding_dimension=2,
        embedding_metadata=None,
    )

    db.add(embedding)
    db.commit()

    vector_index = FakeVectorIndex()

    service = VectorIndexService(
        document_repository=DocumentRepository(),
        chunk_embedding_repository=ChunkEmbeddingRepository(),
        vector_index=vector_index,
    )

    indexed_count = service.index_document(db=db, document_id=document.id)

    assert indexed_count == 1
    assert len(vector_index.records) == 1

    record = vector_index.records[0]

    assert record.chunk_id == chunk.id
    assert record.document_id == document.id
    assert record.chunk_index == 0
    assert record.filename == "index-test.txt"
    assert record.content == "管理员可以在系统设置中重置密码。"
    assert record.embedding_model == "test-model"
    assert list(record.vector) == [1.0, 0.0]

def test_vector_store_factory_creates_database_components() -> None:
    """验证database配置只提供数据库检索实现。"""

    components = VectorStoreFactory.create(
        settings=build_vector_store_settings("database")
    )

    assert isinstance(components.vector_store, DatabaseVectorStore)
    assert components.vector_index is None

def test_vector_store_factory_reuses_qdrant_for_search_and_index() -> None:
    """验证Qdrant查询和索引共享同一个适配器实例。"""

    client = QdrantClient(":memory:")

    try:
        components = VectorStoreFactory.create(
            settings=build_vector_store_settings("qdrant"),
            qdrant_client=client,
        )

        assert isinstance(components.vector_store, QdrantVectorStore)
        assert components.vector_index is components.vector_store

    finally:
        client.close()