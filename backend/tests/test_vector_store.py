import pytest
from sqlalchemy.orm import Session

from app.constants.document_status import DocumentStatus
from app.constants.embedding_status import EmbeddingStatus
from app.models.database.chunk_embedding import ChunkEmbedding
from app.models.database.document import Document
from app.models.database.document_chunk import DocumentChunk
from app.models.database.document_content import DocumentContent
from app.repositories.chunk_embedding_repository import ChunkEmbeddingRepository
from app.services.vector_store.database import DatabaseVectorStore


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