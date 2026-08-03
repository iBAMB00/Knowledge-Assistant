from sqlalchemy.orm import Session

from app.constants.document_status import DocumentStatus
from app.constants.embedding_status import EmbeddingStatus
from app.models.database.chunk_embedding import ChunkEmbedding
from app.models.database.document import Document
from app.models.database.document_chunk import DocumentChunk
from app.models.database.document_content import DocumentContent
from app.repositories.chunk_embedding_repository import (
    ChunkEmbeddingRepository,
)
from app.repositories.document_chunk_repository import (
    DocumentChunkRepository,
)
from app.repositories.document_repository import DocumentRepository
from app.services.embedding.mock import MockEmbeddingProvider
from app.services.embedding_service import EmbeddingService


def test_mock_embedding_provider_is_deterministic() -> None:
    """
    验证Mock Embedding结果稳定且维度正确。
    """

    provider = MockEmbeddingProvider(
        dimension=8,
    )

    text = "企业知识库文档测试"

    first_vector = provider.embed_query(text)
    second_vector = provider.embed_query(text)

    assert first_vector == second_vector
    assert len(first_vector) == 8
    assert provider.model_name == "mock-sha256"

    assert all(
        -1.0 <= value <= 1.0
        for value in first_vector
    )

    vectors = provider.embed_documents([
        text,
        "第二个知识切片",
    ])

    assert len(vectors) == 2
    assert all(
        len(vector) == 8
        for vector in vectors
    )


def test_process_document_with_mock_embedding(
    db: Session,
) -> None:
    """
    验证Mock Embedding完整持久化流程。

    测试前置条件：
    - 文档已经完成解析和切片
    - Document状态为chunked
    - Chunk状态为pending
    """

    document = Document(
        filename="embedding-test.txt",
        stored_name="embedding-test-stored.txt",
        path="tests/uploads/embedding-test-stored.txt",
        size=100,
        status=DocumentStatus.CHUNKED.value,
    )

    db.add(document)
    db.flush()

    document_content = DocumentContent(
        document_id=document.id,
        content=(
            "Knowledge Assistant用于企业私有知识库问答。"
            "系统会将文本切片转换为向量。"
        ),
        parser_type="txt",
        parser_version="1.0",
    )

    db.add(document_content)
    db.flush()

    chunks = [
        DocumentChunk(
            document_content_id=document_content.id,
            chunk_index=0,
            content="Knowledge Assistant用于企业私有知识库问答。",
            token_count=None,
            chunk_strategy="recursive_character",
            embedding_status=EmbeddingStatus.PENDING.value,
            chunk_metadata=None,
        ),
        DocumentChunk(
            document_content_id=document_content.id,
            chunk_index=1,
            content="系统会将文本切片转换为向量。",
            token_count=None,
            chunk_strategy="recursive_character",
            embedding_status=EmbeddingStatus.PENDING.value,
            chunk_metadata=None,
        ),
        DocumentChunk(
            document_content_id=document_content.id,
            chunk_index=2,
            content="向量将用于后续语义检索和RAG问答。",
            token_count=None,
            chunk_strategy="recursive_character",
            embedding_status=EmbeddingStatus.PENDING.value,
            chunk_metadata=None,
        ),
    ]

    db.add_all(chunks)
    db.commit()

    document_repository = DocumentRepository()
    document_chunk_repository = DocumentChunkRepository()
    chunk_embedding_repository = ChunkEmbeddingRepository()

    embedding_service = EmbeddingService(
        document_repository=document_repository,
        document_chunk_repository=(
            document_chunk_repository
        ),
        chunk_embedding_repository=(
            chunk_embedding_repository
        ),
        embedding_provider=MockEmbeddingProvider(
            dimension=8,
        ),
    )

    processed_count = embedding_service.process_document(
        db=db,
        document_id=document.id,
        batch_size=2,
    )

    assert processed_count == len(chunks)

    db.expire_all()

    saved_document = document_repository.find_by_id(
        db=db,
        document_id=document.id,
    )

    assert saved_document is not None
    assert (
        saved_document.status
        == DocumentStatus.COMPLETED.value
    )

    saved_chunks = (
        document_chunk_repository.find_by_document_id(
            db=db,
            document_id=document.id,
        )
    )

    assert len(saved_chunks) == len(chunks)

    for chunk in saved_chunks:
        assert (
            chunk.embedding_status
            == EmbeddingStatus.COMPLETED.value
        )

        embedding = (
            chunk_embedding_repository.find_by_chunk_id(
                db=db,
                document_chunk_id=chunk.id,
            )
        )

        assert embedding is not None
        assert embedding.document_chunk_id == chunk.id
        assert embedding.embedding_model == "mock-sha256"
        assert embedding.embedding_dimension == 8
        assert len(embedding.vector) == 8

    embedding_count = (
        db.query(ChunkEmbedding)
        .filter(
            ChunkEmbedding.document_chunk_id.in_(
                [chunk.id for chunk in saved_chunks]
            )
        )
        .count()
    )

    assert embedding_count == len(saved_chunks)

    # 已完成文档再次执行，不应重复生成向量。
    second_processed_count = (
        embedding_service.process_document(
            db=db,
            document_id=document.id,
            batch_size=2,
        )
    )

    assert second_processed_count == 0

    second_embedding_count = (
        db.query(ChunkEmbedding)
        .filter(
            ChunkEmbedding.document_chunk_id.in_(
                [chunk.id for chunk in saved_chunks]
            )
        )
        .count()
    )

    assert second_embedding_count == embedding_count


def test_process_document_retries_failed_chunks(
    db: Session,
) -> None:
    """
    验证向量化失败的Chunk可以重新处理。
    """

    document = Document(
        filename="embedding-retry.txt",
        stored_name="embedding-retry-stored.txt",
        path="tests/uploads/embedding-retry-stored.txt",
        size=100,
        status=DocumentStatus.EMBEDDING_FAILED.value,
    )

    db.add(document)
    db.flush()

    document_content = DocumentContent(
        document_id=document.id,
        content="向量化重试测试全文。",
        parser_type="txt",
        parser_version="1.0",
    )

    db.add(document_content)
    db.flush()

    completed_chunk = DocumentChunk(
        document_content_id=document_content.id,
        chunk_index=0,
        content="已经完成向量化的切片。",
        token_count=None,
        chunk_strategy="recursive_character",
        embedding_status=EmbeddingStatus.COMPLETED.value,
        chunk_metadata=None,
    )

    failed_chunk = DocumentChunk(
        document_content_id=document_content.id,
        chunk_index=1,
        content="需要重新向量化的切片。",
        token_count=None,
        chunk_strategy="recursive_character",
        embedding_status=EmbeddingStatus.FAILED.value,
        chunk_metadata=None,
    )

    db.add_all([
        completed_chunk,
        failed_chunk,
    ])
    db.flush()

    existing_embedding = ChunkEmbedding(
        document_chunk_id=completed_chunk.id,
        vector=[0.1] * 8,
        embedding_model="mock-sha256",
        embedding_dimension=8,
        embedding_metadata=None,
    )

    db.add(existing_embedding)
    db.commit()

    document_repository = DocumentRepository()
    document_chunk_repository = DocumentChunkRepository()
    chunk_embedding_repository = ChunkEmbeddingRepository()

    embedding_service = EmbeddingService(
        document_repository=document_repository,
        document_chunk_repository=(
            document_chunk_repository
        ),
        chunk_embedding_repository=(
            chunk_embedding_repository
        ),
        embedding_provider=MockEmbeddingProvider(
            dimension=8,
        ),
    )

    processed_count = (
        embedding_service.process_document(
            db=db,
            document_id=document.id,
        )
    )

    assert processed_count == 1

    db.expire_all()

    saved_document = document_repository.find_by_id(
        db=db,
        document_id=document.id,
    )

    assert saved_document is not None
    assert (
        saved_document.status
        == DocumentStatus.COMPLETED.value
    )

    saved_completed_chunk = db.get(
        DocumentChunk,
        completed_chunk.id,
    )

    saved_failed_chunk = db.get(
        DocumentChunk,
        failed_chunk.id,
    )

    assert saved_completed_chunk is not None
    assert saved_failed_chunk is not None

    assert (
        saved_completed_chunk.embedding_status
        == EmbeddingStatus.COMPLETED.value
    )

    assert (
        saved_failed_chunk.embedding_status
        == EmbeddingStatus.COMPLETED.value
    )

    retry_embedding = (
        chunk_embedding_repository.find_by_chunk_id(
            db=db,
            document_chunk_id=failed_chunk.id,
        )
    )

    assert retry_embedding is not None
    assert retry_embedding.embedding_dimension == 8

    embedding_count = (
        db.query(ChunkEmbedding)
        .filter(
            ChunkEmbedding.document_chunk_id.in_(
                [
                    completed_chunk.id,
                    failed_chunk.id,
                ]
            )
        )
        .count()
    )

    assert embedding_count == 2