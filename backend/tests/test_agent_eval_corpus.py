from collections.abc import Sequence

from sqlalchemy.orm import Session

from app.constants.document_status import DocumentStatus
from app.constants.embedding_status import EmbeddingStatus
from app.models.database.document import Document
from app.repositories.chunk_embedding_repository import ChunkEmbeddingRepository
from app.repositories.document_chunk_repository import DocumentChunkRepository
from app.repositories.document_content_repository import DocumentContentRepository
from app.repositories.document_repository import DocumentRepository
from app.repositories.knowledge_base_repository import KnowledgeBaseRepository
from app.repositories.user_repository import UserRepository
from app.services.embedding.base import EmbeddingProvider
from app.services.evaluation.agent_eval_corpus_service import (
    AgentEvaluationCorpusService,
)
from app.services.evaluation.agent_eval_fixture_service import (
    AgentEvaluationFixtureService,
)
from app.services.vector_store.base import VectorIndex, VectorIndexRecord


class FakeEmbeddingProvider(EmbeddingProvider):
    def __init__(self) -> None:
        self.document_calls = 0

    @property
    def model_name(self) -> str:
        return "eval-test-embedding"

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.document_calls += 1
        return [
            [float(index + 1), float(len(text) % 17 + 1), 0.5, -0.5]
            for index, text in enumerate(texts)
        ]

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 1.0, 0.5, -0.5]


class FakeVectorIndex(VectorIndex):
    def __init__(self) -> None:
        self.deleted_document_ids: list[int] = []
        self.upsert_batches: list[list[VectorIndexRecord]] = []

    def ensure_collection(self) -> None:
        return None

    def upsert(self, records: Sequence[VectorIndexRecord]) -> None:
        self.upsert_batches.append(list(records))

    def delete_by_document_id(self, document_id: int) -> None:
        self.deleted_document_ids.append(document_id)


def _fixture_document_id(db: Session) -> int:
    manifest = AgentEvaluationFixtureService(
        user_repository=UserRepository(),
        knowledge_base_repository=KnowledgeBaseRepository(),
        document_repository=DocumentRepository(),
    ).prepare(db=db)
    return manifest.primary_document_id


def _corpus_service(
    *,
    embedding_provider: EmbeddingProvider,
    vector_index: VectorIndex | None,
) -> AgentEvaluationCorpusService:
    return AgentEvaluationCorpusService(
        document_repository=DocumentRepository(),
        document_content_repository=DocumentContentRepository(),
        document_chunk_repository=DocumentChunkRepository(),
        chunk_embedding_repository=ChunkEmbeddingRepository(),
        embedding_provider=embedding_provider,
        vector_index=vector_index,
    )


def test_eval_corpus_prepare_builds_parent_child_embeddings_and_index(
    db: Session,
) -> None:
    document_id = _fixture_document_id(db)
    embedding_provider = FakeEmbeddingProvider()
    vector_index = FakeVectorIndex()
    service = _corpus_service(
        embedding_provider=embedding_provider,
        vector_index=vector_index,
    )

    result = service.prepare(db=db, document_id=document_id)

    assert result.corpus_version == "1.0.0"
    assert result.document_id == document_id
    assert result.embedding_model == "eval-test-embedding"
    assert result.embedding_generated is True
    assert result.indexed_point_count == 2
    assert result.evidence_source_ref == (
        f"doc:{document_id}:chunk:{result.evidence_chunk_id}"
    )

    content = DocumentContentRepository().find_by_document_id(
        db=db,
        document_id=document_id,
    )
    assert content is not None
    assert content.content == AgentEvaluationCorpusService.PARENT_CONTENT

    chunks = DocumentChunkRepository().find_by_document_content_id(
        db=db,
        document_content_id=content.id,
    )
    assert len(chunks) == 2
    parent = next(chunk for chunk in chunks if chunk.parent_chunk_id is None)
    child = next(chunk for chunk in chunks if chunk.parent_chunk_id is not None)
    assert child.parent_chunk_id == parent.id
    assert child.id == result.evidence_chunk_id
    assert child.content == AgentEvaluationCorpusService.CHILD_CONTENT
    assert all(
        chunk.embedding_status == EmbeddingStatus.COMPLETED.value
        for chunk in chunks
    )

    embeddings = [
        ChunkEmbeddingRepository().find_by_chunk_id(
            db=db,
            document_chunk_id=chunk.id,
        )
        for chunk in chunks
    ]
    assert all(embedding is not None for embedding in embeddings)
    assert all(
        embedding.embedding_model == "eval-test-embedding"
        for embedding in embeddings
        if embedding is not None
    )

    assert vector_index.deleted_document_ids == [document_id]
    assert len(vector_index.upsert_batches) == 1
    assert {record.chunk_role for record in vector_index.upsert_batches[0]} == {
        "parent",
        "child",
    }


def test_eval_corpus_prepare_is_idempotent_and_reindexes_without_reembedding(
    db: Session,
) -> None:
    document_id = _fixture_document_id(db)
    embedding_provider = FakeEmbeddingProvider()
    vector_index = FakeVectorIndex()
    service = _corpus_service(
        embedding_provider=embedding_provider,
        vector_index=vector_index,
    )

    first = service.prepare(db=db, document_id=document_id)
    second = service.prepare(db=db, document_id=document_id)

    assert second.parent_chunk_id == first.parent_chunk_id
    assert second.evidence_chunk_id == first.evidence_chunk_id
    assert second.evidence_source_ref == first.evidence_source_ref
    assert first.embedding_generated is True
    assert second.embedding_generated is False
    assert embedding_provider.document_calls == 1

    # 外部索引每次都重建，Qdrant 被清空后再次 prepare 能自愈。
    assert vector_index.deleted_document_ids == [document_id, document_id]
    assert len(vector_index.upsert_batches) == 2
    assert all(len(batch) == 2 for batch in vector_index.upsert_batches)


def test_eval_corpus_rejects_non_reserved_document(db: Session) -> None:
    fixture_id = _fixture_document_id(db)
    fixture_document = DocumentRepository().find_by_id(
        db=db,
        document_id=fixture_id,
    )
    assert fixture_document is not None

    normal_document = Document(
        knowledge_base_id=fixture_document.knowledge_base_id,
        filename="normal-business-document.txt",
        storage_key="normal/business/document.txt",
        size=10,
        status=DocumentStatus.COMPLETED.value,
    )
    db.add(normal_document)
    db.commit()
    db.refresh(normal_document)

    service = _corpus_service(
        embedding_provider=FakeEmbeddingProvider(),
        vector_index=FakeVectorIndex(),
    )

    import pytest

    with pytest.raises(ValueError, match="reserved fixture document"):
        service.prepare(db=db, document_id=normal_document.id)
