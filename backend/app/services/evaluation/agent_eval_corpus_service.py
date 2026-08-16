from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.agent.evidence import build_knowledge_source_ref
from app.constants.document_status import DocumentStatus
from app.constants.embedding_status import EmbeddingStatus
from app.models.database.chunk_embedding import ChunkEmbedding
from app.models.database.document_chunk import DocumentChunk
from app.models.database.document_content import DocumentContent
from app.repositories.chunk_embedding_repository import ChunkEmbeddingRepository
from app.repositories.document_chunk_repository import DocumentChunkRepository
from app.repositories.document_content_repository import DocumentContentRepository
from app.repositories.document_repository import DocumentRepository
from app.services.embedding.base import EmbeddingProvider
from app.services.vector_index_service import VectorIndexService
from app.services.vector_store.base import VectorIndex


@dataclass(frozen=True)
class AgentEvaluationCorpusResult:
    """Agent Eval 正向检索语料准备结果，不包含语料正文。"""

    corpus_version: str
    document_id: int
    parent_chunk_id: int
    evidence_chunk_id: int
    evidence_source_ref: str
    embedding_model: str
    embedding_generated: bool
    indexed_point_count: int


class AgentEvaluationCorpusService:
    """
    准备 Agent Live Eval 专用、确定性的最小 RAG 正向语料。

    该服务只允许用于 Eval Fixture：它直接维护专用 Document 的
    DocumentContent / Parent-Child Chunk / ChunkEmbedding，并复用现有
    VectorIndexService 同步外部索引。不会写入生产 AgentRun/SSE。
    """

    CORPUS_VERSION = "1.0.0"
    PARSER_TYPE = "agent_eval_fixture"
    PARSER_VERSION = "1.0.0"
    CHUNK_STRATEGY = "agent_eval_fixture"
    EXPECTED_FILENAME = "__agent_eval_primary_document__.txt"
    EXPECTED_STORAGE_KEY = "agent-eval-fixture/primary/document.txt"

    PARENT_CONTENT = (
        "Agent Eval Qdrant 部署基准。Knowledge Assistant 使用 Qdrant 保存并检索"
        "知识库 Chunk 的向量索引。在 Docker Compose 网络中，Qdrant 服务名为 "
        "qdrant，HTTP API 默认端口为 6333。应用容器可通过 "
        "http://qdrant:6333 访问该服务；宿主机是否映射 6333 取决于 Compose "
        "端口配置。"
    )
    CHILD_CONTENT = (
        "Qdrant 部署：Docker Compose 内部服务名为 qdrant，HTTP API 默认端口为 "
        "6333，应用容器可通过 http://qdrant:6333 访问。"
    )

    def __init__(
        self,
        *,
        document_repository: DocumentRepository,
        document_content_repository: DocumentContentRepository,
        document_chunk_repository: DocumentChunkRepository,
        chunk_embedding_repository: ChunkEmbeddingRepository,
        embedding_provider: EmbeddingProvider,
        vector_index: VectorIndex | None,
    ) -> None:
        self.document_repository = document_repository
        self.document_content_repository = document_content_repository
        self.document_chunk_repository = document_chunk_repository
        self.chunk_embedding_repository = chunk_embedding_repository
        self.embedding_provider = embedding_provider
        self.vector_index = vector_index

    def prepare(
        self,
        *,
        db: Session,
        document_id: int,
    ) -> AgentEvaluationCorpusResult:
        """
        创建或修复 Eval 正向语料，并保证 SQL 与外部向量索引可检索。

        SQL 侧语料/Embedding 保持幂等；当内容和 Embedding Model 未变化时，
        不重复调用 Embedding Provider。外部索引每次 prepare 都按 document_id
        删除后重建，以便 Qdrant Collection 被清空或发生漂移时自动恢复。
        """

        if document_id <= 0:
            raise ValueError("document_id must be greater than 0")

        document = self.document_repository.find_by_id(
            db=db,
            document_id=document_id,
        )
        if document is None:
            raise ValueError("agent evaluation corpus document not found")

        if (
            document.filename != self.EXPECTED_FILENAME
            or document.storage_key != self.EXPECTED_STORAGE_KEY
        ):
            raise ValueError(
                "agent evaluation corpus can only modify the reserved fixture document"
            )

        if DocumentStatus(document.status) != DocumentStatus.COMPLETED:
            raise ValueError(
                "agent evaluation corpus document must be completed"
            )

        try:
            content, chunks = self._ensure_content_and_chunks(
                db=db,
                document_id=document.id,
            )
            embedding_generated = self._ensure_embeddings(
                db=db,
                chunks=chunks,
            )

            # Eval Fixture 的文档元数据保持与实际语料大小一致。
            document.size = len(self.PARENT_CONTENT.encode("utf-8"))
            db.commit()
            db.refresh(content)
            for chunk in chunks:
                db.refresh(chunk)

        except Exception:
            db.rollback()
            raise

        parent_chunk, child_chunk = self._resolve_fixture_chunks(chunks)
        indexed_point_count = 0

        if self.vector_index is not None:
            # 先删后建，确保 prepare 能修复外部索引漂移/残留 Point。
            self.vector_index.delete_by_document_id(document.id)
            indexed_point_count = VectorIndexService(
                document_repository=self.document_repository,
                chunk_embedding_repository=self.chunk_embedding_repository,
                vector_index=self.vector_index,
            ).index_document(
                db=db,
                document_id=document.id,
            )

        return AgentEvaluationCorpusResult(
            corpus_version=self.CORPUS_VERSION,
            document_id=document.id,
            parent_chunk_id=parent_chunk.id,
            evidence_chunk_id=child_chunk.id,
            evidence_source_ref=build_knowledge_source_ref(
                document_id=document.id,
                chunk_id=child_chunk.id,
            ),
            embedding_model=self.embedding_provider.model_name,
            embedding_generated=embedding_generated,
            indexed_point_count=indexed_point_count,
        )

    def _ensure_content_and_chunks(
        self,
        *,
        db: Session,
        document_id: int,
    ) -> tuple[DocumentContent, list[DocumentChunk]]:
        content = self.document_content_repository.find_by_document_id(
            db=db,
            document_id=document_id,
        )

        if content is None:
            content = self.document_content_repository.create(
                db=db,
                document_content=DocumentContent(
                    document_id=document_id,
                    content=self.PARENT_CONTENT,
                    parser_type=self.PARSER_TYPE,
                    parser_version=self.PARSER_VERSION,
                    structure_metadata=None,
                ),
            )
        elif not self._content_matches(content):
            # 旧 Chunk ID 可能仍存在于外部索引；先按 Document 清理。
            if self.vector_index is not None:
                self.vector_index.delete_by_document_id(document_id)

            self.document_chunk_repository.delete_by_document_content_id(
                db=db,
                document_content_id=content.id,
            )
            content = self.document_content_repository.save_or_update(
                db=db,
                document_content=DocumentContent(
                    document_id=document_id,
                    content=self.PARENT_CONTENT,
                    parser_type=self.PARSER_TYPE,
                    parser_version=self.PARSER_VERSION,
                    structure_metadata=None,
                ),
            )

        chunks = self.document_chunk_repository.find_by_document_content_id(
            db=db,
            document_content_id=content.id,
        )
        if self._chunks_match(chunks):
            return content, chunks

        if chunks:
            if self.vector_index is not None:
                self.vector_index.delete_by_document_id(document_id)
            self.document_chunk_repository.delete_by_document_content_id(
                db=db,
                document_content_id=content.id,
            )

        parent = DocumentChunk(
            document_content_id=content.id,
            chunk_index=0,
            content=self.PARENT_CONTENT,
            token_count=None,
            chunk_strategy=self.CHUNK_STRATEGY,
            embedding_status=EmbeddingStatus.PENDING.value,
            chunk_metadata={
                "chunk_role": "parent",
                "corpus_version": self.CORPUS_VERSION,
            },
            parent_chunk_id=None,
        )
        self.document_chunk_repository.save_all(db=db, chunks=[parent])

        child = DocumentChunk(
            document_content_id=content.id,
            chunk_index=1,
            content=self.CHILD_CONTENT,
            token_count=None,
            chunk_strategy=self.CHUNK_STRATEGY,
            embedding_status=EmbeddingStatus.PENDING.value,
            chunk_metadata={
                "chunk_role": "child",
                "corpus_version": self.CORPUS_VERSION,
                "parent_chunk_index": 0,
            },
            parent_chunk_id=parent.id,
        )
        self.document_chunk_repository.save_all(db=db, chunks=[child])
        return content, [parent, child]

    def _ensure_embeddings(
        self,
        *,
        db: Session,
        chunks: list[DocumentChunk],
    ) -> bool:
        model_name = self.embedding_provider.model_name
        existing = {
            chunk.id: self.chunk_embedding_repository.find_by_chunk_id(
                db=db,
                document_chunk_id=chunk.id,
            )
            for chunk in chunks
        }

        reusable = all(
            embedding is not None
            and embedding.embedding_model == model_name
            and bool(embedding.vector)
            and chunk.embedding_status == EmbeddingStatus.COMPLETED.value
            for chunk in chunks
            for embedding in [existing[chunk.id]]
        )
        if reusable:
            return False

        vectors = self.embedding_provider.embed_documents(
            [chunk.content for chunk in chunks]
        )
        if len(vectors) != len(chunks):
            raise RuntimeError(
                "agent evaluation corpus embedding count does not match chunks"
            )
        if not vectors or any(not vector for vector in vectors):
            raise RuntimeError("agent evaluation corpus embedding is empty")

        expected_dimension = len(vectors[0])
        if any(len(vector) != expected_dimension for vector in vectors):
            raise RuntimeError(
                "agent evaluation corpus embedding dimensions are inconsistent"
            )

        for chunk, vector in zip(chunks, vectors, strict=True):
            self.chunk_embedding_repository.save_or_update(
                db=db,
                embedding=ChunkEmbedding(
                    document_chunk_id=chunk.id,
                    vector=list(vector),
                    embedding_model=model_name,
                    embedding_dimension=len(vector),
                    embedding_metadata={
                        "source": "agent_eval_fixture",
                        "corpus_version": self.CORPUS_VERSION,
                    },
                ),
            )
            chunk.embedding_status = EmbeddingStatus.COMPLETED.value

        db.flush()
        return True

    @classmethod
    def _content_matches(cls, content: DocumentContent) -> bool:
        return (
            content.content == cls.PARENT_CONTENT
            and content.parser_type == cls.PARSER_TYPE
            and content.parser_version == cls.PARSER_VERSION
        )

    @classmethod
    def _chunks_match(cls, chunks: list[DocumentChunk]) -> bool:
        if len(chunks) != 2:
            return False

        try:
            parent, child = cls._resolve_fixture_chunks(chunks)
        except ValueError:
            return False

        return (
            parent.chunk_index == 0
            and parent.content == cls.PARENT_CONTENT
            and parent.chunk_strategy == cls.CHUNK_STRATEGY
            and child.chunk_index == 1
            and child.content == cls.CHILD_CONTENT
            and child.chunk_strategy == cls.CHUNK_STRATEGY
            and child.parent_chunk_id == parent.id
        )

    @staticmethod
    def _resolve_fixture_chunks(
        chunks: list[DocumentChunk],
    ) -> tuple[DocumentChunk, DocumentChunk]:
        parents = [chunk for chunk in chunks if chunk.parent_chunk_id is None]
        children = [chunk for chunk in chunks if chunk.parent_chunk_id is not None]
        if len(parents) != 1 or len(children) != 1:
            raise ValueError(
                "agent evaluation corpus requires exactly one parent and one child"
            )
        return parents[0], children[0]
