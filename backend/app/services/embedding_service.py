from sqlalchemy.orm import Session

from app.constants.document_status import DocumentStatus
from app.constants.embedding_status import EmbeddingStatus
from app.models.database.chunk_embedding import ChunkEmbedding
from app.repositories.chunk_embedding_repository import (
    ChunkEmbeddingRepository,
)
from app.repositories.document_chunk_repository import (
    DocumentChunkRepository,
)
from app.repositories.document_repository import DocumentRepository
from app.services.embedding.base import EmbeddingProvider
from app.services.status_machine import StatusMachine


class EmbeddingService:
    """
    文档切片向量化编排服务。

    负责：
    - 按文档批量查询Chunk
    - 调用EmbeddingProvider
    - 保存ChunkEmbedding
    - 管理Chunk向量状态
    - 汇总Document整体状态
    - 管理事务边界
    """

    def __init__(
        self,
        document_repository: DocumentRepository,
        document_chunk_repository: DocumentChunkRepository,
        chunk_embedding_repository: ChunkEmbeddingRepository,
        embedding_provider: EmbeddingProvider,
    ) -> None:
        """初始化Embedding服务。"""

        self.document_repository = document_repository
        self.document_chunk_repository = (
            document_chunk_repository
        )
        self.chunk_embedding_repository = (
            chunk_embedding_repository
        )
        self.embedding_provider = embedding_provider

    def process_document(
        self,
        db: Session,
        document_id: int,
        batch_size: int = 100,
    ) -> int:
        """
        向量化指定文档的全部可处理Chunk。

        Returns:
            本次成功处理的Chunk数量。
        """

        if batch_size <= 0:
            raise ValueError(
                "batch_size must be greater than 0"
            )

        document = self.document_repository.find_by_id(
            db=db,
            document_id=document_id,
        )

        if document is None:
            raise ValueError("document not found")

        current_status = DocumentStatus(document.status)

        if current_status == DocumentStatus.COMPLETED:
            return 0

        if current_status not in {
            DocumentStatus.CHUNKED,
            DocumentStatus.EMBEDDING,
            DocumentStatus.EMBEDDING_FAILED,
        }:
            raise ValueError(
                "invalid document status for embedding: "
                f"{current_status.value}"
            )

        if current_status != DocumentStatus.EMBEDDING:
            StatusMachine.transition_document(
                document=document,
                target_status=DocumentStatus.EMBEDDING,
            )

            db.commit()
            db.refresh(document)

        success_count = 0
        active_chunk_ids: list[int] = []

        try:
            while True:
                chunks = (
                    self.document_chunk_repository
                    .find_retriable_by_document_id(
                        db=db,
                        document_id=document_id,
                        limit=batch_size,
                    )
                )

                if not chunks:
                    break

                active_chunk_ids = [
                    chunk.id
                    for chunk in chunks
                ]

                for chunk in chunks:
                    StatusMachine.transition_embedding(
                        chunk=chunk,
                        target_status=(
                            EmbeddingStatus.PROCESSING
                        ),
                    )

                # 模型调用前持久化processing状态。
                db.commit()

                texts = [
                    chunk.content
                    for chunk in chunks
                ]

                vectors = (
                    self.embedding_provider
                    .embed_documents(texts)
                )

                self._validate_vectors(
                    vectors=vectors,
                    expected_count=len(chunks),
                )

                for chunk, vector in zip(
                    chunks,
                    vectors,
                    strict=True,
                ):
                    embedding = ChunkEmbedding(
                        document_chunk_id=chunk.id,
                        vector=vector,
                        embedding_model=(
                            self.embedding_provider.model_name
                        ),
                        embedding_dimension=len(vector),
                        embedding_metadata=None,
                    )

                    self.chunk_embedding_repository\
                        .save_or_update(
                            db=db,
                            embedding=embedding,
                        )

                    StatusMachine.transition_embedding(
                        chunk=chunk,
                        target_status=(
                            EmbeddingStatus.COMPLETED
                        ),
                    )

                db.commit()

                success_count += len(chunks)
                active_chunk_ids = []

        except Exception:
            self._safe_mark_embedding_failed(
                db=db,
                document_id=document_id,
                chunk_ids=active_chunk_ids,
            )
            raise

        self._finalize_document_status(
            db=db,
            document_id=document_id,
        )

        return success_count

    def _finalize_document_status(
        self,
        db: Session,
        document_id: int,
    ) -> None:
        """
        根据全部Chunk状态汇总Document状态。
        """

        document = self.document_repository.find_by_id(
            db=db,
            document_id=document_id,
        )

        if document is None:
            raise ValueError("document not found")

        status_counts = (
            self.document_chunk_repository
            .count_embedding_statuses_by_document_id(
                db=db,
                document_id=document_id,
            )
        )

        total_count = sum(status_counts.values())

        if total_count == 0:
            StatusMachine.transition_document(
                document=document,
                target_status=(
                    DocumentStatus.EMBEDDING_FAILED
                ),
            )
            db.commit()

            raise ValueError(
                "document has no chunks"
            )

        failed_count = status_counts.get(
            EmbeddingStatus.FAILED,
            0,
        )

        pending_count = status_counts.get(
            EmbeddingStatus.PENDING,
            0,
        )

        processing_count = status_counts.get(
            EmbeddingStatus.PROCESSING,
            0,
        )

        completed_count = status_counts.get(
            EmbeddingStatus.COMPLETED,
            0,
        )

        if failed_count > 0:
            StatusMachine.transition_document(
                document=document,
                target_status=(
                    DocumentStatus.EMBEDDING_FAILED
                ),
            )

        elif pending_count > 0 or processing_count > 0:
            # 当前仍有Chunk未完成，保持embedding状态。
            db.commit()
            return

        elif completed_count == total_count:
            StatusMachine.transition_document(
                document=document,
                target_status=DocumentStatus.COMPLETED,
            )

        else:
            StatusMachine.transition_document(
                document=document,
                target_status=(
                    DocumentStatus.EMBEDDING_FAILED
                ),
            )

        db.commit()

    def _safe_mark_embedding_failed(
        self,
        db: Session,
        document_id: int,
        chunk_ids: list[int],
    ) -> None:
        """
        回滚事务并安全记录Embedding失败状态。
        """

        db.rollback()
        db.expire_all()

        try:
            chunks = (
                self.document_chunk_repository
                .find_by_ids(
                    db=db,
                    chunk_ids=chunk_ids,
                )
            )

            for chunk in chunks:
                if (
                    EmbeddingStatus(
                        chunk.embedding_status
                    )
                    == EmbeddingStatus.PROCESSING
                ):
                    StatusMachine.transition_embedding(
                        chunk=chunk,
                        target_status=EmbeddingStatus.FAILED,
                    )

            document = self.document_repository.find_by_id(
                db=db,
                document_id=document_id,
            )

            if (
                document is not None
                and DocumentStatus(document.status)
                == DocumentStatus.EMBEDDING
            ):
                StatusMachine.transition_document(
                    document=document,
                    target_status=(
                        DocumentStatus.EMBEDDING_FAILED
                    ),
                )

            db.commit()

        except Exception:
            db.rollback()

    def _validate_vectors(
        self,
        vectors: list[list[float]],
        expected_count: int,
    ) -> None:
        """
        验证EmbeddingProvider返回结果。
        """

        if len(vectors) != expected_count:
            raise ValueError(
                "embedding vector count does not match "
                "input text count"
            )

        if not vectors:
            raise ValueError(
                "embedding vectors are empty"
            )

        expected_dimension = len(vectors[0])

        if expected_dimension == 0:
            raise ValueError(
                "embedding vector is empty"
            )

        for vector in vectors:
            if len(vector) != expected_dimension:
                raise ValueError(
                    "embedding vector dimensions "
                    "are inconsistent"
                )