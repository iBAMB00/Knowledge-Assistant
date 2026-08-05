from sqlalchemy.orm import Session

from app.constants.embedding_status import EmbeddingStatus
from app.models.database.chunk_embedding import ChunkEmbedding
from app.models.database.document import Document
from app.models.database.document_chunk import DocumentChunk
from app.models.database.document_content import DocumentContent


class ChunkEmbeddingRepository:
    """
    Chunk向量数据访问层。

    负责：
    - 保存Chunk向量
    - 查询Chunk向量
    - 更新已有Chunk向量
    - 查询向量检索候选数据

    不负责：
    - 向量生成
    - 相似度计算
    - Chunk状态流转
    - 业务流程
    - 事务提交
    """

    def create(
        self,
        db: Session,
        embedding: ChunkEmbedding,
    ) -> ChunkEmbedding:
        """
        保存Chunk向量。
        """

        db.add(embedding)
        db.flush()

        return embedding

    def find_by_chunk_id(
        self,
        db: Session,
        document_chunk_id: int,
    ) -> ChunkEmbedding | None:
        """
        根据Chunk ID查询向量记录。
        """

        return (
            db.query(ChunkEmbedding)
            .filter(
                ChunkEmbedding.document_chunk_id
                == document_chunk_id
            )
            .first()
        )

    def save_or_update(
        self,
        db: Session,
        embedding: ChunkEmbedding,
    ) -> ChunkEmbedding:
        """
        保存或更新Chunk向量。

        一个Chunk当前只保留一条有效向量记录。
        """

        existing = self.find_by_chunk_id(
            db=db,
            document_chunk_id=(
                embedding.document_chunk_id
            ),
        )

        if existing is None:
            return self.create(
                db=db,
                embedding=embedding,
            )

        existing.vector = embedding.vector
        existing.embedding_model = (
            embedding.embedding_model
        )
        existing.embedding_dimension = (
            embedding.embedding_dimension
        )
        existing.embedding_metadata = (
            embedding.embedding_metadata
        )

        db.flush()

        return existing

    def find_search_candidates(
        self,
        db: Session,
        embedding_model: str,
        document_id: int | None = None,
    ) -> list[
        tuple[
            ChunkEmbedding,
            DocumentChunk,
            DocumentContent,
            Document,
        ]
    ]:
        """
        查询向量检索候选数据。

        只返回：
        - 已完成向量化的Chunk
        - 向量不为空
        - 与查询模型一致的向量

        Args:
            db:
                数据库会话。
            embedding_model:
                查询向量使用的模型名称。
            document_id:
                可选文档过滤条件。
        """

        query = (
            db.query(
                ChunkEmbedding,
                DocumentChunk,
                DocumentContent,
                Document,
            )
            .join(
                DocumentChunk,
                (
                    ChunkEmbedding.document_chunk_id
                    == DocumentChunk.id
                ),
            )
            .join(
                DocumentContent,
                (
                    DocumentChunk.document_content_id
                    == DocumentContent.id
                ),
            )
            .join(
                Document,
                (
                    DocumentContent.document_id
                    == Document.id
                ),
            )
            .filter(
                DocumentChunk.embedding_status
                == EmbeddingStatus.COMPLETED.value
            )
            .filter(
                ChunkEmbedding.vector.isnot(None)
            )
            .filter(
                ChunkEmbedding.embedding_model
                == embedding_model
            )
        )

        if document_id is not None:
            query = query.filter(
                DocumentContent.document_id
                == document_id
            )

        return query.all()

    def find_by_document_id_for_index(
        self,
        db: Session,
        document_id: int,
    ) -> list[tuple[ChunkEmbedding, DocumentChunk]]:
        """
        查询指定文档可写入向量索引的数据。

        只返回已经完成向量化且向量不为空的Chunk。
        """

        return (
            db.query(ChunkEmbedding, DocumentChunk)
            .join(
                DocumentChunk,
                ChunkEmbedding.document_chunk_id == DocumentChunk.id,
            )
            .join(
                DocumentContent,
                DocumentChunk.document_content_id == DocumentContent.id,
            )
            .filter(
                DocumentContent.document_id == document_id,
                DocumentChunk.embedding_status == EmbeddingStatus.COMPLETED.value,
                ChunkEmbedding.vector.isnot(None),
            )
            .order_by(DocumentChunk.id.asc())
            .all()
        )