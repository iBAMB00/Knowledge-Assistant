from sqlalchemy.orm import Session

from app.models.database.chunk_embedding import ChunkEmbedding


class ChunkEmbeddingRepository:
    """
    Chunk向量数据访问层。

    负责：
    - 保存Chunk向量
    - 查询Chunk向量
    - 更新已有Chunk向量

    不负责：
    - 向量生成
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
            document_chunk_id=embedding.document_chunk_id,
        )

        if existing is None:
            return self.create(
                db=db,
                embedding=embedding,
            )

        existing.vector = embedding.vector
        existing.embedding_model = embedding.embedding_model
        existing.embedding_dimension = (
            embedding.embedding_dimension
        )
        existing.embedding_metadata = (
            embedding.embedding_metadata
        )

        db.flush()

        return existing