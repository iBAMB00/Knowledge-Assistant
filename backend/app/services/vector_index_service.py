from sqlalchemy.orm import Session

from app.constants.document_status import DocumentStatus
from app.repositories.chunk_embedding_repository import ChunkEmbeddingRepository
from app.repositories.document_repository import DocumentRepository
from app.services.vector_store.base import VectorIndex, VectorIndexRecord


class VectorIndexService:
    """
    文档向量索引同步服务。

    负责从SQL读取已经完成的Chunk向量，
    并写入外部VectorIndex。

    不负责：
    - 调用Embedding模型
    - 修改Chunk向量状态
    - 提交数据库事务
    """

    def __init__(
        self,
        document_repository: DocumentRepository,
        chunk_embedding_repository: ChunkEmbeddingRepository,
        vector_index: VectorIndex,
    ) -> None:
        """初始化向量索引同步服务。"""

        self.document_repository = document_repository
        self.chunk_embedding_repository = chunk_embedding_repository
        self.vector_index = vector_index

    def index_document(self, db: Session, document_id: int) -> int:
        """
        将指定文档的全部已完成向量同步到外部索引。

        Returns:
            本次写入的Point数量。
        """

        if document_id <= 0:
            raise ValueError("document_id must be greater than zero")

        document = self.document_repository.find_by_id(db=db, document_id=document_id)

        if document is None:
            raise ValueError("document not found")

        if DocumentStatus(document.status) != DocumentStatus.COMPLETED:
            raise ValueError("document is not ready for vector indexing")

        rows = self.chunk_embedding_repository.find_by_document_id_for_index(
            db=db,
            document_id=document_id,
        )

        if not rows:
            raise ValueError("document has no completed embeddings")

        records: list[VectorIndexRecord] = []

        for embedding, chunk in rows:
            if embedding.vector is None:
                raise RuntimeError(f"chunk embedding vector is missing: chunk_id={chunk.id}")

            records.append(
                VectorIndexRecord(
                    chunk_id=chunk.id,
                    document_id=document.id,
                    chunk_index=chunk.chunk_index,
                    filename=document.filename,
                    content=chunk.content,
                    embedding_model=embedding.embedding_model,
                    vector=embedding.vector,
                )
            )

        self.vector_index.upsert(records)

        return len(records)