from sqlalchemy import func
from sqlalchemy.orm import Session

from app.constants.embedding_status import EmbeddingStatus
from app.models.database.document_chunk import DocumentChunk
from app.models.database.document_content import DocumentContent


class DocumentChunkRepository:
    """
    文档切片数据访问层。

    负责：
    - 保存文档切片
    - 查询文档切片
    - 删除文档切片
    - 统计向量化状态

    不负责：
    - 切片算法
    - Chunk向量化状态流转
    - 事务提交
    """

    def save_all(
        self,
        db: Session,
        chunks: list[DocumentChunk],
    ) -> list[DocumentChunk]:
        """
        批量保存文档切片。

        Args:
            db:
                数据库会话。

            chunks:
                文档切片数据库对象列表。

        Returns:
            保存后的文档切片列表。
        """

        db.add_all(chunks)
        db.flush()

        return chunks

    def find_by_document_content_id(
        self,
        db: Session,
        document_content_id: int,
    ) -> list[DocumentChunk]:
        """
        根据解析内容ID查询切片。

        Args:
            db:
                数据库会话。

            document_content_id:
                文档解析内容ID。

        Returns:
            按切片索引升序排列的切片列表。
        """

        return (
            db.query(DocumentChunk)
            .filter(
                DocumentChunk.document_content_id
                == document_content_id
            )
            .order_by(DocumentChunk.chunk_index.asc())
            .all()
        )

    def find_by_document_id(
        self,
        db: Session,
        document_id: int,
    ) -> list[DocumentChunk]:
        """
        根据文档ID查询该文档关联的所有切片。

        当前版本未区分DocumentContent历史版本。
        """

        return (
            db.query(DocumentChunk)
            .join(
                DocumentContent,
                DocumentChunk.document_content_id
                == DocumentContent.id,
            )
            .filter(
                DocumentContent.document_id == document_id,
            )
            .order_by(
                DocumentContent.created_at.asc(),
                DocumentChunk.chunk_index.asc(),
            )
            .all()
        )

    def find_pending_by_document_id(
        self,
        db: Session,
        document_id: int,
        limit: int = 100,
    ) -> list[DocumentChunk]:
        """
        查询指定文档待向量化的Chunk。

        Args:
            db:
                数据库会话。

            document_id:
                文档ID。

            limit:
                单次查询的最大数量。

        Returns:
            待向量化Chunk列表。
        """

        return (
            db.query(DocumentChunk)
            .join(
                DocumentContent,
                DocumentChunk.document_content_id
                == DocumentContent.id,
            )
            .filter(
                DocumentContent.document_id == document_id,
                DocumentChunk.embedding_status
                == EmbeddingStatus.PENDING.value,
            )
            .order_by(DocumentChunk.id.asc())
            .limit(limit)
            .all()
        )

    def count_embedding_statuses_by_document_id(
        self,
        db: Session,
        document_id: int,
    ) -> dict[EmbeddingStatus, int]:
        """
        汇总指定文档下Chunk的向量化状态数量。

        Args:
            db:
                数据库会话。

            document_id:
                文档ID。

        Returns:
            各向量化状态对应的Chunk数量。
        """

        rows = (
            db.query(
                DocumentChunk.embedding_status,
                func.count(DocumentChunk.id),
            )
            .join(
                DocumentContent,
                DocumentChunk.document_content_id
                == DocumentContent.id,
            )
            .filter(
                DocumentContent.document_id == document_id,
            )
            .group_by(DocumentChunk.embedding_status)
            .all()
        )

        return {
            EmbeddingStatus(status): count
            for status, count in rows
        }

    def delete_by_document_content_id(
        self,
        db: Session,
        document_content_id: int,
    ) -> None:
        """
        根据解析内容ID删除全部切片。

        该批量删除依赖数据库外键级联清理ChunkEmbedding。
        """

        (
            db.query(DocumentChunk)
            .filter(
                DocumentChunk.document_content_id
                == document_content_id
            )
            .delete(synchronize_session=False)
        )

        db.flush()