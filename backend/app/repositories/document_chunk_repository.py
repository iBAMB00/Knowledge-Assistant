from sqlalchemy.orm import Session

from app.models.database.document_chunk import DocumentChunk
from app.models.database.document_content import DocumentContent



class DocumentChunkRepository:
    """
    文档切片数据访问层。

    负责 document_chunks 表的数据操作。
    不包含切片业务逻辑。
    不负责事务提交。
    """

    def save_all(
        self,
        db: Session,
        chunks: list[DocumentChunk],
    ) -> list[DocumentChunk]:
        """
        保存所有文档切片。

        Args:
            db:
                数据库会话。

            chunks:
                文档切片数据库对象列表。

        Returns:
            保存后的切片对象列表。

        Notes:
            只执行 flush，
            commit 由 Service 层负责。
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
        根据解析内容ID查询所有切片。

        Args:
            db:
                数据库会话。

            document_content_id:
                文档解析内容ID。

        Returns:
            文档切片列表。
        """

        return (
            db.query(DocumentChunk)
            .filter(
                DocumentChunk.document_content_id
                == document_content_id
            )
            .order_by(
                DocumentChunk.chunk_index
            )
            .all()
        )

    def find_by_document_id(
        self,
        db: Session,
        document_id: int,
    ) -> list[DocumentChunk]:
        """
        根据文档ID查询切片。

        当前返回该文档当前解析内容对应的切片。
        """

        return (
            db.query(DocumentChunk)
            .join(
                DocumentContent,
                DocumentChunk.document_content_id
                == DocumentContent.id,
            )
            .filter(
                DocumentContent.document_id
                == document_id,
            )
            .order_by(
                DocumentChunk.chunk_index
            )
            .all()
        )

    def delete_by_document_content_id(
        self,
        db: Session,
        document_content_id: int,
    ) -> None:
        """
        根据解析内容ID删除所有切片。

        Args:
            db:
                数据库会话。

            document_content_id:
                文档解析内容ID。
        """

        (
            db.query(DocumentChunk)
            .filter(
                DocumentChunk.document_content_id
                == document_content_id
            )
            .delete()
        )

        db.flush()