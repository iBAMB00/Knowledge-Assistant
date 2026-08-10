from sqlalchemy import func
from sqlalchemy.orm import Session

from app.constants.embedding_status import EmbeddingStatus
from app.models.database.document_chunk import DocumentChunk
from app.models.database.document_content import DocumentContent
from app.models.database.document import Document


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

    def find_by_document_content_ids(
        self,
        db: Session,
        document_content_ids: list[int],
    ) -> list[DocumentChunk]:
        """批量查询多个解析内容版本对应的全部Chunk。"""
        normalized_ids = sorted(set(document_content_ids))
        if not normalized_ids:
            return []

        return (
            db.query(DocumentChunk)
            .filter(DocumentChunk.document_content_id.in_(normalized_ids))
            .order_by(DocumentChunk.document_content_id.asc(), DocumentChunk.chunk_index.asc())
            .all()
        )


    def find_by_ids(
        self,
        db: Session,
        chunk_ids: list[int],
        knowledge_base_id: int | None = None,
    ) -> list[DocumentChunk]:
        """根据 Chunk ID 查询切片，可选按知识库继续限制边界。"""

        if not chunk_ids:
            return []

        query = db.query(DocumentChunk)

        if knowledge_base_id is not None:
            query = (
                query
                .join(
                    DocumentContent,
                    DocumentChunk.document_content_id
                    == DocumentContent.id,
                )
                .join(
                    Document,
                    DocumentContent.document_id == Document.id,
                )
                .filter(
                    Document.knowledge_base_id == knowledge_base_id
                )
            )

        return (
            query
            .filter(DocumentChunk.id.in_(chunk_ids))
            .order_by(DocumentChunk.id.asc())
            .all()
        )

    def find_document_ids_by_chunk_ids(
        self,
        db: Session,
        chunk_ids: list[int],
    ) -> dict[int, int]:
        """
        批量查询Chunk与所属文档的映射。
        """

        normalized_chunk_ids = sorted(
            set(chunk_ids)
        )

        if not normalized_chunk_ids:
            return {}

        rows = (
            db.query(
                DocumentChunk.id,
                DocumentContent.document_id,
            )
            .join(
                DocumentContent,
                DocumentChunk.document_content_id
                == DocumentContent.id,
            )
            .filter(
                DocumentChunk.id.in_(
                    normalized_chunk_ids
                )
            )
            .all()
        )

        return {
            chunk_id: document_id
            for chunk_id, document_id in rows
        }



    def find_retrieval_candidates(
        self,
        db: Session,
        document_id: int | None = None,
        knowledge_base_id: int | None = None,
        chunk_role: str | None = None,
    ) -> list[tuple[DocumentChunk, DocumentContent, Document]]:
        """查询 BM25 等文本检索使用的已向量化 Chunk。"""

        query = (
            db.query(
                DocumentChunk,
                DocumentContent,
                Document,
            )
            .join(
                DocumentContent,
                DocumentChunk.document_content_id == DocumentContent.id,
            )
            .join(
                Document,
                DocumentContent.document_id == Document.id,
            )
            .filter(
                DocumentChunk.embedding_status
                == EmbeddingStatus.COMPLETED.value
            )
        )

        if document_id is not None:
            query = query.filter(
                DocumentContent.document_id == document_id
            )

        if knowledge_base_id is not None:
            query = query.filter(
                Document.knowledge_base_id == knowledge_base_id
            )

        if chunk_role == "parent":
            query = query.filter(
                DocumentChunk.parent_chunk_id.is_(None)
            )
        elif chunk_role == "child":
            query = query.filter(
                DocumentChunk.parent_chunk_id.isnot(None)
            )
        elif chunk_role is not None:
            raise ValueError(
                "chunk_role must be 'parent', 'child' or None"
            )

        return query.order_by(DocumentChunk.id.asc()).all()

    def find_processable_by_document_id(
        self,
        db: Session,
        document_id: int,
        limit: int = 100,
    ) -> list[DocumentChunk]:
        """
            查询指定文档可向量化的Chunk。

            包括：
            - 尚未处理的pending Chunk
            - 上次处理失败、允许重试的failed Chunk

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
                DocumentChunk.embedding_status.in_(
                    [
                        EmbeddingStatus.PENDING.value,
                        EmbeddingStatus.FAILED.value,
                    ]
                ),
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
