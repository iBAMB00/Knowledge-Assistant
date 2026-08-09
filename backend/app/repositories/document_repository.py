from sqlalchemy.orm import Session

from app.models.database.document import Document


class DocumentRepository:
    """
    文档数据访问层。

    负责：
    - 新增文档
    - 查询文档
    - 删除文档

    不负责：
    - 文档状态流转
    - 业务流程
    - 事务提交
    """

    def create(
        self,
        db: Session,
        document: Document,
    ) -> Document:
        """
        保存文档记录。

        Args:
            db:
                数据库会话。

            document:
                文档数据库对象。

        Returns:
            保存后的文档对象。
        """

        db.add(document)
        db.flush()

        return document

    def find_all(
        self,
        db: Session,
        knowledge_base_id: int | None = None,
    ) -> list[Document]:
        """
        查询所有文档。

        Args:
            db:
                数据库会话。

        Returns:
            文档数据库对象列表。
        """

        query = db.query(Document)

        if knowledge_base_id is not None:
            query = query.filter(
                Document.knowledge_base_id == knowledge_base_id
            )

        return (
            query
            .order_by(Document.created_at.desc())
            .all()
        )

    def find_by_id(
        self,
        db: Session,
        document_id: int,
    ) -> Document | None:
        """
        根据文档ID查询文档。

        Args:
            db:
                数据库会话。

            document_id:
                文档ID。

        Returns:
            文档数据库对象；不存在时返回None。
        """

        return db.get(
            Document,
            document_id,
        )
    

    def count_by_knowledge_base_id(
        self,
        db: Session,
        knowledge_base_id: int,
    ) -> int:
        """统计知识库中的文档数量。"""
        return (
            db.query(Document)
            .filter(Document.knowledge_base_id == knowledge_base_id)
            .count()
        )

    def delete(
        self,
        db: Session,
        document: Document,
    ) -> None:
        """
        删除文档记录。

        Args:
            db:
                数据库会话。

            document:
                待删除的文档数据库对象。
        """
        db.delete(document)
        db.flush()