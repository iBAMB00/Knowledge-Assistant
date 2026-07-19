from sqlalchemy.orm import Session

from app.models.database.document import Document
from app.constants.document_status import DocumentStatus


class DocumentRepository:
    """
    文档数据访问层。

    负责文档数据的数据库操作。
    不包含业务逻辑。
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

        # 将变更发送给数据库，使自增ID等字段可用，
        # 但当前事务仍然可以回滚。
        db.flush()

        return document
    
    def find_all(
        self,
        db: Session,
    ) -> list[Document]:
        """
        查询所有文档。

        Args:
            db:
                数据库会话。

        Returns:
            文档数据库对象列表。
        """

        # 未来考虑分页查询
        return (
            db.query(Document)
            .order_by(Document.created_at.desc()) # 按创建时间降序排序
            .all()
        )
    
    def find_by_id(
        self,
        db: Session,
        document_id: int,
    ) -> Document | None:
        """
        根据 ID 查询文档记录。
    
        Args:
            db:
                数据库会话。

            document_id:
                文档 ID。
    
        Returns:
            文档数据库对象或 None。
        """
        return db.get(
            Document,
            document_id,
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
                文档数据库对象。
        """
        db.delete(document)
        db.flush()



    def update_status(
        self,
        db: Session,
        document: Document,
        status: str,
    ) -> Document:
        """
        更新文档状态。

        Args:
            db:
                数据库会话。

            document:
                文档数据库对象。

            status:
                新的文档状态。

        Returns:
            更新后的文档对象。
        """

        document.status = status

        db.flush()

        return document