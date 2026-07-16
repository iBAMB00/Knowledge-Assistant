from sqlalchemy.orm import Session

from app.models.database.document import Document


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

        db.commit()

        db.refresh(document)

        return document