from sqlalchemy.orm import Session

from app.models.database.document_content import DocumentContent


class DocumentContentRepository:
    """
    文档解析内容数据访问层。

    负责文档解析结果的数据库操作。
    不包含业务逻辑。
    """

    def create(
        self,
        db: Session,
        document_content: DocumentContent,
    ) -> DocumentContent:
        """
        保存文档解析内容。

        Args:
            db:
                数据库会话。

            document_content:
                文档解析内容对象。

        Returns:
            保存后的解析内容对象。
        """

        db.add(document_content)

        db.commit()

        db.refresh(document_content)

        return document_content


    def find_by_document_id(
        self,
        db: Session,
        document_id: int,
    ) -> DocumentContent | None:
        """
        根据文档ID查询解析内容。

        Args:
            db:
                数据库会话。

            document_id:
                文档ID。

        Returns:
            文档解析内容。
            不存在时返回None。
        """

        return (
            db.query(DocumentContent)
            .filter(
                DocumentContent.document_id == document_id
            )
            .first()
        )