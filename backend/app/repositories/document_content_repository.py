from sqlalchemy.orm import Session

from app.models.database.document_content import DocumentContent


class DocumentContentRepository:
    """
    文档解析内容数据访问层。

    负责 document_contents 表的数据操作。
    不包含业务逻辑。
    不负责事务提交。
    """

    def create(
        self,
        db: Session,
        document_content: DocumentContent,
    ) -> DocumentContent:
        """
        保存解析内容。

        Args:
            db:
                数据库会话。

            document_content:
                文档解析内容对象。

        Returns:
            保存后的解析内容对象。

        Notes:
            只执行 flush，
            commit 由 Service 层负责。
        """

        db.add(document_content)

        db.flush()

        return document_content


    def find_by_document_id(
        self,
        db: Session,
        document_id: int,
    ) -> DocumentContent | None:
        """
        根据文档ID查询解析结果。

        Args:
            db:
                数据库会话。

            document_id:
                文档ID。

        Returns:
            文档解析内容，不存在返回 None。
        """

        return (
            db.query(DocumentContent)
            .filter(
                DocumentContent.document_id == document_id
            )
            .first()
        )


    def save_or_update(
        self,
        db: Session,
        document_content: DocumentContent,
    ) -> DocumentContent:
        """
        保存当前解析版本或更新解析结果

        用于：
        - 第一次解析
        - 重新解析覆盖旧结果
        """

        existing_content = self.find_by_document_id(
            db=db,
            document_id=document_content.document_id,
        )

        if existing_content is None:
            return self.create(
                db=db,
                document_content=document_content,
            )

        existing_content.content = (
            document_content.content
        )

        existing_content.parser_type = (
            document_content.parser_type
        )

        existing_content.parser_version = (
            document_content.parser_version
        )

        db.flush()

        return existing_content