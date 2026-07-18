from sqlalchemy.orm import Session

from app.schemas.document_info import DocumentInfo
from app.services.storage_service import StorageService
from app.repositories.document_repository import DocumentRepository
from app.models.database.document import Document
from app.schemas.document_response import DocumentResponse


class DocumentService:
    """
    文档业务
    负责文档生命周期管理，
    不负责具体文件存储实现。
    """

    def __init__(
            self, 
            storage_service: StorageService,
            document_repository: DocumentRepository
        ) -> None:
        """
        初始化文档服务。

        Args:
            storage_service: 文件存储服务。
            document_repository: 文档数据库操作仓库。
        """
        self.storage_service = storage_service
        self.document_repository = document_repository

    def upload_document(
        self,
        db: Session,
        filename: str,
        content: bytes,
    ) -> DocumentInfo:
        """
        保存上传的文档，并返回文档基础信息。

        Args:
            filename: 用户上传时的原始文件名。
            content: 文档的二进制内容。

        Returns:
            保存后的文档基础信息。

        Raises:
            ValueError: 文件名为空或文件内容为空时抛出。
        """
        cleaned_filename = filename.strip()

        if not cleaned_filename:
            raise ValueError("filename cannot be empty")

        if not content:
            raise ValueError("file content cannot be empty")

        # 1. 保存文件到存储服务
        stored_result = self.storage_service.save(cleaned_filename, content)
        # 2. 创建数据库对象
        document = Document(
            filename=cleaned_filename,
            stored_name=stored_result.stored_name,
            path=stored_result.path,
            size=len(content),
            status="uploaded",
        )
        # 3. 保存数据库对象
        saved_document = self.document_repository.create(
            db=db, 
            document=document,
        )

        # 4. 返回文档基础信息
        return DocumentInfo(
            filename=saved_document.filename,
            stored_name=saved_document.stored_name,
            path=saved_document.path,
            size=saved_document.size,
        )

    def list_documents(
        self,
        db: Session,
    ) -> list[DocumentResponse]:
        """
        查询文档列表。
        """

        documents = self.document_repository.find_all(
            db=db,
        )

        return [
            DocumentResponse(
                id=document.id,
                filename=document.filename,
                stored_name=document.stored_name,
                size=document.size,
                status=document.status,
                created_at=document.created_at,
            )
            for document in documents
        ]

    def delete_document(self) -> None:
        """删除指定文档，后续步骤实现。"""
        pass