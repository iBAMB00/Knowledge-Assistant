from app.models.document import DocumentInfo
from app.services.storage_service import StorageService


class DocumentService:
    """
    文档业务
    负责文档生命周期管理，
    不负责具体文件存储实现。
    """

    def __init__(self, storage_service: StorageService) -> None:
        """
        初始化文档服务。

        Args:
            storage_service: 文件存储服务。
        """
        self.storage_service = storage_service

    def upload_document(
        self,
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

        stored_result = self.storage_service.save(cleaned_filename, content)

        return DocumentInfo(
            filename=cleaned_filename,
            stored_name=stored_result.stored_name,
            path=stored_result.path,
            size=len(content),
        )

    def list_documents(self) -> None:
        """查询已上传文档，后续步骤实现。"""
        pass

    def delete_document(self) -> None:
        """删除指定文档，后续步骤实现。"""
        pass