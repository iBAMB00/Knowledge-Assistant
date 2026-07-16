from pathlib import Path
from uuid import uuid4

from app.models.document import DocumentInfo


class DocumentService:
    """
    文档服务。

    负责文档上传、查询和删除等文档生命周期管理。
    """

    def __init__(self, upload_dir: str = "uploads") -> None:
        """
        初始化文档服务，并确保上传目录存在。

        Args:
            upload_dir: 服务端保存上传文档的目录。
        """
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)

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

        original_path = Path(cleaned_filename)
        safe_filename = original_path.name

        # 使用 UUID 生成唯一文件名，防止同名文件相互覆盖。
        stored_name = f"{uuid4().hex}_{safe_filename}"
        stored_path = self.upload_dir / stored_name

        stored_path.write_bytes(content)

        return DocumentInfo(
            filename=safe_filename,
            stored_name=stored_name,
            path=str(stored_path),
            size=len(content),
        )

    def list_documents(self) -> None:
        """查询已上传文档，后续步骤实现。"""
        pass

    def delete_document(self) -> None:
        """删除指定文档，后续步骤实现。"""
        pass