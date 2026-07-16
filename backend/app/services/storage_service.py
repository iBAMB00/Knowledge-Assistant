from pathlib import Path
from uuid import uuid4

from app.schema.storage_result import StorageResult


class StorageService:
    """
    文件存储服务。

    负责文件的保存、读取和删除。
    当前使用本地文件系统实现。
    后续可替换为 MinIO、OSS 等对象存储。
    """

    def __init__(self, storage_dir: str = "uploads") -> None:
        """
        初始化文件存储目录。

        Args:
            storage_dir:
                本地文件存储目录。
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    def save(
        self,
        filename: str,
        content: bytes,
    ) -> str:
        """
        保存文件到存储目录。

        Args:
            filename:
                原始文件名。

            content:
                文件二进制内容。

        Returns:
            文件保存后的路径。
        """
        safe_filename = Path(filename).name

        stored_name = (
            f"{uuid4().hex}_{safe_filename}"
        )

        file_path = self.storage_dir / stored_name

        file_path.write_bytes(content)

        return StorageResult(
            stored_name=stored_name,
            path=str(file_path),
        )

    def delete(self, path: str) -> None:
        """
        删除文件。

        Args:
            path:
                文件路径。
        """
        file_path = Path(path)

        if file_path.exists():
            file_path.unlink()