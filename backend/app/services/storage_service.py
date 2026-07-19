from pathlib import Path
from uuid import uuid4

from app.schemas.storage_result import StorageResult


class StorageService:
    """
    本地文件存储服务。

    负责文件的保存、读取和删除。
    当前使用本地文件系统实现，业务层不直接操作文件系统，
    后续可以替换为 MinIO、OSS 等对象存储实现。
    """

    def __init__(
        self,
        storage_dir: str = "uploads",
    ) -> None:
        """
        初始化文件存储目录。

        Args:
            storage_dir:
                本地文件存储根目录。
        """

        # 使用绝对规范路径，便于后续校验文件是否位于存储目录中。
        self.storage_dir = Path(storage_dir).resolve()

        self.storage_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    def save(
        self,
        filename: str,
        content: bytes,
    ) -> StorageResult:
        """
        保存文件到本地存储目录。

        Args:
            filename:
                用户上传时的原始文件名。

            content:
                文件二进制内容。

        Returns:
            文件保存结果，包含存储文件名和文件路径。
        """

        # 只保留文件名，防止原始文件名携带目录穿越路径。
        safe_filename = Path(filename).name

        if not safe_filename:
            raise ValueError("filename cannot be empty")

        stored_name = f"{uuid4().hex}_{safe_filename}"

        file_path = self.storage_dir / stored_name

        file_path.write_bytes(content)

        return StorageResult(
            stored_name=stored_name,
            path=str(file_path),
        )

    def read(
        self,
        path: str,
    ) -> bytes:
        """
        读取已保存文件的二进制内容。

        Args:
            path:
                文件保存路径。

        Returns:
            文件二进制内容。

        Raises:
            ValueError:
                文件路径不在存储目录内。

            FileNotFoundError:
                文件不存在。
        """

        file_path = self._resolve_path(path)

        if not file_path.is_file():
            raise FileNotFoundError(
                f"stored file not found: {path}"
            )

        return file_path.read_bytes()

    def delete(
        self,
        path: str,
    ) -> None:
        """
        删除已保存的文件。

        文件不存在时保持幂等，不抛出异常。

        Args:
            path:
                文件保存路径。

        Raises:
            ValueError:
                文件路径不在存储目录内。
        """

        file_path = self._resolve_path(path)

        if file_path.is_file():
            file_path.unlink()

    def _resolve_path(
        self,
        path: str,
    ) -> Path:
        """
        解析并校验文件路径。

        确保文件只能位于当前存储目录内，防止目录穿越和越权访问。

        Args:
            path:
                待校验的文件路径。

        Returns:
            规范化后的绝对路径。

        Raises:
            ValueError:
                路径超出存储目录。
        """

        file_path = Path(path).resolve()

        if not file_path.is_relative_to(
            self.storage_dir
        ):
            raise ValueError(
                "file path is outside storage directory"
            )

        return file_path