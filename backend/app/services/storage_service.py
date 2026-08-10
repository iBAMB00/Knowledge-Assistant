from pathlib import Path
from uuid import uuid4

from app.schemas.storage_result import StorageResult
from app.services.storage.base import StorageProvider
from app.services.storage.local import LocalStorageProvider


class StorageService:
    """
    业务无关的统一文件存储服务。

    StorageService 只暴露稳定 storage_key；具体本地路径或 MinIO Bucket
    由 StorageProvider 负责解析，调用方不依赖物理存储位置。
    """

    def __init__(
        self,
        provider: StorageProvider | str | Path | None = None,
        storage_dir: str | None = None,
    ) -> None:
        """
        初始化存储服务。

        storage_dir 仅为兼容现有测试/本机调用；正式应用通过 Storage Factory
        按配置注入 LocalStorageProvider 或 MinioStorageProvider。
        """
        if isinstance(provider, (str, Path)):
            if storage_dir is not None:
                raise ValueError("provider path and storage_dir cannot be set together")
            storage_dir = str(provider)
            provider = None

        if provider is not None and storage_dir is not None:
            raise ValueError("provider and storage_dir cannot be set together")

        self.provider = provider or LocalStorageProvider(
            storage_dir=storage_dir or "uploads"
        )

    def save(
        self,
        filename: str,
        content: bytes,
        knowledge_base_id: int | None = None,
    ) -> StorageResult:
        """生成稳定 key 并保存文件。"""
        safe_filename = filename.replace("\\", "/").split("/")[-1].strip()

        if not safe_filename:
            raise ValueError("filename cannot be empty")
        if not content:
            raise ValueError("file content cannot be empty")

        prefix = (
            f"knowledge-bases/{knowledge_base_id}"
            if knowledge_base_id is not None
            else "legacy"
        )
        storage_key = f"{prefix}/{uuid4().hex}_{safe_filename}"

        self.provider.save(storage_key, content)

        return StorageResult(
            storage_key=storage_key,
            size=len(content),
        )

    def read(self, storage_key: str) -> bytes:
        """按稳定 storage_key 读取文件。"""
        return self.provider.read(storage_key)

    def delete(self, storage_key: str) -> None:
        """按稳定 storage_key 幂等删除文件。"""
        self.provider.delete(storage_key)

    def exists(self, storage_key: str) -> bool:
        """判断 storage_key 对应文件是否存在。"""
        return self.provider.exists(storage_key)
