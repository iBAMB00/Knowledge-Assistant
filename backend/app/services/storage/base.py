from abc import ABC, abstractmethod
from pathlib import PurePosixPath


class StorageProvider(ABC):
    """底层文件存储 Provider 接口。"""

    @abstractmethod
    def save(self, storage_key: str, content: bytes) -> None:
        """按稳定 storage_key 保存对象。"""

    @abstractmethod
    def read(self, storage_key: str) -> bytes:
        """读取对象内容。"""

    @abstractmethod
    def delete(self, storage_key: str) -> None:
        """幂等删除对象。"""

    @abstractmethod
    def exists(self, storage_key: str) -> bool:
        """判断对象是否存在。"""


def normalize_storage_key(storage_key: str) -> str:
    """校验并规范化对象 key，拒绝绝对路径和目录穿越。"""
    normalized = storage_key.strip().replace("\\", "/")

    if not normalized:
        raise ValueError("storage key cannot be empty")

    key_path = PurePosixPath(normalized)

    if key_path.is_absolute() or ".." in key_path.parts:
        raise ValueError("invalid storage key")

    parts = [part for part in key_path.parts if part not in {"", "."}]

    if not parts:
        raise ValueError("storage key cannot be empty")

    return "/".join(parts)
