from pathlib import Path

from app.services.storage.base import StorageProvider, normalize_storage_key


class LocalStorageProvider(StorageProvider):
    """基于本地目录的文件存储 Provider。"""

    def __init__(self, storage_dir: str = "uploads") -> None:
        self.storage_dir = Path(storage_dir).resolve()
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def save(self, storage_key: str, content: bytes) -> None:
        file_path = self._resolve_key(storage_key)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(content)

    def read(self, storage_key: str) -> bytes:
        file_path = self._resolve_key(storage_key)

        if not file_path.is_file():
            raise FileNotFoundError(
                f"stored object not found: {storage_key}"
            )

        return file_path.read_bytes()

    def delete(self, storage_key: str) -> None:
        file_path = self._resolve_key(storage_key)

        if file_path.is_file():
            file_path.unlink()
            self._remove_empty_parent_directories(file_path.parent)

    def exists(self, storage_key: str) -> bool:
        return self._resolve_key(storage_key).is_file()

    def resolve_path(self, storage_key: str) -> Path:
        """返回本地 Provider 对应路径；主要用于调试和测试。"""
        return self._resolve_key(storage_key)

    def _resolve_key(self, storage_key: str) -> Path:
        normalized_key = normalize_storage_key(storage_key)
        file_path = (self.storage_dir / normalized_key).resolve()

        if not file_path.is_relative_to(self.storage_dir):
            raise ValueError("storage key is outside storage directory")

        return file_path

    def _remove_empty_parent_directories(self, directory: Path) -> None:
        current = directory

        while current != self.storage_dir:
            try:
                current.rmdir()
            except OSError:
                break

            current = current.parent
