from dataclasses import dataclass


@dataclass(frozen=True)
class StorageResult:
    """文件存储结果；不暴露具体磁盘路径或对象存储实现。"""

    storage_key: str
    size: int
