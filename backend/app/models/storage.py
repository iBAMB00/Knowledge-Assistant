from dataclasses import dataclass


@dataclass(frozen=True)
class StorageResult:
    """
    文件存储结果。

    描述文件保存后的存储信息。
    """

    stored_name: str
    path: str