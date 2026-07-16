from pathlib import Path


class DocumentService:
    """负责文档上传、查询和删除等文档生命周期管理。"""

    def __init__(self, upload_dir: str = "uploads") -> None:
        """
        初始化文档服务，并确保上传目录存在。

        Args:
            upload_dir: 文档上传目录。
        """
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)