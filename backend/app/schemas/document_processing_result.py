from dataclasses import dataclass
from datetime import datetime

from app.constants.document_status import DocumentStatus


@dataclass(frozen=True)
class DocumentProcessingResult:
    """
    文档处理链内部结果。

    与公共 DocumentResponse 分离，允许历史 legacy 文档继续保留
    knowledge_base_id=None，用于离线评估、恢复和兼容处理。
    """

    id: int
    knowledge_base_id: int | None
    filename: str
    size: int
    status: DocumentStatus
    created_at: datetime
