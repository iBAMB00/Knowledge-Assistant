from datetime import datetime

from pydantic import BaseModel

from app.constants.document_status import DocumentStatus


class DocumentResponse(BaseModel):
    """
    对外统一文档响应模型。

    v1.0 公共 API 中可见文档必须归属于 KnowledgeBase，
    且状态使用 DocumentStatus 枚举统一约束。
    """

    id: int
    knowledge_base_id: int
    filename: str
    size: int
    status: DocumentStatus
    created_at: datetime
