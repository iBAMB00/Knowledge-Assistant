from datetime import datetime

from pydantic import BaseModel


class DocumentResponse(BaseModel):
    """
    文档查询响应模型。

    用于返回数据库中的文档信息。
    """

    id: int
    knowledge_base_id: int | None
    filename: str
    stored_name: str
    size: int
    status: str
    created_at: datetime