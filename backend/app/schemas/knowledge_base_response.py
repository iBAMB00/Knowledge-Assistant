from datetime import datetime

from pydantic import BaseModel, ConfigDict


class KnowledgeBaseResponse(BaseModel):
    """公开知识库信息。"""

    model_config = ConfigDict(from_attributes=True)

    id: int
    owner_id: int
    name: str
    description: str | None
    created_at: datetime
    updated_at: datetime
