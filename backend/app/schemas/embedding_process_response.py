from pydantic import BaseModel


class EmbeddingProcessResponse(BaseModel):
    """
    文档向量化处理结果。
    """

    document_id: int

    processed_count: int