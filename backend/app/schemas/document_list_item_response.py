from app.schemas.active_processing_job_response import (
    ActiveProcessingJobResponse,
)
from app.schemas.document_response import DocumentResponse


class DocumentListItemResponse(DocumentResponse):
    """
    文档列表项响应。

    在文档基础信息上增加当前活动任务摘要；
    文档详情接口继续使用DocumentResponse，保持原契约不变。
    """

    active_job: ActiveProcessingJobResponse | None
