from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.constants.processing_job_status import ProcessingJobStatus
from app.constants.processing_job_type import ProcessingJobType


class ProcessingJobResponse(BaseModel):
    """
    文档处理任务公开响应。
    """

    model_config = ConfigDict(
        from_attributes=True,
    )

    id: int
    document_id: int
    job_type: ProcessingJobType
    status: ProcessingJobStatus

    progress: int = Field(
        ge=0,
        le=100,
    )

    error_message: str | None

    created_at: datetime
    updated_at: datetime
    started_at: datetime | None
    finished_at: datetime | None