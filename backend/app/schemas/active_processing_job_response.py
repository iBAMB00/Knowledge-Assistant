from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.constants.processing_job_stage import ProcessingJobStage
from app.constants.processing_job_status import ProcessingJobStatus
from app.constants.processing_job_type import ProcessingJobType


class ActiveProcessingJobResponse(BaseModel):
    """
    文档列表中的活动处理任务摘要。

    只暴露前端展示当前任务状态所需的字段，
    不返回完整任务历史信息。
    """

    model_config = ConfigDict(
        from_attributes=True,
    )

    id: int
    job_type: ProcessingJobType
    status: ProcessingJobStatus
    stage: ProcessingJobStage

    progress: int = Field(
        ge=0,
        le=100,
    )

    error_message: str | None
    started_at: datetime | None
