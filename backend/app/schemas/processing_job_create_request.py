from pydantic import BaseModel

from app.constants.processing_job_type import ProcessingJobType


class ProcessingJobCreateRequest(BaseModel):
    """
    创建文档处理任务请求。
    """

    job_type: ProcessingJobType