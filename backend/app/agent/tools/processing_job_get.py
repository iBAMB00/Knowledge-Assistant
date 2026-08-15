import logging
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Session

from app.agent.context import ToolExecutionContext
from app.agent.tools.base import (
    BaseAgentTool,
    ToolExecutionError,
    ToolResourceNotFoundError,
    ToolRiskLevel,
)
from app.constants.processing_job_stage import ProcessingJobStage
from app.constants.processing_job_status import ProcessingJobStatus
from app.constants.processing_job_type import ProcessingJobType
from app.services.knowledge_base_access_policy import (
    KnowledgeBaseAccessPolicy,
    ResourceAccessNotFoundError,
)
from app.services.processing_job_service import (
    ProcessingJobNotFoundError,
    ProcessingJobService,
)


logger = logging.getLogger(__name__)


class ProcessingJobGetInput(BaseModel):
    """get_processing_job 的模型可见参数。"""

    model_config = ConfigDict(extra="forbid")

    job_id: int = Field(
        strict=True,
        gt=0,
        description="Processing job ID to inspect.",
    )


class ProcessingJobGetOutput(BaseModel):
    """Agent 可见的文档处理任务公开状态。"""

    model_config = ConfigDict(extra="forbid")

    id: int
    document_id: int
    job_type: ProcessingJobType
    status: ProcessingJobStatus
    stage: ProcessingJobStage
    progress: int
    error_message: str | None
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None
    finished_at: datetime | None


class ProcessingJobGetTool(
    BaseAgentTool[ProcessingJobGetInput, ProcessingJobGetOutput]
):
    """把 ProcessingJobService.get_job 包装成当前 KB 范围内只读 Tool。"""

    name = "get_processing_job"
    version = "1.0.0"
    description = (
        "Get the public status of one document processing job. The job is "
        "returned only when its document belongs to the server-authorized "
        "knowledge base."
    )
    risk_level = ToolRiskLevel.READ_ONLY
    input_model = ProcessingJobGetInput
    output_model = ProcessingJobGetOutput

    def __init__(
        self,
        processing_job_service: ProcessingJobService,
        access_policy: KnowledgeBaseAccessPolicy,
    ) -> None:
        self.processing_job_service = processing_job_service
        self.access_policy = access_policy

    def execute(
        self,
        db: Session,
        context: ToolExecutionContext,
        tool_input: ProcessingJobGetInput,
    ) -> ProcessingJobGetOutput:
        """查询 Job 后校验其 Document 必须属于当前可信 KnowledgeBase。"""

        principal = context.to_access_principal()

        try:
            job = self.processing_job_service.get_job(
                db=db,
                job_id=tool_input.job_id,
            )
            self.access_policy.ensure_document_in_knowledge_base(
                db=db,
                document_id=job.document_id,
                knowledge_base_id=context.knowledge_base_id,
                user=principal,
            )
        except (ProcessingJobNotFoundError, ResourceAccessNotFoundError) as exc:
            raise ToolResourceNotFoundError(
                "processing job not found"
            ) from exc
        except Exception as exc:
            logger.error(
                "ProcessingJobGetTool failed: request_id=%s error_type=%s",
                context.request_id,
                type(exc).__name__,
            )
            raise ToolExecutionError("processing job lookup failed") from exc

        return ProcessingJobGetOutput(
            id=job.id,
            document_id=job.document_id,
            job_type=ProcessingJobType(job.job_type),
            status=ProcessingJobStatus(job.status),
            stage=ProcessingJobStage(job.stage),
            progress=job.progress,
            error_message=job.error_message,
            created_at=job.created_at,
            updated_at=job.updated_at,
            started_at=job.started_at,
            finished_at=job.finished_at,
        )
