import logging
from datetime import datetime

from pydantic import BaseModel, ConfigDict
from sqlalchemy.orm import Session

from app.agent.context import ToolExecutionContext
from app.agent.tools.base import (
    BaseAgentTool,
    ToolExecutionError,
    ToolInvalidArgumentsError,
    ToolResourceNotFoundError,
    ToolRiskLevel,
)
from app.constants.document_status import DocumentStatus
from app.constants.processing_job_stage import ProcessingJobStage
from app.constants.processing_job_status import ProcessingJobStatus
from app.constants.processing_job_type import ProcessingJobType
from app.services.document_service import DocumentService
from app.services.knowledge_base_access_policy import (
    KnowledgeBaseAccessPolicy,
    ResourceAccessNotFoundError,
)


logger = logging.getLogger(__name__)


class DocumentListInput(BaseModel):
    """list_documents 使用服务端已授权 KnowledgeBase，不接收范围参数。"""

    model_config = ConfigDict(extra="forbid")


class DocumentActiveJobItem(BaseModel):
    """文档当前活动处理任务的最小摘要。"""

    model_config = ConfigDict(extra="forbid")

    id: int
    job_type: ProcessingJobType
    status: ProcessingJobStatus
    stage: ProcessingJobStage
    progress: int
    started_at: datetime | None


class DocumentListItem(BaseModel):
    """Agent 可见的文档列表项。"""

    model_config = ConfigDict(extra="forbid")

    id: int
    knowledge_base_id: int
    filename: str
    size: int
    status: DocumentStatus
    created_at: datetime
    active_job: DocumentActiveJobItem | None = None


class DocumentListOutput(BaseModel):
    """当前 KnowledgeBase 的文档列表。"""

    model_config = ConfigDict(extra="forbid")

    count: int
    items: list[DocumentListItem]


class DocumentListTool(
    BaseAgentTool[DocumentListInput, DocumentListOutput]
):
    """把现有 DocumentService.list_documents 包装成只读 Agent Tool。"""

    name = "list_documents"
    version = "1.0.0"
    description = (
        "List documents in the server-authorized knowledge base, including "
        "a minimal active processing-job summary when present. The knowledge "
        "base scope is injected by the server and cannot be overridden."
    )
    risk_level = ToolRiskLevel.READ_ONLY
    input_model = DocumentListInput
    output_model = DocumentListOutput

    def __init__(
        self,
        document_service: DocumentService,
        access_policy: KnowledgeBaseAccessPolicy,
    ) -> None:
        self.document_service = document_service
        self.access_policy = access_policy

    def execute(
        self,
        db: Session,
        context: ToolExecutionContext,
        tool_input: DocumentListInput,
    ) -> DocumentListOutput:
        """校验当前 KB scope 后复用 DocumentService 返回文档列表。"""

        del tool_input
        principal = context.to_access_principal()

        try:
            self.access_policy.get_accessible_knowledge_base(
                db=db,
                knowledge_base_id=context.knowledge_base_id,
                user=principal,
            )
            documents = self.document_service.list_documents(
                db=db,
                knowledge_base_id=context.knowledge_base_id,
            )
        except ResourceAccessNotFoundError as exc:
            raise ToolResourceNotFoundError(str(exc)) from exc
        except ValueError as exc:
            raise ToolInvalidArgumentsError(str(exc)) from exc
        except Exception as exc:
            logger.error(
                "DocumentListTool failed: request_id=%s error_type=%s",
                context.request_id,
                type(exc).__name__,
            )
            raise ToolExecutionError("document listing failed") from exc

        items: list[DocumentListItem] = []
        for document in documents:
            active_job = document.active_job
            items.append(
                DocumentListItem(
                    id=document.id,
                    knowledge_base_id=document.knowledge_base_id,
                    filename=document.filename,
                    size=document.size,
                    status=document.status,
                    created_at=document.created_at,
                    active_job=(
                        DocumentActiveJobItem(
                            id=active_job.id,
                            job_type=active_job.job_type,
                            status=active_job.status,
                            stage=active_job.stage,
                            progress=active_job.progress,
                            started_at=active_job.started_at,
                        )
                        if active_job is not None
                        else None
                    ),
                )
            )

        return DocumentListOutput(
            count=len(items),
            items=items,
        )
