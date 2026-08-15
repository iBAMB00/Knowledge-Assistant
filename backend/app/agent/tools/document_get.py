import logging
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field
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
from app.services.document_service import DocumentService
from app.services.knowledge_base_access_policy import (
    KnowledgeBaseAccessPolicy,
    ResourceAccessNotFoundError,
)


logger = logging.getLogger(__name__)


class DocumentGetInput(BaseModel):
    """get_document 的模型可见参数。"""

    model_config = ConfigDict(extra="forbid")

    document_id: int = Field(
        strict=True,
        gt=0,
        description="Document ID to inspect inside the current knowledge base.",
    )


class DocumentGetOutput(BaseModel):
    """Agent 可见的文档公开元数据。"""

    model_config = ConfigDict(extra="forbid")

    id: int
    knowledge_base_id: int
    filename: str
    size: int
    status: DocumentStatus
    created_at: datetime


class DocumentGetTool(
    BaseAgentTool[DocumentGetInput, DocumentGetOutput]
):
    """读取当前 KnowledgeBase 内单个文档公开元数据。"""

    name = "get_document"
    version = "1.0.0"
    description = (
        "Get public metadata for one document inside the server-authorized "
        "knowledge base. The document must belong to that knowledge base."
    )
    risk_level = ToolRiskLevel.READ_ONLY
    input_model = DocumentGetInput
    output_model = DocumentGetOutput

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
        tool_input: DocumentGetInput,
    ) -> DocumentGetOutput:
        """先做当前 KB 下的文档授权，再复用 DocumentService。"""

        principal = context.to_access_principal()

        try:
            self.access_policy.ensure_document_in_knowledge_base(
                db=db,
                document_id=tool_input.document_id,
                knowledge_base_id=context.knowledge_base_id,
                user=principal,
            )
            document = self.document_service.get_document_by_id(
                db=db,
                document_id=tool_input.document_id,
            )
        except ResourceAccessNotFoundError as exc:
            raise ToolResourceNotFoundError("document not found") from exc
        except ValueError as exc:
            if str(exc) == "document not found":
                raise ToolResourceNotFoundError("document not found") from exc
            raise ToolInvalidArgumentsError(str(exc)) from exc
        except Exception as exc:
            logger.error(
                "DocumentGetTool failed: request_id=%s error_type=%s",
                context.request_id,
                type(exc).__name__,
            )
            raise ToolExecutionError("document lookup failed") from exc

        return DocumentGetOutput(
            id=document.id,
            knowledge_base_id=document.knowledge_base_id,
            filename=document.filename,
            size=document.size,
            status=document.status,
            created_at=document.created_at,
        )
