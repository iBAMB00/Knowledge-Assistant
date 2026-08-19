import logging
from datetime import datetime

from pydantic import BaseModel, ConfigDict
from sqlalchemy.orm import Session

from app.agent.context import ToolExecutionContext
from app.agent.tools.base import (
    BaseAgentTool,
    ToolExecutionError,
    ToolRiskLevel,
)
from app.services.knowledge_base_service import KnowledgeBaseService


logger = logging.getLogger(__name__)


class KnowledgeBaseListInput(BaseModel):
    """list_knowledge_bases 不需要模型提供业务参数。"""

    model_config = ConfigDict(extra="forbid")


class KnowledgeBaseListItem(BaseModel):
    """Agent 可见的单个知识库摘要。"""

    model_config = ConfigDict(extra="forbid")

    id: int
    owner_id: int
    name: str
    description: str | None
    created_at: datetime


class KnowledgeBaseListOutput(BaseModel):
    """当前调用主体可访问的知识库列表。"""

    model_config = ConfigDict(extra="forbid")

    count: int
    items: list[KnowledgeBaseListItem]


class KnowledgeBaseListTool(
    BaseAgentTool[KnowledgeBaseListInput, KnowledgeBaseListOutput]
):
    """把现有 KnowledgeBaseService 安全包装成只读列表 Tool。"""

    name = "list_knowledge_bases"
    version = "1.0.0"
    description = (
        "List knowledge bases accessible to the current authenticated user. "
        "User identity and role are injected by the server and cannot be "
        "provided or overridden by model arguments."
    )
    risk_level = ToolRiskLevel.READ_ONLY
    input_model = KnowledgeBaseListInput
    output_model = KnowledgeBaseListOutput

    def __init__(self, knowledge_base_service: KnowledgeBaseService) -> None:
        self.knowledge_base_service = knowledge_base_service

    def execute(
        self,
        db: Session,
        context: ToolExecutionContext,
        tool_input: KnowledgeBaseListInput,
    ) -> KnowledgeBaseListOutput:
        """返回当前可信主体可访问的知识库，不复制列表业务规则。"""

        del tool_input
        principal = context.to_access_principal()

        try:
            knowledge_bases = self.knowledge_base_service.list_accessible(
                db=db,
                user=principal,
            )
        except Exception as exc:
            logger.error(
                "KnowledgeBaseListTool failed: request_id=%s error_type=%s",
                context.request_id,
                type(exc).__name__,
            )
            raise ToolExecutionError(
                "knowledge base listing failed"
            ) from exc

        items = [
            KnowledgeBaseListItem(
                id=knowledge_base.id,
                owner_id=knowledge_base.owner_id,
                name=knowledge_base.name,
                description=knowledge_base.description,
                created_at=knowledge_base.created_at,
            )
            for knowledge_base in knowledge_bases
        ]

        return KnowledgeBaseListOutput(
            count=len(items),
            items=items,
        )
