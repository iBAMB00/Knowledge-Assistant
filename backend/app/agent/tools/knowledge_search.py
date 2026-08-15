import logging

from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy.orm import Session

from app.agent.context import ToolExecutionContext
from app.agent.evidence import build_knowledge_source_ref
from app.agent.tools.base import (
    BaseAgentTool,
    ToolExecutionError,
    ToolInvalidArgumentsError,
    ToolResourceNotFoundError,
    ToolRiskLevel,
)
from app.services.knowledge_base_access_policy import (
    KnowledgeBaseAccessPolicy,
    ResourceAccessNotFoundError,
)
from app.services.retrieval_service import RetrievalService


logger = logging.getLogger(__name__)


class KnowledgeSearchInput(BaseModel):
    """
    search_knowledge 的模型可见输入。

    knowledge_base_id / user_id / role 等安全上下文故意不在此处，
    防止模型通过 Tool 参数覆盖服务端可信范围。
    """

    model_config = ConfigDict(extra="forbid")

    query: str = Field(
        min_length=1,
        max_length=2000,
        description="需要从当前知识库检索的问题或检索语句。",
    )
    top_k: int | None = Field(
        default=None,
        strict=True,
        gt=0,
        le=10,
        description="最多返回多少条证据；省略时使用服务端默认值。",
    )
    document_id: int | None = Field(
        default=None,
        strict=True,
        gt=0,
        description="可选：仅在当前知识库内检索指定文档。",
    )

    @field_validator("query", mode="before")
    @classmethod
    def normalize_query(cls, value: object) -> object:
        """去除查询首尾空白，并拒绝纯空白输入。"""

        if not isinstance(value, str):
            return value

        normalized = value.strip()
        if not normalized:
            raise ValueError("query cannot be empty")
        return normalized


class KnowledgeSearchItem(BaseModel):
    """单条知识检索证据。"""

    model_config = ConfigDict(extra="forbid")

    document_id: int
    filename: str
    chunk_id: int
    chunk_index: int
    content: str
    score: float
    source_ref: str = Field(
        min_length=1,
        description=(
            "Stable evidence reference. When the final answer uses this "
            "evidence, cite it exactly as [source:<source_ref>]."
        ),
    )


class KnowledgeSearchOutput(BaseModel):
    """search_knowledge 的结构化结果。"""

    model_config = ConfigDict(extra="forbid")

    result_count: int = Field(ge=0)
    items: list[KnowledgeSearchItem]


class KnowledgeSearchTool(
    BaseAgentTool[KnowledgeSearchInput, KnowledgeSearchOutput]
):
    """把现有 RetrievalService 安全包装成只读 Agent Tool。"""

    name = "search_knowledge"
    version = "1.1.0"
    description = (
        "Search the current server-authorized knowledge base for evidence "
        "relevant to the user's query. The knowledge-base scope and user "
        "identity are injected by the server and cannot be overridden."
    )
    risk_level = ToolRiskLevel.READ_ONLY
    input_model = KnowledgeSearchInput
    output_model = KnowledgeSearchOutput

    def __init__(
        self,
        retrieval_service: RetrievalService,
        access_policy: KnowledgeBaseAccessPolicy,
    ) -> None:
        self.retrieval_service = retrieval_service
        self.access_policy = access_policy

    def extract_evidence_refs(
        self,
        output: KnowledgeSearchOutput,
    ) -> list[str]:
        """只暴露无正文 source_ref，供 Eval/Tracing 使用。"""

        return [item.source_ref for item in output.items]

    def execute(
        self,
        db: Session,
        context: ToolExecutionContext,
        tool_input: KnowledgeSearchInput,
    ) -> KnowledgeSearchOutput:
        """
        在可信 KnowledgeBase 范围内执行检索。

        Tool 只做权限边界、参数转交和结果适配；
        检索算法仍完全由 RetrievalService 负责。
        """

        principal = context.to_access_principal()

        try:
            self.access_policy.get_accessible_knowledge_base(
                db=db,
                knowledge_base_id=context.knowledge_base_id,
                user=principal,
            )

            if tool_input.document_id is not None:
                self.access_policy.ensure_document_in_knowledge_base(
                    db=db,
                    document_id=tool_input.document_id,
                    knowledge_base_id=context.knowledge_base_id,
                    user=principal,
                )

            results = self.retrieval_service.retrieve(
                db=db,
                query=tool_input.query,
                top_k=tool_input.top_k,
                document_id=tool_input.document_id,
                knowledge_base_id=context.knowledge_base_id,
            )

        except ResourceAccessNotFoundError as exc:
            raise ToolResourceNotFoundError(str(exc)) from exc

        except ValueError as exc:
            raise ToolInvalidArgumentsError(str(exc)) from exc

        except Exception as exc:
            logger.error(
                "KnowledgeSearchTool failed: request_id=%s error_type=%s",
                context.request_id,
                type(exc).__name__,
            )
            raise ToolExecutionError("knowledge search failed") from exc

        items = [
            KnowledgeSearchItem(
                document_id=result.document_id,
                filename=result.filename,
                chunk_id=result.chunk_id,
                chunk_index=result.chunk_index,
                content=result.content,
                score=result.score,
                source_ref=build_knowledge_source_ref(
                    document_id=result.document_id,
                    chunk_id=result.chunk_id,
                ),
            )
            for result in results
        ]

        return KnowledgeSearchOutput(
            result_count=len(items),
            items=items,
        )
