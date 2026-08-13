from typing import Any

import pytest
from pydantic import ValidationError
from sqlalchemy.orm import Session

from app.agent.context import ToolExecutionContext
from app.agent.tools.base import (
    ToolExecutionError,
    ToolResourceNotFoundError,
    ToolRiskLevel,
)
from app.agent.tools.knowledge_search import (
    KnowledgeSearchInput,
    KnowledgeSearchTool,
)
from app.constants.user_role import UserRole
from app.models.database.document import Document
from app.models.database.knowledge_base import KnowledgeBase
from app.models.database.user import User
from app.repositories.document_repository import DocumentRepository
from app.repositories.knowledge_base_repository import KnowledgeBaseRepository
from app.schemas.vector_search_result import VectorSearchResult
from app.services.knowledge_base_access_policy import KnowledgeBaseAccessPolicy


class FakeRetrievalService:
    """KnowledgeSearchTool 测试使用的最小 RetrievalService 替身。"""

    def __init__(
        self,
        results: list[VectorSearchResult] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.results = results or []
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def retrieve(self, **kwargs: Any) -> list[VectorSearchResult]:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.results


def _create_user(
    db: Session,
    email: str,
    role: UserRole = UserRole.USER,
) -> User:
    user = User(
        email=email,
        password_hash="test-password-hash",
        role=role.value,
        is_active=True,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def _create_knowledge_base(
    db: Session,
    owner: User,
    name: str,
) -> KnowledgeBase:
    knowledge_base = KnowledgeBase(
        owner_id=owner.id,
        name=name,
    )
    db.add(knowledge_base)
    db.commit()
    db.refresh(knowledge_base)
    return knowledge_base


def _build_policy() -> KnowledgeBaseAccessPolicy:
    return KnowledgeBaseAccessPolicy(
        knowledge_base_repository=KnowledgeBaseRepository(),
        document_repository=DocumentRepository(),
    )


def _build_context(
    user: User,
    knowledge_base_id: int,
) -> ToolExecutionContext:
    return ToolExecutionContext(
        user_id=user.id,
        role=UserRole(user.role),
        knowledge_base_id=knowledge_base_id,
        request_id="agent-tool-test-request",
    )


def test_knowledge_search_contract_excludes_trusted_context() -> None:
    """LLM 可见 Tool Schema 不得包含服务端可信身份与权限字段。"""

    tool = KnowledgeSearchTool(
        retrieval_service=FakeRetrievalService(),  # type: ignore[arg-type]
        access_policy=_build_policy(),
    )

    contract = tool.get_contract()
    input_properties = contract.input_schema["properties"]

    assert contract.name == "search_knowledge"
    assert contract.version == "1.0.0"
    assert contract.risk_level == ToolRiskLevel.READ_ONLY
    assert set(input_properties) == {
        "query",
        "top_k",
        "document_id",
    }
    assert "user_id" not in input_properties
    assert "role" not in input_properties
    assert "knowledge_base_id" not in input_properties
    assert "request_id" not in input_properties
    assert "agent_run_id" not in input_properties


def test_knowledge_search_input_rejects_empty_or_trusted_fields() -> None:
    """模型参数必须拒绝空查询以及越权注入可信上下文。"""

    with pytest.raises(ValidationError):
        KnowledgeSearchInput(query="   ")

    with pytest.raises(ValidationError):
        KnowledgeSearchInput.model_validate(
            {
                "query": "查询部署流程",
                "knowledge_base_id": 999,
            }
        )


def test_knowledge_search_executes_retrieval_in_trusted_scope(
    db: Session,
) -> None:
    """合法调用必须复用既有权限策略并固定 KnowledgeBase 范围。"""

    owner = _create_user(db, "tool-owner@example.com")
    knowledge_base = _create_knowledge_base(db, owner, "Tool Owner KB")
    retrieval = FakeRetrievalService(
        results=[
            VectorSearchResult(
                document_id=11,
                filename="runbook.txt",
                chunk_id=101,
                chunk_index=3,
                content="部署前必须完成数据库迁移。",
                score=0.93,
            )
        ]
    )
    tool = KnowledgeSearchTool(
        retrieval_service=retrieval,  # type: ignore[arg-type]
        access_policy=_build_policy(),
    )

    result = tool.execute(
        db=db,
        context=_build_context(owner, knowledge_base.id),
        tool_input=KnowledgeSearchInput(
            query="  部署前要做什么？  ",
            top_k=3,
        ),
    )

    assert result.result_count == 1
    assert result.items[0].content == "部署前必须完成数据库迁移。"
    assert retrieval.calls[0]["query"] == "部署前要做什么？"
    assert retrieval.calls[0]["top_k"] == 3
    assert retrieval.calls[0]["knowledge_base_id"] == knowledge_base.id
    assert retrieval.calls[0]["document_id"] is None


def test_knowledge_search_returns_structured_empty_result(
    db: Session,
) -> None:
    """无召回时返回结构化空结果，而不是 None 或异常。"""

    owner = _create_user(db, "tool-empty@example.com")
    knowledge_base = _create_knowledge_base(db, owner, "Empty Tool KB")
    tool = KnowledgeSearchTool(
        retrieval_service=FakeRetrievalService(),  # type: ignore[arg-type]
        access_policy=_build_policy(),
    )

    result = tool.execute(
        db=db,
        context=_build_context(owner, knowledge_base.id),
        tool_input=KnowledgeSearchInput(query="没有答案的问题"),
    )

    assert result.result_count == 0
    assert result.items == []


def test_knowledge_search_blocks_cross_user_access_before_retrieval(
    db: Session,
) -> None:
    """跨用户访问必须在调用 RetrievalService 前被统一权限策略阻断。"""

    owner = _create_user(db, "tool-cross-owner@example.com")
    other = _create_user(db, "tool-cross-other@example.com")
    knowledge_base = _create_knowledge_base(db, owner, "Cross User KB")
    retrieval = FakeRetrievalService()
    tool = KnowledgeSearchTool(
        retrieval_service=retrieval,  # type: ignore[arg-type]
        access_policy=_build_policy(),
    )

    with pytest.raises(
        ToolResourceNotFoundError,
        match="knowledge base not found",
    ):
        tool.execute(
            db=db,
            context=_build_context(other, knowledge_base.id),
            tool_input=KnowledgeSearchInput(query="内部资料"),
        )

    assert retrieval.calls == []


def test_knowledge_search_blocks_document_outside_context_kb(
    db: Session,
) -> None:
    """document_id 是业务参数，但不能逃逸 Trusted KB Scope。"""

    owner = _create_user(db, "tool-doc-owner@example.com")
    kb_one = _create_knowledge_base(db, owner, "Document Scope One")
    kb_two = _create_knowledge_base(db, owner, "Document Scope Two")
    document = Document(
        knowledge_base_id=kb_two.id,
        filename="other-kb.txt",
        storage_key="other-kb.txt",
        stored_name="other-kb.txt",
        path="uploads/other-kb.txt",
        size=10,
        status="uploaded",
    )
    db.add(document)
    db.commit()
    db.refresh(document)

    retrieval = FakeRetrievalService()
    tool = KnowledgeSearchTool(
        retrieval_service=retrieval,  # type: ignore[arg-type]
        access_policy=_build_policy(),
    )

    with pytest.raises(
        ToolResourceNotFoundError,
        match="document not found",
    ):
        tool.execute(
            db=db,
            context=_build_context(owner, kb_one.id),
            tool_input=KnowledgeSearchInput(
                query="读取另一知识库文档",
                document_id=document.id,
            ),
        )

    assert retrieval.calls == []


def test_knowledge_search_maps_unexpected_service_error(
    db: Session,
) -> None:
    """底层未预期异常不能直接泄漏为 Agent Tool 的内部错误细节。"""

    owner = _create_user(db, "tool-error@example.com")
    knowledge_base = _create_knowledge_base(db, owner, "Tool Error KB")
    tool = KnowledgeSearchTool(
        retrieval_service=FakeRetrievalService(
            error=RuntimeError("provider secret detail")
        ),  # type: ignore[arg-type]
        access_policy=_build_policy(),
    )

    with pytest.raises(
        ToolExecutionError,
        match="knowledge search failed",
    ) as error:
        tool.execute(
            db=db,
            context=_build_context(owner, knowledge_base.id),
            tool_input=KnowledgeSearchInput(query="触发异常"),
        )

    assert "provider secret detail" not in str(error.value)
