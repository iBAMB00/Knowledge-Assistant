from datetime import datetime, timezone
from types import SimpleNamespace
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
from app.agent.tools.document_get import (
    DocumentGetInput,
    DocumentGetTool,
)
from app.agent.tools.document_list import (
    DocumentListInput,
    DocumentListTool,
)
from app.agent.tools.knowledge_base_list import (
    KnowledgeBaseListInput,
    KnowledgeBaseListTool,
)
from app.agent.tools.processing_job_get import (
    ProcessingJobGetInput,
    ProcessingJobGetTool,
)
from app.constants.document_status import DocumentStatus
from app.constants.processing_job_stage import ProcessingJobStage
from app.constants.processing_job_status import ProcessingJobStatus
from app.constants.processing_job_type import ProcessingJobType
from app.constants.user_role import UserRole
from app.models.database.document import Document
from app.models.database.knowledge_base import KnowledgeBase
from app.models.database.user import User
from app.repositories.document_repository import DocumentRepository
from app.repositories.knowledge_base_repository import KnowledgeBaseRepository
from app.schemas.active_processing_job_response import ActiveProcessingJobResponse
from app.schemas.document_list_item_response import DocumentListItemResponse
from app.schemas.document_response import DocumentResponse
from app.services.knowledge_base_access_policy import KnowledgeBaseAccessPolicy
from app.services.knowledge_base_service import KnowledgeBaseService


class FakeKnowledgeBaseService:
    def __init__(
        self,
        items: list[Any] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.items = items or []
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def list_accessible(self, **kwargs: Any) -> list[Any]:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.items


class FakeDocumentService:
    def __init__(
        self,
        *,
        documents: list[DocumentListItemResponse] | None = None,
        document: DocumentResponse | None = None,
        error: Exception | None = None,
    ) -> None:
        self.documents = documents or []
        self.document = document
        self.error = error
        self.list_calls: list[dict[str, Any]] = []
        self.get_calls: list[dict[str, Any]] = []

    def list_documents(self, **kwargs: Any) -> list[DocumentListItemResponse]:
        self.list_calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.documents

    def get_document_by_id(self, **kwargs: Any) -> DocumentResponse:
        self.get_calls.append(kwargs)
        if self.error is not None:
            raise self.error
        if self.document is None:
            raise ValueError("document not found")
        return self.document


class FakeProcessingJobService:
    def __init__(
        self,
        job: Any | None = None,
        error: Exception | None = None,
    ) -> None:
        self.job = job
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def get_job(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        if self.job is None:
            from app.services.processing_job_service import ProcessingJobNotFoundError

            raise ProcessingJobNotFoundError("processing job not found")
        return self.job


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


def _create_document(
    db: Session,
    knowledge_base: KnowledgeBase,
    filename: str,
) -> Document:
    document = Document(
        knowledge_base_id=knowledge_base.id,
        filename=filename,
        storage_key=f"tool-test/{knowledge_base.id}/{filename}",
        size=128,
        status=DocumentStatus.PARSED.value,
    )
    db.add(document)
    db.commit()
    db.refresh(document)
    return document


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
        request_id="agent-read-tool-test",
    )


def test_read_tool_contracts_are_read_only_and_exclude_trusted_scope() -> None:
    """四个补充 Tool 都必须是只读，并拒绝模型注入可信上下文。"""

    kb_tool = KnowledgeBaseListTool(FakeKnowledgeBaseService())  # type: ignore[arg-type]
    doc_service = FakeDocumentService()
    policy = _build_policy()
    list_tool = DocumentListTool(doc_service, policy)  # type: ignore[arg-type]
    get_tool = DocumentGetTool(doc_service, policy)  # type: ignore[arg-type]
    job_tool = ProcessingJobGetTool(
        FakeProcessingJobService(),  # type: ignore[arg-type]
        policy,
    )

    contracts = {
        tool.name: tool.get_contract()
        for tool in [kb_tool, list_tool, get_tool, job_tool]
    }

    assert set(contracts) == {
        "list_knowledge_bases",
        "list_documents",
        "get_document",
        "get_processing_job",
    }
    assert all(
        contract.risk_level == ToolRiskLevel.READ_ONLY
        for contract in contracts.values()
    )
    assert contracts["list_knowledge_bases"].input_schema["properties"] == {}
    assert contracts["list_documents"].input_schema["properties"] == {}
    assert set(contracts["get_document"].input_schema["properties"]) == {
        "document_id"
    }
    assert set(contracts["get_processing_job"].input_schema["properties"]) == {
        "job_id"
    }

    with pytest.raises(ValidationError):
        DocumentListInput.model_validate({"knowledge_base_id": 999})

    with pytest.raises(ValidationError):
        DocumentGetInput.model_validate(
            {"document_id": 1, "user_id": 999}
        )


def test_list_knowledge_bases_uses_trusted_principal(db: Session) -> None:
    """list_knowledge_bases 的用户身份必须来自 ToolExecutionContext。"""

    user = _create_user(db, "kb-list-tool@example.com")
    now = datetime.now(timezone.utc)
    service = FakeKnowledgeBaseService(
        items=[
            SimpleNamespace(
                id=21,
                owner_id=user.id,
                name="Runbooks",
                description="Operations",
                created_at=now,
            )
        ]
    )
    tool = KnowledgeBaseListTool(service)  # type: ignore[arg-type]

    result = tool.execute(
        db=db,
        context=_build_context(user, knowledge_base_id=1),
        tool_input=KnowledgeBaseListInput(),
    )

    assert result.count == 1
    assert result.items[0].name == "Runbooks"
    principal = service.calls[0]["user"]
    assert principal.id == user.id
    assert principal.role == user.role



def test_list_knowledge_bases_preserves_existing_user_and_admin_semantics(
    db: Session,
) -> None:
    """Tool 使用最小 Principal 后仍保持现有 KnowledgeBaseService 权限语义。"""

    owner_a = _create_user(db, "kb-tool-owner-a@example.com")
    owner_b = _create_user(db, "kb-tool-owner-b@example.com")
    admin = _create_user(
        db,
        "kb-tool-admin@example.com",
        role=UserRole.ADMIN,
    )
    kb_a = _create_knowledge_base(db, owner_a, "Owner A")
    kb_b = _create_knowledge_base(db, owner_b, "Owner B")

    repository = KnowledgeBaseRepository()
    document_repository = DocumentRepository()
    service = KnowledgeBaseService(
        knowledge_base_repository=repository,
        document_repository=document_repository,
        access_policy=KnowledgeBaseAccessPolicy(
            knowledge_base_repository=repository,
            document_repository=document_repository,
        ),
    )
    tool = KnowledgeBaseListTool(service)

    owner_result = tool.execute(
        db=db,
        context=_build_context(owner_a, kb_a.id),
        tool_input=KnowledgeBaseListInput(),
    )
    admin_result = tool.execute(
        db=db,
        context=_build_context(admin, kb_a.id),
        tool_input=KnowledgeBaseListInput(),
    )

    assert [item.id for item in owner_result.items] == [kb_a.id]
    assert {item.id for item in admin_result.items} >= {kb_a.id, kb_b.id}

def test_list_documents_uses_current_kb_scope_and_returns_structured_items(
    db: Session,
) -> None:
    """文档列表只能读取当前可信 KB，并保留最小活动任务摘要。"""

    user = _create_user(db, "doc-list-tool@example.com")
    kb = _create_knowledge_base(db, user, "Documents")
    now = datetime.now(timezone.utc)
    service = FakeDocumentService(
        documents=[
            DocumentListItemResponse(
                id=7,
                knowledge_base_id=kb.id,
                filename="deploy.md",
                size=512,
                status=DocumentStatus.PARSED,
                created_at=now,
                active_job=ActiveProcessingJobResponse(
                    id=30,
                    job_type=ProcessingJobType.EMBEDDING,
                    status=ProcessingJobStatus.RUNNING,
                    stage=ProcessingJobStage.EMBEDDING,
                    progress=70,
                    error_message=None,
                    started_at=now,
                ),
            )
        ]
    )
    tool = DocumentListTool(
        document_service=service,  # type: ignore[arg-type]
        access_policy=_build_policy(),
    )

    result = tool.execute(
        db=db,
        context=_build_context(user, kb.id),
        tool_input=DocumentListInput(),
    )

    assert result.count == 1
    assert result.items[0].filename == "deploy.md"
    assert result.items[0].active_job is not None
    assert result.items[0].active_job.progress == 70
    assert service.list_calls[0]["knowledge_base_id"] == kb.id


def test_list_documents_blocks_cross_user_before_service_call(
    db: Session,
) -> None:
    """越权 KB 必须在调用 DocumentService 前被拒绝。"""

    owner = _create_user(db, "doc-list-owner@example.com")
    attacker = _create_user(db, "doc-list-attacker@example.com")
    kb = _create_knowledge_base(db, owner, "Private")
    service = FakeDocumentService()
    tool = DocumentListTool(
        document_service=service,  # type: ignore[arg-type]
        access_policy=_build_policy(),
    )

    with pytest.raises(ToolResourceNotFoundError):
        tool.execute(
            db=db,
            context=_build_context(attacker, kb.id),
            tool_input=DocumentListInput(),
        )

    assert service.list_calls == []


def test_get_document_blocks_cross_kb_before_service_call(db: Session) -> None:
    """模型猜到其他 KB 的 document_id 也不能越过当前 Trusted KB。"""

    user = _create_user(db, "doc-get-tool@example.com")
    kb_a = _create_knowledge_base(db, user, "A")
    kb_b = _create_knowledge_base(db, user, "B")
    document = _create_document(db, kb_b, "secret.md")
    service = FakeDocumentService()
    tool = DocumentGetTool(
        document_service=service,  # type: ignore[arg-type]
        access_policy=_build_policy(),
    )

    with pytest.raises(ToolResourceNotFoundError):
        tool.execute(
            db=db,
            context=_build_context(user, kb_a.id),
            tool_input=DocumentGetInput(document_id=document.id),
        )

    assert service.get_calls == []


def test_get_document_returns_public_metadata(db: Session) -> None:
    """合法文档读取只返回公开元数据，不返回 storage_key/path。"""

    user = _create_user(db, "doc-get-ok@example.com")
    kb = _create_knowledge_base(db, user, "Docs")
    document = _create_document(db, kb, "guide.md")
    service = FakeDocumentService(
        document=DocumentResponse(
            id=document.id,
            knowledge_base_id=kb.id,
            filename=document.filename,
            size=document.size,
            status=DocumentStatus(document.status),
            created_at=document.created_at,
        )
    )
    tool = DocumentGetTool(
        document_service=service,  # type: ignore[arg-type]
        access_policy=_build_policy(),
    )

    result = tool.execute(
        db=db,
        context=_build_context(user, kb.id),
        tool_input=DocumentGetInput(document_id=document.id),
    )

    assert result.filename == "guide.md"
    assert "storage_key" not in result.model_dump()
    assert "path" not in result.model_dump()


def test_get_processing_job_enforces_job_document_kb_scope(db: Session) -> None:
    """job_id 可由模型选择，但 Job 对应 Document 必须属于当前 KB。"""

    user = _create_user(db, "job-get-tool@example.com")
    kb_a = _create_knowledge_base(db, user, "A")
    kb_b = _create_knowledge_base(db, user, "B")
    document_b = _create_document(db, kb_b, "other.md")
    now = datetime.now(timezone.utc)
    service = FakeProcessingJobService(
        job=SimpleNamespace(
            id=91,
            document_id=document_b.id,
            job_type=ProcessingJobType.FULL_PIPELINE.value,
            status=ProcessingJobStatus.RUNNING.value,
            stage=ProcessingJobStage.EMBEDDING.value,
            progress=50,
            error_message=None,
            created_at=now,
            updated_at=now,
            started_at=now,
            finished_at=None,
        )
    )
    tool = ProcessingJobGetTool(
        processing_job_service=service,  # type: ignore[arg-type]
        access_policy=_build_policy(),
    )

    with pytest.raises(ToolResourceNotFoundError):
        tool.execute(
            db=db,
            context=_build_context(user, kb_a.id),
            tool_input=ProcessingJobGetInput(job_id=91),
        )


def test_get_processing_job_returns_public_status(db: Session) -> None:
    """合法 Job 返回可供 Agent 判断进度的公开状态。"""

    user = _create_user(db, "job-get-ok@example.com")
    kb = _create_knowledge_base(db, user, "Jobs")
    document = _create_document(db, kb, "job.md")
    now = datetime.now(timezone.utc)
    service = FakeProcessingJobService(
        job=SimpleNamespace(
            id=44,
            document_id=document.id,
            job_type=ProcessingJobType.DOCUMENT_PROCESSING.value,
            status=ProcessingJobStatus.SUCCEEDED.value,
            stage=ProcessingJobStage.COMPLETED.value,
            progress=100,
            error_message=None,
            created_at=now,
            updated_at=now,
            started_at=now,
            finished_at=now,
        )
    )
    tool = ProcessingJobGetTool(
        processing_job_service=service,  # type: ignore[arg-type]
        access_policy=_build_policy(),
    )

    result = tool.execute(
        db=db,
        context=_build_context(user, kb.id),
        tool_input=ProcessingJobGetInput(job_id=44),
    )

    assert result.status == ProcessingJobStatus.SUCCEEDED
    assert result.progress == 100
    assert result.document_id == document.id


def test_read_tool_service_exception_is_sanitized(db: Session) -> None:
    """未处理的 Service 异常不能把底层详情直接暴露给模型。"""

    user = _create_user(db, "read-tool-error@example.com")
    service = FakeKnowledgeBaseService(
        error=RuntimeError("database-secret-detail")
    )
    tool = KnowledgeBaseListTool(service)  # type: ignore[arg-type]

    with pytest.raises(ToolExecutionError) as exc_info:
        tool.execute(
            db=db,
            context=_build_context(user, knowledge_base_id=1),
            tool_input=KnowledgeBaseListInput(),
        )

    assert str(exc_info.value) == "knowledge base listing failed"
    assert "database-secret-detail" not in str(exc_info.value)


def test_native_agent_runner_registers_complete_read_only_tool_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """生产 Runner 依赖装配必须真正注册 Roadmap 规定的五个只读 Tool。"""

    import app.api.dependencies.agent as agent_dependencies
    import app.services.llm_service as llm_service_module

    class FakeLLMService:
        pass

    agent_dependencies.get_native_agent_runner.cache_clear()

    monkeypatch.setattr(
        agent_dependencies,
        "get_agent_access_policy",
        lambda: object(),
    )
    monkeypatch.setattr(
        agent_dependencies,
        "get_agent_retrieval_service",
        lambda: object(),
    )
    monkeypatch.setattr(
        agent_dependencies,
        "get_agent_knowledge_base_service",
        lambda: object(),
    )
    monkeypatch.setattr(
        agent_dependencies,
        "get_agent_document_service",
        lambda: object(),
    )
    monkeypatch.setattr(
        agent_dependencies,
        "get_agent_processing_job_service",
        lambda: object(),
    )
    monkeypatch.setattr(
        llm_service_module,
        "LLMService",
        FakeLLMService,
    )

    try:
        runner = agent_dependencies.get_native_agent_runner()
    finally:
        agent_dependencies.get_native_agent_runner.cache_clear()

    assert [tool.name for tool in runner.tools] == [
        "search_knowledge",
        "list_knowledge_bases",
        "list_documents",
        "get_document",
        "get_processing_job",
    ]
    assert all(
        tool.risk_level == ToolRiskLevel.READ_ONLY
        for tool in runner.tools
    )
