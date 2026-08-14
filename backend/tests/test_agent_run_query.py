import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.api.agent import router as agent_router
from app.api.dependencies.agent import (
    get_agent_access_policy,
    get_agent_run_query_service,
)
from app.api.dependencies.auth import get_current_user
from app.constants.agent_run_status import AgentRunStatus
from app.constants.agent_tool_call_status import AgentToolCallStatus
from app.constants.user_role import UserRole
from app.core.database import get_db
from app.models.database.agent_run import AgentRun
from app.models.database.agent_tool_call import AgentToolCall
from app.models.database.knowledge_base import KnowledgeBase
from app.models.database.user import User
from app.repositories.agent_run_repository import AgentRunRepository
from app.repositories.agent_tool_call_repository import AgentToolCallRepository
from app.repositories.document_repository import DocumentRepository
from app.repositories.knowledge_base_repository import KnowledgeBaseRepository
from app.services.agent_run_query_service import (
    AgentRunNotFoundError,
    AgentRunQueryService,
)
from app.services.knowledge_base_access_policy import KnowledgeBaseAccessPolicy


def _create_user(
    db: Session,
    *,
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


def _create_kb(
    db: Session,
    *,
    owner: User,
    name: str,
) -> KnowledgeBase:
    knowledge_base = KnowledgeBase(
        owner_id=owner.id,
        name=name,
        description="AgentRun query test",
    )
    db.add(knowledge_base)
    db.commit()
    db.refresh(knowledge_base)
    return knowledge_base


def _create_run(
    db: Session,
    *,
    user: User,
    knowledge_base: KnowledgeBase,
    request_id: str,
    status: AgentRunStatus = AgentRunStatus.SUCCEEDED,
    tool_call_count: int = 0,
    error_type: str | None = None,
) -> AgentRun:
    run = AgentRun(
        user_id=user.id,
        knowledge_base_id=knowledge_base.id,
        request_id=request_id,
        status=status.value,
        model_provider="test-provider",
        model_name="test-model",
        tool_call_count=tool_call_count,
        error_type=error_type,
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    return run


def _create_tool_call(
    db: Session,
    *,
    run: AgentRun,
    provider_call_id: str,
    tool_name: str = "search_knowledge",
    status: AgentToolCallStatus = AgentToolCallStatus.SUCCEEDED,
    duration_ms: int | None = 12,
    error_type: str | None = None,
) -> AgentToolCall:
    tool_call = AgentToolCall(
        agent_run_id=run.id,
        provider_call_id=provider_call_id,
        tool_name=tool_name,
        tool_version="1.0.0",
        status=status.value,
        duration_ms=duration_ms,
        error_type=error_type,
    )
    db.add(tool_call)
    db.commit()
    db.refresh(tool_call)
    return tool_call


def _query_service() -> AgentRunQueryService:
    return AgentRunQueryService(
        agent_run_repository=AgentRunRepository(),
        tool_call_repository=AgentToolCallRepository(),
    )


def _build_client(db: Session, current_user: User) -> TestClient:
    app = FastAPI()
    app.include_router(agent_router)

    def override_get_db():
        yield db

    access_policy = KnowledgeBaseAccessPolicy(
        knowledge_base_repository=KnowledgeBaseRepository(),
        document_repository=DocumentRepository(),
    )

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_current_user] = lambda: current_user
    app.dependency_overrides[get_agent_access_policy] = lambda: access_policy
    app.dependency_overrides[get_agent_run_query_service] = _query_service

    return TestClient(app)


def test_query_service_lists_only_owned_runs_with_filter_and_limit(
    db: Session,
) -> None:
    owner = _create_user(db, email="run-owner@example.com")
    other = _create_user(db, email="run-other@example.com")
    kb_a = _create_kb(db, owner=owner, name="Run Query A")
    kb_b = _create_kb(db, owner=owner, name="Run Query B")
    other_kb = _create_kb(db, owner=other, name="Run Query Other")

    oldest = _create_run(
        db,
        user=owner,
        knowledge_base=kb_a,
        request_id="run-oldest",
    )
    filtered = _create_run(
        db,
        user=owner,
        knowledge_base=kb_b,
        request_id="run-filtered",
    )
    newest = _create_run(
        db,
        user=owner,
        knowledge_base=kb_a,
        request_id="run-newest",
    )
    _create_run(
        db,
        user=other,
        knowledge_base=other_kb,
        request_id="run-foreign",
    )

    service = _query_service()

    recent = service.list_owned_runs(
        db=db,
        user_id=owner.id,
        limit=2,
    )
    assert [run.id for run in recent] == [newest.id, filtered.id]

    filtered_runs = service.list_owned_runs(
        db=db,
        user_id=owner.id,
        knowledge_base_id=kb_a.id,
        limit=20,
    )
    assert [run.id for run in filtered_runs] == [newest.id, oldest.id]


def test_query_service_detail_rejects_cross_user_even_for_admin(
    db: Session,
) -> None:
    owner = _create_user(db, email="detail-owner@example.com")
    admin = _create_user(
        db,
        email="detail-admin@example.com",
        role=UserRole.ADMIN,
    )
    kb = _create_kb(db, owner=owner, name="Detail Isolation KB")
    run = _create_run(
        db,
        user=owner,
        knowledge_base=kb,
        request_id="detail-isolation",
    )

    with pytest.raises(
        AgentRunNotFoundError,
        match="agent run not found",
    ):
        _query_service().get_owned_run_detail(
            db=db,
            user_id=admin.id,
            agent_run_id=run.id,
        )


def test_get_agent_run_returns_safe_tool_summaries(
    db: Session,
) -> None:
    owner = _create_user(db, email="api-detail-owner@example.com")
    kb = _create_kb(db, owner=owner, name="API Detail KB")
    run = _create_run(
        db,
        user=owner,
        knowledge_base=kb,
        request_id="req-agent-detail",
        tool_call_count=2,
    )
    first = _create_tool_call(
        db,
        run=run,
        provider_call_id="provider-secret-call-1",
    )
    second = _create_tool_call(
        db,
        run=run,
        provider_call_id="provider-secret-call-2",
        status=AgentToolCallStatus.FAILED,
        duration_ms=20,
        error_type="execution_failed",
    )

    response = _build_client(db, owner).get(
        f"/agent/runs/{run.id}"
    )

    assert response.status_code == 200
    body = response.json()
    assert body["id"] == run.id
    assert body["knowledge_base_id"] == kb.id
    assert body["request_id"] == "req-agent-detail"
    assert body["model_provider"] == "test-provider"
    assert body["model_name"] == "test-model"
    assert body["tool_call_count"] == 2
    assert [item["tool_name"] for item in body["tool_calls"]] == [
        first.tool_name,
        second.tool_name,
    ]
    assert body["tool_calls"][1]["status"] == "failed"
    assert body["tool_calls"][1]["error_type"] == "execution_failed"

    # 对外只暴露安全运行摘要，不暴露用户身份与 Provider Tool Call ID。
    assert "user_id" not in body
    assert all(
        "provider_call_id" not in item
        for item in body["tool_calls"]
    )
    assert "provider-secret-call-1" not in response.text


def test_agent_run_api_lists_only_current_user_runs(
    db: Session,
) -> None:
    owner = _create_user(db, email="api-list-owner@example.com")
    other = _create_user(db, email="api-list-other@example.com")
    kb = _create_kb(db, owner=owner, name="API List KB")
    other_kb = _create_kb(db, owner=other, name="API List Other KB")

    first = _create_run(
        db,
        user=owner,
        knowledge_base=kb,
        request_id="api-list-first",
    )
    second = _create_run(
        db,
        user=owner,
        knowledge_base=kb,
        request_id="api-list-second",
    )
    _create_run(
        db,
        user=other,
        knowledge_base=other_kb,
        request_id="api-list-foreign",
    )

    response = _build_client(db, owner).get(
        f"/agent/runs?knowledge_base_id={kb.id}&limit=1"
    )

    assert response.status_code == 200
    body = response.json()
    assert [item["id"] for item in body] == [second.id]
    assert body[0]["knowledge_base_id"] == kb.id
    assert "user_id" not in body[0]
    assert "request_id" not in body[0]
    assert first.id != second.id


def test_agent_run_api_hides_foreign_run_and_unauthorized_kb_filter(
    db: Session,
) -> None:
    owner = _create_user(db, email="api-isolation-owner@example.com")
    other = _create_user(db, email="api-isolation-other@example.com")
    owner_kb = _create_kb(db, owner=owner, name="API Isolation Owner KB")
    other_kb = _create_kb(db, owner=other, name="API Isolation Other KB")
    foreign_run = _create_run(
        db,
        user=other,
        knowledge_base=other_kb,
        request_id="api-isolation-foreign",
    )
    _create_run(
        db,
        user=owner,
        knowledge_base=owner_kb,
        request_id="api-isolation-owner",
    )

    client = _build_client(db, owner)

    detail = client.get(f"/agent/runs/{foreign_run.id}")
    filtered_list = client.get(
        f"/agent/runs?knowledge_base_id={other_kb.id}"
    )

    assert detail.status_code == 404
    assert detail.json() == {"detail": "agent run not found"}
    assert filtered_list.status_code == 404
    assert filtered_list.json() == {"detail": "knowledge base not found"}


def test_agent_run_api_validates_query_bounds(db: Session) -> None:
    owner = _create_user(db, email="api-validation-owner@example.com")
    client = _build_client(db, owner)

    invalid_limit = client.get("/agent/runs?limit=101")
    invalid_kb = client.get("/agent/runs?knowledge_base_id=0")
    invalid_run = client.get("/agent/runs/0")

    assert invalid_limit.status_code == 422
    assert invalid_kb.status_code == 422
    assert invalid_run.status_code == 422
