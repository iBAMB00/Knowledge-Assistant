import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

import app.api.knowledge_chat as knowledge_chat_api
from app.api.dependencies.auth import get_current_user
from app.core.database import get_db
from app.models.database.user import User
from app.schemas.knowledge_chat_response import (
    KnowledgeChatResponse,
    KnowledgeChatSource,
)


class FakeKnowledgeChatService:
    """
    知识库问答接口测试使用的服务。
    """

    def __init__(self) -> None:
        self.received_question: str | None = None
        self.received_top_k: int | None = None
        self.received_score_threshold: float | None = None
        self.received_document_id: int | None = None
        self.received_knowledge_base_id: int | None = None

    def chat(
        self,
        db: Session,
        question: str,
        top_k: int | None = None,
        score_threshold: float | None = None,
        document_id: int | None = None,
        knowledge_base_id: int | None = None,
    ) -> KnowledgeChatResponse:
        """
        返回固定知识库问答结果。
        """

        normalized_question = question.strip()

        if not normalized_question:
            raise ValueError(
                "question cannot be empty"
            )

        self.received_question = normalized_question
        self.received_top_k = top_k
        self.received_score_threshold = score_threshold
        self.received_document_id = document_id
        self.received_knowledge_base_id = knowledge_base_id

        return KnowledgeChatResponse(
            answer=(
                "管理员可以在系统设置中"
                "重置用户密码。[来源 1]"
            ),
            sources=[
                KnowledgeChatSource(
                    source_number=1,
                    document_id=1,
                    filename="test-document.txt",
                    excerpt=(
                        "管理员可以在系统设置中"
                        "重置用户密码。"
                    ),
                )
            ],
        )


@pytest.fixture
def client(
    db: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[
    TestClient,
    FakeKnowledgeChatService,
]:
    """
    创建只包含知识库问答路由的测试应用。
    """

    fake_service = FakeKnowledgeChatService()

    monkeypatch.setattr(
        knowledge_chat_api,
        "knowledge_chat_service",
        fake_service,
    )

    class FakeAccessPolicy:
        def get_accessible_knowledge_base(self, **kwargs):
            return None

        def ensure_document_in_knowledge_base(self, **kwargs):
            return None

    monkeypatch.setattr(
        knowledge_chat_api,
        "knowledge_base_access_policy",
        FakeAccessPolicy(),
    )

    app = FastAPI()
    app.include_router(
        knowledge_chat_api.router
    )

    def override_get_db():
        yield db

    app.dependency_overrides[
        get_db
    ] = override_get_db
    app.dependency_overrides[get_current_user] = lambda: User(
        id=1,
        email="tester@example.com",
        password_hash="hash",
        role="user",
        is_active=True,
    )

    return TestClient(app), fake_service


def test_knowledge_chat_returns_answer_and_sources(
    client: tuple[
        TestClient,
        FakeKnowledgeChatService,
    ],
) -> None:
    """
    验证接口返回回答和公开来源。
    """

    test_client, fake_service = client

    response = test_client.post(
        "/knowledge/chat",
        json={
            "question": "  如何重置用户密码？  ",
            "knowledge_base_id": 1,
            "top_k": 5,
            "document_id": 1,
        },
    )

    assert response.status_code == 200

    assert response.json() == {
        "answer": (
            "管理员可以在系统设置中"
            "重置用户密码。[来源 1]"
        ),
        "sources": [
            {
                "source_number": 1,
                "document_id": 1,
                "filename": "test-document.txt",
                "excerpt": (
                    "管理员可以在系统设置中"
                    "重置用户密码。"
                ),
            }
        ],
    }

    assert fake_service.received_question == (
        "如何重置用户密码？"
    )

    assert fake_service.received_top_k == 5

    assert (
        fake_service.received_score_threshold
        is None
    )

    assert (
        fake_service.received_document_id
        == 1
    )
    assert fake_service.received_knowledge_base_id == 1

def test_knowledge_chat_uses_config_defaults_when_optional_parameters_omitted(
    client: tuple[
        TestClient,
        FakeKnowledgeChatService,
    ],
) -> None:
    """
    验证请求未提供可选检索参数时，
    Router 将 None 传给 Service。
    """

    test_client, fake_service = client

    response = test_client.post(
        "/knowledge/chat",
        json={
            "question": "如何重置用户密码？",
            "knowledge_base_id": 1,
        },
    )

    assert response.status_code == 200

    assert fake_service.received_top_k is None
    assert (
        fake_service.received_score_threshold
        is None
    )
    assert fake_service.received_document_id is None

def test_knowledge_chat_response_hides_internal_fields(
    client: tuple[
        TestClient,
        FakeKnowledgeChatService,
    ],
) -> None:
    """
    验证接口响应不暴露内部检索字段。
    """

    test_client, _ = client

    response = test_client.post(
        "/knowledge/chat",
        json={
            "question": "测试问题",
            "knowledge_base_id": 1,
        },
    )

    assert response.status_code == 200

    source = response.json()["sources"][0]

    assert "chunk_id" not in source
    assert "chunk_index" not in source
    assert "score" not in source
    assert "content" not in source


def test_knowledge_chat_rejects_empty_question(
    client: tuple[
        TestClient,
        FakeKnowledgeChatService,
    ],
) -> None:
    """
    验证空问题返回 400。
    """

    test_client, _ = client

    response = test_client.post(
        "/knowledge/chat",
        json={
            "question": "   ",
            "knowledge_base_id": 1,
        },
    )

    assert response.status_code == 400

    assert response.json() == {
        "detail": "question cannot be empty"
    }


@pytest.mark.parametrize(
    ("payload", "field_name"),
    [
        (
            {
                "question": "测试问题",
                "top_k": 0,
            },
            "top_k",
        ),
        (
            {
                "question": "测试问题",
                "score_threshold": 0.1,
            },
            "score_threshold",
        ),
        (
            {
                "question": "测试问题",
                "document_id": 0,
            },
            "document_id",
        ),
    ],
)
def test_knowledge_chat_rejects_invalid_parameters(
    client: tuple[
        TestClient,
        FakeKnowledgeChatService,
    ],
    payload: dict,
    field_name: str,
) -> None:
    """
    验证请求模型拒绝非法参数。
    """

    test_client, _ = client

    response = test_client.post(
        "/knowledge/chat",
        json={"knowledge_base_id": 1, **payload},
    )

    assert response.status_code == 422

    error_fields = [
        error["loc"][-1]
        for error in response.json()["detail"]
    ]

    assert field_name in error_fields