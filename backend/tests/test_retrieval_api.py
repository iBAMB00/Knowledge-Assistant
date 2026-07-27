import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

import app.api.retrieval as retrieval_api
from app.core.database import get_db
from app.schemas.vector_search_result import (
    VectorSearchResult,
)


class FakeRetrievalService:
    """
    检索调试接口测试使用的服务。
    """

    def __init__(self) -> None:
        self.received_query: str | None = None
        self.received_top_k: int | None = None
        self.received_score_threshold: float | None = None
        self.received_document_id: int | None = None

    def retrieve(
        self,
        db: Session,
        query: str,
        top_k: int | None = None,
        score_threshold: float | None = None,
        document_id: int | None = None,
    ) -> list[VectorSearchResult]:
        """
        返回固定检索结果。
        """

        normalized_query = query.strip()

        if not normalized_query:
            raise ValueError(
                "query cannot be empty"
            )

        self.received_query = normalized_query
        self.received_top_k = top_k
        self.received_score_threshold = score_threshold
        self.received_document_id = document_id

        return [
            VectorSearchResult(
                document_id=1,
                chunk_id=10,
                chunk_index=0,
                content="企业知识库检索结果",
                score=0.92,
            )
        ]


@pytest.fixture
def client(
    db: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[TestClient, FakeRetrievalService]:
    """
    创建只包含检索路由的测试应用。
    """

    fake_service = FakeRetrievalService()

    monkeypatch.setattr(
        retrieval_api,
        "retrieval_service",
        fake_service,
    )

    app = FastAPI()
    app.include_router(retrieval_api.router)

    def override_get_db():
        yield db

    app.dependency_overrides[get_db] = override_get_db

    return TestClient(app), fake_service


def test_debug_retrieval_returns_results(
    client: tuple[
        TestClient,
        FakeRetrievalService,
    ],
) -> None:
    """
    验证检索调试接口返回召回结果。
    """

    test_client, fake_service = client

    response = test_client.post(
        "/knowledge/retrieval/debug",
        json={
            "query": "  企业知识库如何检索？  ",
            "top_k": 3,
            "score_threshold": 0.5,
            "document_id": 1,
        },
    )

    assert response.status_code == 200

    assert response.json() == [
        {
            "document_id": 1,
            "chunk_id": 10,
            "chunk_index": 0,
            "content": "企业知识库检索结果",
            "score": 0.92,
        }
    ]

    assert fake_service.received_query == (
        "企业知识库如何检索？"
    )

    assert fake_service.received_top_k == 3
    assert fake_service.received_score_threshold == 0.5
    assert fake_service.received_document_id == 1


def test_debug_retrieval_rejects_empty_query(
    client: tuple[
        TestClient,
        FakeRetrievalService,
    ],
) -> None:
    """
    验证空查询返回400。
    """

    test_client, _ = client

    response = test_client.post(
        "/knowledge/retrieval/debug",
        json={
            "query": "   ",
        },
    )

    assert response.status_code == 400

    assert response.json() == {
        "detail": "query cannot be empty"
    }


@pytest.mark.parametrize(
    ("payload", "field_name"),
    [
        (
            {
                "query": "测试",
                "top_k": 0,
            },
            "top_k",
        ),
        (
            {
                "query": "测试",
                "score_threshold": 1.1,
            },
            "score_threshold",
        ),
        (
            {
                "query": "测试",
                "document_id": 0,
            },
            "document_id",
        ),
    ],
)
def test_debug_retrieval_rejects_invalid_parameters(
    client: tuple[
        TestClient,
        FakeRetrievalService,
    ],
    payload: dict,
    field_name: str,
) -> None:
    """
    验证请求模型拒绝非法参数。
    """

    test_client, _ = client

    response = test_client.post(
        "/knowledge/retrieval/debug",
        json=payload,
    )

    assert response.status_code == 422

    error_fields = [
        error["loc"][-1]
        for error in response.json()["detail"]
    ]

    assert field_name in error_fields