from collections.abc import Iterator

import pytest
import json
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

import app.api.knowledge_chat as knowledge_chat_api
from app.core.database import get_db
from app.schemas.vector_search_result import (
    VectorSearchResult,
)
from app.schemas.knowledge_chat_response import (
    KnowledgeChatSource,
)
from app.services.knowledge_chat_service import (
    KnowledgeChatPreparation,
    KnowledgeChatService,
)
from app.services.rag.context_builder import (
    ContextBuilder,
)


class FakeRetrievalService:
    """
    流式知识库问答测试使用的检索服务。
    """

    def __init__(
        self,
        results: list[VectorSearchResult],
    ) -> None:
        self.results = results

    def retrieve(
        self,
        db: Session,
        query: str,
        top_k: int | None = None,
        score_threshold: float | None = None,
        document_id: int | None = None,
    ) -> list[VectorSearchResult]:
        """
        返回预设检索结果。
        """

        return self.results


class FakeStreamingLLMService:
    """
    流式知识库问答测试使用的LLM服务。
    """

    def __init__(
        self,
        contents: list[str],
    ) -> None:
        self.contents = contents
        self.received_message: str | None = None
        self.stream_call_count = 0

    def chat(self, message: str) -> str:
        """
        当前测试不使用非流式调用。
        """

        return "".join(self.contents)

    def stream_chat(
        self,
        message: str,
    ) -> Iterator[str]:
        """
        返回预设流式内容。
        """

        self.received_message = message
        self.stream_call_count += 1

        yield from self.contents


class FakeKnowledgeChatService:
    """
    SSE接口测试使用的知识库问答服务。
    """

    def __init__(self) -> None:
        self.received_question: str | None = None

    def prepare(
        self,
        db: Session,
        question: str,
        top_k: int | None = None,
        score_threshold: float | None = None,
        document_id: int | None = None,
    ) -> KnowledgeChatPreparation:
        """
        返回固定问答准备结果。
        """

        normalized_question = question.strip()

        if not normalized_question:
            raise ValueError(
                "question cannot be empty"
            )

        self.received_question = normalized_question

        return KnowledgeChatPreparation(
            prompt="测试Prompt",
            sources=[
                KnowledgeChatSource(
                    source_number=1,
                    document_id=1,
                    excerpt="管理员可以重置密码。",
                )
            ],
        )

    def stream_chat(
        self,
        preparation: KnowledgeChatPreparation,
    ) -> Iterator[str]:
        """
        返回固定流式内容。
        """

        yield "管理员"
        yield "可以重置密码。[来源 1]"


def build_search_result() -> VectorSearchResult:
    """
    创建测试使用的检索结果。
    """

    return VectorSearchResult(
        document_id=1,
        chunk_id=10,
        chunk_index=0,
        content="管理员可以在系统设置中重置密码。",
        score=0.95,
    )


def test_stream_service_returns_model_contents(
    db: Session,
) -> None:
    """
    验证知识库服务返回模型流式片段。
    """

    llm_service = FakeStreamingLLMService(
        contents=[
            "管理员",
            "可以重置密码。[来源 1]",
        ]
    )

    service = KnowledgeChatService(
        retrieval_service=FakeRetrievalService(
            results=[
                build_search_result(),
            ]
        ),
        context_builder=ContextBuilder(),
        llm_service=llm_service,
    )

    preparation = service.prepare(
        db=db,
        question="如何重置密码？",
    )

    contents = list(
        service.stream_chat(preparation)
    )

    assert contents == [
        "管理员",
        "可以重置密码。[来源 1]",
    ]

    assert len(preparation.sources) == 1

    source = preparation.sources[0]

    assert source.source_number == 1
    assert source.document_id == 1
    assert source.excerpt == (
        "管理员可以在系统设置中重置密码。"
    )

    assert llm_service.stream_call_count == 1
    assert llm_service.received_message is not None
    assert (
        "如何重置密码？"
        in llm_service.received_message
    )
    assert (
        "[来源 1]"
        in llm_service.received_message
    )


def test_stream_service_skips_llm_without_context(
    db: Session,
) -> None:
    """
    验证无可靠知识时直接返回固定回答。
    """

    llm_service = FakeStreamingLLMService(
        contents=[]
    )

    service = KnowledgeChatService(
        retrieval_service=FakeRetrievalService(
            results=[]
        ),
        context_builder=ContextBuilder(),
        llm_service=llm_service,
    )

    preparation = service.prepare(
        db=db,
        question="知识库中不存在的问题",
    )

    contents = list(
        service.stream_chat(preparation)
    )

    assert contents == [
        KnowledgeChatService.NO_RELIABLE_ANSWER
    ]

    assert preparation.sources == []
    assert llm_service.stream_call_count == 0


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

    app = FastAPI()
    app.include_router(
        knowledge_chat_api.router
    )

    def override_get_db():
        yield db

    app.dependency_overrides[get_db] = (
        override_get_db
    )

    return TestClient(app), fake_service


def test_stream_api_returns_sse_events(
    client: tuple[
        TestClient,
        FakeKnowledgeChatService,
    ],
) -> None:
    """
    验证知识库流式接口返回完整SSE事件。
    """

    test_client, fake_service = client

    response = test_client.post(
        "/knowledge/chat/stream",
        json={
            "question": "  如何重置密码？  ",
            "top_k": 5,
            "score_threshold": 0.6,
            "document_id": 1,
        },
    )

    assert response.status_code == 200
    assert response.headers[
        "content-type"
    ].startswith(
        "text/event-stream"
    )

    body = response.text

    metadata_position = body.index(
        "event: metadata"
    )
    first_message_position = body.index(
        '"content": "管理员"'
    )
    second_message_position = body.index(
        '"content": "可以重置密码。[来源 1]"'
    )
    done_position = body.index(
        "event: done"
    )

    assert (
        metadata_position
        < first_message_position
        < second_message_position
        < done_position
    )

    metadata_event = body.split("\n\n", maxsplit=1)[0]

    metadata_data_line = next(
        line
        for line in metadata_event.splitlines()
        if line.startswith("data: ")
    )

    metadata_payload = json.loads(
        metadata_data_line.removeprefix("data: ")
    )

    assert metadata_payload == {
        "sources": [
            {
                "source_number": 1,
                "document_id": 1,
                "excerpt": "管理员可以重置密码。",
            }
        ]
    }

    source = metadata_payload["sources"][0]

    assert "chunk_id" not in source
    assert "chunk_index" not in source
    assert "score" not in source
    assert "content" not in source

    assert fake_service.received_question == (
        "如何重置密码？"
    )
    assert fake_service.received_question == (
        "如何重置密码？"
    )


def test_stream_api_rejects_empty_question(
    client: tuple[
        TestClient,
        FakeKnowledgeChatService,
    ],
) -> None:
    """
    验证流式接口拒绝空问题。
    """

    test_client, _ = client

    response = test_client.post(
        "/knowledge/chat/stream",
        json={
            "question": "   ",
        },
    )

    assert response.status_code == 400

    assert response.json() == {
        "detail": "question cannot be empty"
    }