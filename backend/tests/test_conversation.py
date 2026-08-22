import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.conversation import router as conversation_router
from app.api.dependencies.auth import get_current_user
from app.constants.conversation_message_role import ConversationMessageRole
from app.constants.conversation_mode import ConversationMode
from app.core.database import get_db
from app.models.database.conversation_message import ConversationMessage
from app.models.database.user import User
from app.repositories.conversation_message_repository import (
    ConversationMessageRepository,
)
from app.repositories.conversation_repository import ConversationRepository
from app.repositories.document_repository import DocumentRepository
from app.repositories.knowledge_base_repository import KnowledgeBaseRepository
from app.services.conversation_service import (
    ConversationNotFoundError,
    ConversationService,
)
from app.services.knowledge_base_access_policy import KnowledgeBaseAccessPolicy
from app.services.knowledge_base_service import (
    KnowledgeBaseConflictError,
    KnowledgeBaseService,
)


def _create_user(db, email: str) -> User:
    user = User(
        email=email,
        password_hash="test-password-hash",
        role="user",
        is_active=True,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def _build_services() -> tuple[ConversationService, KnowledgeBaseService]:
    knowledge_base_repository = KnowledgeBaseRepository()
    document_repository = DocumentRepository()
    access_policy = KnowledgeBaseAccessPolicy(
        knowledge_base_repository=knowledge_base_repository,
        document_repository=document_repository,
    )
    knowledge_base_service = KnowledgeBaseService(
        knowledge_base_repository=knowledge_base_repository,
        document_repository=document_repository,
        access_policy=access_policy,
    )
    conversation_service = ConversationService(
        conversation_repository=ConversationRepository(),
        message_repository=ConversationMessageRepository(),
        access_policy=access_policy,
    )
    return conversation_service, knowledge_base_service


def test_conversation_is_owned_by_user_and_fixed_to_mode_and_kb(db) -> None:
    conversation_service, knowledge_base_service = _build_services()
    owner = _create_user(db, "conversation-owner@example.com")
    other = _create_user(db, "conversation-other@example.com")
    kb = knowledge_base_service.create(db, owner, "Agent KB")

    conversation = conversation_service.create(
        db=db,
        user=owner,
        mode=ConversationMode.AGENT,
        knowledge_base_id=kb.id,
    )

    assert conversation.user_id == owner.id
    assert conversation.mode == ConversationMode.AGENT.value
    assert conversation.knowledge_base_id == kb.id
    assert conversation.title is None
    assert [
        item.id
        for item in conversation_service.list_owned(
            db=db,
            user_id=owner.id,
        )
    ] == [conversation.id]
    assert conversation_service.list_owned(
        db=db,
        user_id=other.id,
    ) == []

    with pytest.raises(
        ConversationNotFoundError,
        match="conversation not found",
    ):
        conversation_service.get_owned(
            db=db,
            user_id=other.id,
            conversation_id=conversation.id,
        )


def test_append_message_auto_titles_and_history_keeps_chronological_order(db) -> None:
    conversation_service, knowledge_base_service = _build_services()
    owner = _create_user(db, "conversation-message@example.com")
    kb = knowledge_base_service.create(db, owner, "History KB")
    conversation = conversation_service.create(
        db=db,
        user=owner,
        mode=ConversationMode.RAG,
        knowledge_base_id=kb.id,
    )

    first_content = "蓝鲸计划什么时候需要回滚？"
    conversation_service.append_message(
        db=db,
        user_id=owner.id,
        conversation_id=conversation.id,
        role=ConversationMessageRole.USER,
        content=first_content,
    )
    conversation_service.append_message(
        db=db,
        user_id=owner.id,
        conversation_id=conversation.id,
        role=ConversationMessageRole.ASSISTANT,
        content="发布后十分钟内错误率超过 5% 时立即回滚。",
    )

    messages = conversation_service.list_messages(
        db=db,
        user_id=owner.id,
        conversation_id=conversation.id,
    )
    refreshed_conversation = conversation_service.get_owned(
        db=db,
        user_id=owner.id,
        conversation_id=conversation.id,
    )

    assert [item.role for item in messages] == [
        ConversationMessageRole.USER.value,
        ConversationMessageRole.ASSISTANT.value,
    ]
    assert [item.content for item in messages] == [
        first_content,
        "发布后十分钟内错误率超过 5% 时立即回滚。",
    ]
    assert refreshed_conversation.title == first_content


def test_conversation_list_supports_mode_and_knowledge_base_filters(db) -> None:
    conversation_service, knowledge_base_service = _build_services()
    owner = _create_user(db, "conversation-filter@example.com")
    kb_one = knowledge_base_service.create(db, owner, "KB One")
    kb_two = knowledge_base_service.create(db, owner, "KB Two")

    rag = conversation_service.create(
        db=db,
        user=owner,
        mode=ConversationMode.RAG,
        knowledge_base_id=kb_one.id,
        title="RAG",
    )
    agent = conversation_service.create(
        db=db,
        user=owner,
        mode=ConversationMode.AGENT,
        knowledge_base_id=kb_two.id,
        title="Agent",
    )

    assert [
        item.id
        for item in conversation_service.list_owned(
            db=db,
            user_id=owner.id,
            mode=ConversationMode.RAG,
        )
    ] == [rag.id]
    assert [
        item.id
        for item in conversation_service.list_owned(
            db=db,
            user_id=owner.id,
            knowledge_base_id=kb_two.id,
        )
    ] == [agent.id]


def test_knowledge_base_with_conversation_cannot_be_deleted(db) -> None:
    conversation_service, knowledge_base_service = _build_services()
    owner = _create_user(db, "conversation-kb-delete@example.com")
    kb = knowledge_base_service.create(db, owner, "Referenced KB")
    conversation_service.create(
        db=db,
        user=owner,
        mode=ConversationMode.AGENT,
        knowledge_base_id=kb.id,
    )

    with pytest.raises(
        KnowledgeBaseConflictError,
        match="still in use",
    ):
        knowledge_base_service.delete(db, owner, kb.id)


def test_conversation_api_requires_auth_and_hides_other_users_history(db) -> None:
    conversation_service, knowledge_base_service = _build_services()
    owner = _create_user(db, "conversation-api-owner@example.com")
    other = _create_user(db, "conversation-api-other@example.com")
    kb = knowledge_base_service.create(db, owner, "API Conversation KB")

    app = FastAPI()
    app.include_router(conversation_router)

    def override_get_db():
        yield db

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)

    assert client.get("/conversations/").status_code == 401

    app.dependency_overrides[get_current_user] = lambda: owner
    created = client.post(
        "/conversations/",
        json={
            "mode": "agent",
            "knowledge_base_id": kb.id,
            "title": "MCP 联调",
        },
    )
    assert created.status_code == 201
    conversation_id = created.json()["id"]
    assert created.json()["mode"] == "agent"
    assert created.json()["knowledge_base_id"] == kb.id
    assert created.json()["title"] == "MCP 联调"

    conversation_service.append_message(
        db=db,
        user_id=owner.id,
        conversation_id=conversation_id,
        role=ConversationMessageRole.USER,
        content="测试 MCP Tool",
    )
    conversation_service.append_message(
        db=db,
        user_id=owner.id,
        conversation_id=conversation_id,
        role=ConversationMessageRole.ASSISTANT,
        content="MCP Tool 调用成功",
    )

    listed = client.get("/conversations/?mode=agent")
    history = client.get(
        f"/conversations/{conversation_id}/messages"
    )
    assert listed.status_code == 200
    assert [item["id"] for item in listed.json()] == [conversation_id]
    assert history.status_code == 200
    assert [item["role"] for item in history.json()] == [
        "user",
        "assistant",
    ]

    app.dependency_overrides[get_current_user] = lambda: other
    assert client.get(f"/conversations/{conversation_id}").status_code == 404
    assert (
        client.get(
            f"/conversations/{conversation_id}/messages"
        ).status_code
        == 404
    )
    assert client.get("/conversations/").json() == []


def test_delete_conversation_cascades_messages_and_hides_other_users(db) -> None:
    conversation_service, knowledge_base_service = _build_services()
    owner = _create_user(db, "conversation-delete-owner@example.com")
    other = _create_user(db, "conversation-delete-other@example.com")
    kb = knowledge_base_service.create(db, owner, "Delete Conversation KB")
    conversation = conversation_service.create(
        db=db,
        user=owner,
        mode=ConversationMode.AGENT,
        knowledge_base_id=kb.id,
    )
    message = conversation_service.append_message(
        db=db,
        user_id=owner.id,
        conversation_id=conversation.id,
        role=ConversationMessageRole.USER,
        content="这条消息会随对话删除。",
    )

    with pytest.raises(
        ConversationNotFoundError,
        match="conversation not found",
    ):
        conversation_service.delete(
            db=db,
            user_id=other.id,
            conversation_id=conversation.id,
        )

    message_id = message.id
    assert db.get(ConversationMessage, message_id) is not None

    conversation_service.delete(
        db=db,
        user_id=owner.id,
        conversation_id=conversation.id,
    )

    with pytest.raises(
        ConversationNotFoundError,
        match="conversation not found",
    ):
        conversation_service.get_owned(
            db=db,
            user_id=owner.id,
            conversation_id=conversation.id,
        )
    assert db.get(ConversationMessage, message_id) is None


def test_conversation_delete_api_returns_204_and_enforces_ownership(db) -> None:
    conversation_service, knowledge_base_service = _build_services()
    owner = _create_user(db, "conversation-delete-api-owner@example.com")
    other = _create_user(db, "conversation-delete-api-other@example.com")
    kb = knowledge_base_service.create(db, owner, "Delete API KB")
    conversation = conversation_service.create(
        db=db,
        user=owner,
        mode=ConversationMode.RAG,
        knowledge_base_id=kb.id,
    )

    app = FastAPI()
    app.include_router(conversation_router)

    def override_get_db():
        yield db

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_current_user] = lambda: other
    client = TestClient(app)

    assert client.delete(
        f"/conversations/{conversation.id}"
    ).status_code == 404

    app.dependency_overrides[get_current_user] = lambda: owner
    response = client.delete(f"/conversations/{conversation.id}")
    assert response.status_code == 204
    assert response.content == b""
    assert client.get(
        f"/conversations/{conversation.id}"
    ).status_code == 404
