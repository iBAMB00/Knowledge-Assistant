import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.dependencies.auth import get_current_user
from app.api.knowledge_base import router as knowledge_base_router
from app.constants.user_role import UserRole
from app.core.database import get_db
from app.models.database.document import Document
from app.models.database.user import User
from app.repositories.document_repository import DocumentRepository
from app.repositories.knowledge_base_repository import KnowledgeBaseRepository
from app.services.knowledge_base_access_policy import (
    KnowledgeBaseAccessPolicy,
    ResourceAccessNotFoundError,
)
from app.services.knowledge_base_service import (
    KnowledgeBaseConflictError,
    KnowledgeBaseService,
)


def _create_user(db, email: str, role: str = UserRole.USER.value) -> User:
    user = User(
        email=email,
        password_hash="test-password-hash",
        role=role,
        is_active=True,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def _build_services() -> tuple[KnowledgeBaseService, KnowledgeBaseAccessPolicy]:
    knowledge_base_repository = KnowledgeBaseRepository()
    document_repository = DocumentRepository()
    policy = KnowledgeBaseAccessPolicy(
        knowledge_base_repository=knowledge_base_repository,
        document_repository=document_repository,
    )
    service = KnowledgeBaseService(
        knowledge_base_repository=knowledge_base_repository,
        document_repository=document_repository,
        access_policy=policy,
    )
    return service, policy


def test_owner_isolation_and_admin_access(db) -> None:
    service, policy = _build_services()
    owner = _create_user(db, "owner@example.com")
    other = _create_user(db, "other@example.com")
    admin = _create_user(db, "admin@example.com", UserRole.ADMIN.value)

    knowledge_base = service.create(
        db=db,
        user=owner,
        name="研发知识库",
        description="内部研发资料",
    )

    assert knowledge_base.owner_id == owner.id
    assert [item.id for item in service.list_accessible(db, owner)] == [
        knowledge_base.id
    ]
    assert service.list_accessible(db, other) == []

    with pytest.raises(
        ResourceAccessNotFoundError,
        match="knowledge base not found",
    ):
        policy.get_accessible_knowledge_base(
            db=db,
            knowledge_base_id=knowledge_base.id,
            user=other,
        )

    assert (
        policy.get_accessible_knowledge_base(
            db=db,
            knowledge_base_id=knowledge_base.id,
            user=admin,
        ).id
        == knowledge_base.id
    )


def test_document_access_follows_knowledge_base_owner(db) -> None:
    service, policy = _build_services()
    owner = _create_user(db, "doc-owner@example.com")
    other = _create_user(db, "doc-other@example.com")
    admin = _create_user(db, "doc-admin@example.com", UserRole.ADMIN.value)
    knowledge_base = service.create(db, owner, "运维知识库")

    document = Document(
        knowledge_base_id=knowledge_base.id,
        filename="runbook.txt",
        storage_key="runbook-stored.txt",
        stored_name="runbook-stored.txt",
        path="uploads/runbook-stored.txt",
        size=100,
        status="uploaded",
    )
    db.add(document)
    db.commit()
    db.refresh(document)

    assert policy.get_accessible_document(db, document.id, owner).id == document.id

    with pytest.raises(
        ResourceAccessNotFoundError,
        match="document not found|knowledge base not found",
    ):
        policy.get_accessible_document(db, document.id, other)

    assert policy.get_accessible_document(db, document.id, admin).id == document.id


def test_legacy_unowned_document_is_not_user_accessible(db) -> None:
    _, policy = _build_services()
    user = _create_user(db, "legacy@example.com")

    document = Document(
        knowledge_base_id=None,
        filename="legacy.txt",
        storage_key="legacy-stored.txt",
        stored_name="legacy-stored.txt",
        path="uploads/legacy-stored.txt",
        size=10,
        status="uploaded",
    )
    db.add(document)
    db.commit()
    db.refresh(document)

    with pytest.raises(
        ResourceAccessNotFoundError,
        match="document not found",
    ):
        policy.get_accessible_document(db, document.id, user)


def test_non_empty_knowledge_base_cannot_be_deleted(db) -> None:
    service, _ = _build_services()
    owner = _create_user(db, "delete-owner@example.com")
    knowledge_base = service.create(db, owner, "不可删除")

    db.add(
        Document(
            knowledge_base_id=knowledge_base.id,
            filename="a.txt",
            storage_key="a-stored.txt",
            stored_name="a-stored.txt",
            path="uploads/a-stored.txt",
            size=1,
            status="uploaded",
        )
    )
    db.commit()

    with pytest.raises(
        KnowledgeBaseConflictError,
        match="must be empty",
    ):
        service.delete(db, owner, knowledge_base.id)


def test_knowledge_base_api_requires_auth_and_scopes_to_current_user(db) -> None:
    owner = _create_user(db, "api-owner@example.com")
    other = _create_user(db, "api-other@example.com")

    app = FastAPI()
    app.include_router(knowledge_base_router)

    def override_get_db():
        yield db

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)

    unauthenticated = client.get("/knowledge-bases/")
    assert unauthenticated.status_code == 401

    app.dependency_overrides[get_current_user] = lambda: owner
    created = client.post(
        "/knowledge-bases/",
        json={"name": "API KB", "description": "owner only"},
    )
    assert created.status_code == 201
    knowledge_base_id = created.json()["id"]
    assert created.json()["owner_id"] == owner.id

    app.dependency_overrides[get_current_user] = lambda: other
    hidden = client.get(f"/knowledge-bases/{knowledge_base_id}")
    assert hidden.status_code == 404
    assert client.get("/knowledge-bases/").json() == []


def test_sql_retrieval_candidate_queries_are_knowledge_base_scoped(db) -> None:
    """Dense SQL候选与BM25候选必须使用相同的KnowledgeBase过滤。"""

    from app.constants.embedding_status import EmbeddingStatus
    from app.models.database.chunk_embedding import ChunkEmbedding
    from app.models.database.document_chunk import DocumentChunk
    from app.models.database.document_content import DocumentContent
    from app.repositories.chunk_embedding_repository import ChunkEmbeddingRepository
    from app.repositories.document_chunk_repository import DocumentChunkRepository

    owner = _create_user(db, "sql-scope@example.com")
    service, _ = _build_services()
    kb_one = service.create(db, owner, "SQL Scope 1")
    kb_two = service.create(db, owner, "SQL Scope 2")

    expected_document_id = None
    for kb, filename in [(kb_one, "one.txt"), (kb_two, "two.txt")]:
        document = Document(
            knowledge_base_id=kb.id,
            filename=filename,
            storage_key=f"stored-{filename}",
            stored_name=f"stored-{filename}",
            path=f"uploads/{filename}",
            size=10,
            status="completed",
        )
        db.add(document)
        db.flush()
        if kb.id == kb_one.id:
            expected_document_id = document.id
        content = DocumentContent(
            document_id=document.id,
            content="alphaomega",
            parser_type="txt",
            parser_version="1.0",
        )
        db.add(content)
        db.flush()
        chunk = DocumentChunk(
            document_content_id=content.id,
            chunk_index=0,
            content="alphaomega",
            token_count=1,
            chunk_strategy="recursive_character",
            embedding_status=EmbeddingStatus.COMPLETED.value,
        )
        db.add(chunk)
        db.flush()
        db.add(
            ChunkEmbedding(
                document_chunk_id=chunk.id,
                vector=[1.0, 0.0],
                embedding_model="test-model",
                embedding_dimension=2,
            )
        )
    db.commit()

    dense_rows = ChunkEmbeddingRepository().find_search_candidates(
        db=db,
        embedding_model="test-model",
        knowledge_base_id=kb_one.id,
    )
    chunk_repository = DocumentChunkRepository()
    bm25_rows = chunk_repository.find_retrieval_candidates(
        db=db,
        knowledge_base_id=kb_one.id,
    )
    all_chunk_ids = [
        chunk.id
        for chunk, _, _ in chunk_repository.find_retrieval_candidates(db=db)
    ]
    scoped_parent_rows = chunk_repository.find_by_ids(
        db=db,
        chunk_ids=all_chunk_ids,
        knowledge_base_id=kb_one.id,
    )

    assert len(dense_rows) == 1
    assert dense_rows[0][2].document_id == expected_document_id
    assert len(bm25_rows) == 1
    assert bm25_rows[0][1].document_id == expected_document_id
    assert len(scoped_parent_rows) == 1
    assert scoped_parent_rows[0].id == bm25_rows[0][0].id
