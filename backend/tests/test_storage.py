from io import BytesIO
from types import SimpleNamespace

import pytest

from app.services.storage.factory import create_storage_service
from app.services.storage.local import LocalStorageProvider
from app.services.storage.minio import MinioStorageProvider
from app.services.storage_service import StorageService


class FakeMinioError(Exception):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class FakeMinioResponse(BytesIO):
    def release_conn(self) -> None:
        pass


class FakeMinioClient:
    """只实现 Provider 测试需要的最小 MinIO Client 契约。"""

    def __init__(self) -> None:
        self.buckets: set[str] = set()
        self.objects: dict[tuple[str, str], bytes] = {}

    def bucket_exists(self, bucket_name: str) -> bool:
        return bucket_name in self.buckets

    def make_bucket(self, bucket_name: str) -> None:
        self.buckets.add(bucket_name)

    def put_object(
        self,
        bucket_name: str,
        object_name: str,
        data,
        length: int,
    ) -> None:
        self.objects[(bucket_name, object_name)] = data.read(length)

    def get_object(self, bucket_name: str, object_name: str):
        try:
            content = self.objects[(bucket_name, object_name)]
        except KeyError as exc:
            raise FakeMinioError("NoSuchKey") from exc
        return FakeMinioResponse(content)

    def remove_object(self, bucket_name: str, object_name: str) -> None:
        self.objects.pop((bucket_name, object_name), None)

    def stat_object(self, bucket_name: str, object_name: str) -> object:
        if (bucket_name, object_name) not in self.objects:
            raise FakeMinioError("NoSuchKey")
        return object()


def test_local_storage_uses_stable_storage_key(tmp_path) -> None:
    service = StorageService(
        provider=LocalStorageProvider(str(tmp_path))
    )

    result = service.save(
        filename="guide.md",
        content=b"knowledge",
        knowledge_base_id=12,
    )

    assert result.storage_key.startswith("knowledge-bases/12/")
    assert result.storage_key.endswith("_guide.md")
    assert service.read(result.storage_key) == b"knowledge"
    assert service.exists(result.storage_key) is True

    service.delete(result.storage_key)

    assert service.exists(result.storage_key) is False


def test_local_storage_rejects_path_traversal(tmp_path) -> None:
    provider = LocalStorageProvider(str(tmp_path))

    with pytest.raises(ValueError, match="invalid storage key"):
        provider.save("../outside.txt", b"secret")


def test_minio_provider_matches_storage_contract() -> None:
    client = FakeMinioClient()
    provider = MinioStorageProvider(
        endpoint="minio:9000",
        access_key="access",
        secret_key="secret",
        bucket_name="knowledge-assistant",
        client=client,
    )
    service = StorageService(provider=provider)

    result = service.save(
        filename="manual.pdf",
        content=b"pdf-bytes",
        knowledge_base_id=7,
    )

    assert "knowledge-assistant" in client.buckets
    assert result.storage_key.startswith("knowledge-bases/7/")
    assert service.exists(result.storage_key) is True
    assert service.read(result.storage_key) == b"pdf-bytes"

    service.delete(result.storage_key)

    assert service.exists(result.storage_key) is False


def test_storage_factory_can_build_local_provider(tmp_path) -> None:
    settings = SimpleNamespace(
        storage_backend="local",
        local_storage_dir=str(tmp_path),
    )

    service = create_storage_service(settings=settings)  # type: ignore[arg-type]
    result = service.save("factory.txt", b"ok", knowledge_base_id=1)

    assert service.read(result.storage_key) == b"ok"


class FailingDocumentRepository:
    def create(self, db, document):
        del db, document
        raise RuntimeError("database write failed")


class DummyDb:
    def __init__(self) -> None:
        self.rollback_count = 0

    def rollback(self) -> None:
        self.rollback_count += 1

    def commit(self) -> None:
        raise AssertionError("commit must not run after repository failure")


def test_document_upload_compensates_object_when_database_write_fails(tmp_path) -> None:
    from app.services.document_service import DocumentService

    storage_service = StorageService(storage_dir=str(tmp_path))
    service = DocumentService(
        storage_service=storage_service,
        document_repository=FailingDocumentRepository(),  # type: ignore[arg-type]
        document_content_repository=object(),  # type: ignore[arg-type]
        document_chunk_repository=object(),  # type: ignore[arg-type]
        processing_job_repository=object(),  # type: ignore[arg-type]
        document_operation_policy=object(),  # type: ignore[arg-type]
    )
    db = DummyDb()

    with pytest.raises(RuntimeError, match="database write failed"):
        service.upload_document(
            db=db,  # type: ignore[arg-type]
            filename="failed.txt",
            content=b"must be compensated",
            knowledge_base_id=9,
        )

    assert db.rollback_count == 1
    assert list(tmp_path.rglob("*")) == []


def test_public_document_response_does_not_expose_storage_implementation() -> None:
    from app.schemas.document_response import DocumentResponse

    assert "stored_name" not in DocumentResponse.model_fields
    assert "path" not in DocumentResponse.model_fields
    assert "storage_key" not in DocumentResponse.model_fields


class MemoryStorageProvider:
    def __init__(self, objects: dict[str, bytes] | None = None) -> None:
        self.objects = dict(objects or {})

    def save(self, storage_key: str, content: bytes) -> None:
        self.objects[storage_key] = content

    def read(self, storage_key: str) -> bytes:
        return self.objects[storage_key]

    def delete(self, storage_key: str) -> None:
        self.objects.pop(storage_key, None)

    def exists(self, storage_key: str) -> bool:
        return storage_key in self.objects


def test_local_to_minio_migration_is_idempotent(db) -> None:
    from app.models.database.document import Document
    from scripts.migrate_local_storage_to_minio import migrate_documents

    db.add_all([
        Document(
            filename="one.txt",
            storage_key="legacy/one.txt",
            size=3,
            status="uploaded",
        ),
        Document(
            filename="two.txt",
            storage_key="legacy/two.txt",
            size=3,
            status="uploaded",
        ),
    ])
    db.flush()

    source = MemoryStorageProvider({
        "legacy/one.txt": b"one",
        "legacy/two.txt": b"two",
    })
    target = MemoryStorageProvider({
        "legacy/one.txt": b"one",
    })

    first = migrate_documents(db, source, target)
    second = migrate_documents(db, source, target)

    assert first.total == 2
    assert first.copied == 1
    assert first.skipped == 1
    assert first.missing_local == 0
    assert target.objects["legacy/two.txt"] == b"two"

    assert second.copied == 0
    assert second.skipped == 2
