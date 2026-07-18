from app.services.document_service import DocumentService
from app.services.storage_service import StorageService
from app.repositories.document_repository import DocumentRepository
from pathlib import Path

from app.core.database import SessionLocal

def test_create_upload_dir():
    storage_service = StorageService()
    document_repository = DocumentRepository()
    service = DocumentService(storage_service, document_repository)

    print(service.storage_service.storage_dir)

    assert service.storage_service.storage_dir.exists()

def test_upload_document(tmp_path: Path):
    """
    测试文档上传功能。
    """

    storage_service = StorageService(storage_dir=str(tmp_path))
    document_repository = DocumentRepository()
    service = DocumentService(storage_service, document_repository)

    filename = "员工手册.pdf"
    content = b"Hello Secure Assistant"

    db = SessionLocal()
    document = service.upload_document(
        db=db,
        filename=filename,
        content=content,
    )

    # 原始文件名
    assert document.filename == filename

    # 文件大小
    assert document.size == len(content)

    # 服务端文件存在
    stored_file = tmp_path / document.stored_name

    assert stored_file.exists()

    # 文件内容一致
    assert stored_file.read_bytes() == content


def test_delete_document(tmp_path: Path):
    """
    测试文档删除功能。
    """

    storage_service = StorageService(
        storage_dir=str(tmp_path)
    )

    document_repository = DocumentRepository()

    service = DocumentService(
        storage_service,
        document_repository,
    )

    db = SessionLocal()

    try:
        filename = "test.pdf"

        document = service.upload_document(
            db=db,
            filename=filename,
            content=b"test document",
        )

        file_path = Path(document.path)

        assert file_path.exists()


        documents = document_repository.find_all(
            db=db,
        )

        database_document = next(
            item
            for item in documents
            if item.filename == filename
        )


        service.delete_document(
            db=db,
            document_id=database_document.id,
        )


        assert not file_path.exists()


        deleted_document = document_repository.find_by_id(
            db=db,
            document_id=database_document.id,
        )

        assert deleted_document is None

    finally:
        db.close()