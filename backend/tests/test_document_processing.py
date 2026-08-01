import pytest
from sqlalchemy.orm import Session

from app.services.document_processing_service import DocumentProcessingService
from app.services.storage_service import StorageService
from app.services.parser_service import ParserService
from app.repositories.document_repository import DocumentRepository
from app.repositories.document_content_repository import DocumentContentRepository
from app.repositories.document_chunk_repository import DocumentChunkRepository
from app.services.chunk_service import ChunkService
from app.models.database.document import Document
from app.models.database.document_content import DocumentContent
from app.models.database.document_chunk import DocumentChunk
from app.constants.document_status import DocumentStatus


@pytest.fixture()
def document_processing_service(tmp_path):
    """
    创建文档处理服务。
    """

    return DocumentProcessingService(
        storage_service=StorageService(str(tmp_path)),
        document_repository=DocumentRepository(),
        document_content_repository=DocumentContentRepository(),
        parser_service=ParserService(),
        chunk_service=ChunkService(),
        document_chunk_repository=DocumentChunkRepository(),
    )


def test_process_document_success(
    db,
    tmp_path,
    document_processing_service: DocumentProcessingService,
):
    """
    测试文档解析、切片、持久化完整流程。
    """

    # 1. 创建测试文件
    storage_service = StorageService(
        str(tmp_path)
    )

    stored_result = storage_service.save(
        "test.txt",
        "这是第一句话。\n这是第二句话。".encode(
            "utf-8"
        ),
    )


    # 2. 创建 Document 数据
    document = Document(
        filename="test.txt",
        stored_name=stored_result.stored_name,
        path=stored_result.path,
        size=stored_result.size,
        status=DocumentStatus.UPLOADED.value,
    )

    db.add(document)
    db.commit()
    db.refresh(document)


    # 3. 执行处理流程
    result = document_processing_service.process_document(
        db=db,
        document_id=document.id,
    )


    # 4. 验证状态
    assert result.status == (
        DocumentStatus.CHUNKED.value
    )


    # 5. 验证解析内容
    content = (
        db.query(DocumentContent)
        .filter(
            DocumentContent.document_id
            == document.id
        )
        .first()
    )

    assert content is not None


    # 6. 验证 Chunk
    chunks = (
        db.query(DocumentChunk)
        .filter(
            DocumentChunk.document_content_id
            == content.id
        )
        .all()
    )

    assert len(chunks) > 0

def test_document_content_upsert_updates_parser_version(
    db: Session,
) -> None:
    """
    验证重新解析时同步更新解析器版本。
    """

    document = Document(
        filename="parser-version.txt",
        stored_name="parser-version-stored.txt",
        path="tests/uploads/parser-version-stored.txt",
        size=100,
        status=DocumentStatus.PARSED.value,
    )

    db.add(document)
    db.flush()

    existing_content = DocumentContent(
        document_id=document.id,
        content="旧解析内容",
        parser_type="txt",
        parser_version="1.0",
    )

    db.add(existing_content)
    db.commit()

    repository = DocumentContentRepository()

    updated_content = DocumentContent(
        document_id=document.id,
        content="新解析内容",
        parser_type="txt",
        parser_version="2.0",
    )

    saved_content = repository.save_or_update(
        db=db,
        document_content=updated_content,
    )

    db.commit()
    db.refresh(saved_content)

    assert saved_content.id == existing_content.id
    assert saved_content.content == "新解析内容"
    assert saved_content.parser_type == "txt"
    assert saved_content.parser_version == "2.0"