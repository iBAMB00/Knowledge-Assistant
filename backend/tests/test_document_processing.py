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
        structure_metadata={
            "version": "1.0",
            "source_format": "txt",
            "sections": [],
        },
    )

    db.add(existing_content)
    db.commit()

    repository = DocumentContentRepository()

    updated_content = DocumentContent(
        document_id=document.id,
        content="新解析内容",
        parser_type="txt",
        parser_version="2.0",
        structure_metadata={
            "version": "1.0",
            "source_format": "markdown",
            "sections": [
                {
                    "section_index": 0,
                    "title": "新标题",
                    "level": 1,
                    "heading_path": ["新标题"],
                    "start_offset": 0,
                    "end_offset": 5,
                }
            ],
        },
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
    assert (
        saved_content.structure_metadata["source_format"]
        == "markdown"
    )

def test_process_document_builds_parent_child_chunks(
    db: Session,
    tmp_path,
    document_processing_service: DocumentProcessingService,
) -> None:
    """验证Parent保留原切片，Child通过parent_chunk_id建立层级。"""

    storage_service = StorageService(str(tmp_path))
    content = (
        "第一段介绍企业知识库的总体架构。" * 30
        + "\n\n"
        + "第二段介绍检索和问答流程。" * 30
    )
    stored_result = storage_service.save(
        "parent-child.txt",
        content.encode("utf-8"),
    )

    document = Document(
        filename="parent-child.txt",
        stored_name=stored_result.stored_name,
        path=stored_result.path,
        size=stored_result.size,
        status=DocumentStatus.UPLOADED.value,
    )
    db.add(document)
    db.commit()
    db.refresh(document)

    document_processing_service.process_document(
        db=db,
        document_id=document.id,
    )

    document_content = (
        db.query(DocumentContent)
        .filter(DocumentContent.document_id == document.id)
        .one()
    )
    chunks = (
        db.query(DocumentChunk)
        .filter(
            DocumentChunk.document_content_id
            == document_content.id
        )
        .order_by(DocumentChunk.chunk_index.asc())
        .all()
    )

    parents = [
        chunk
        for chunk in chunks
        if chunk.parent_chunk_id is None
    ]
    children = [
        chunk
        for chunk in chunks
        if chunk.parent_chunk_id is not None
    ]

    assert parents
    assert children
    assert {chunk.chunk_index for chunk in chunks} == set(
        range(len(chunks))
    )

    parent_ids = {parent.id for parent in parents}
    assert all(
        child.parent_chunk_id in parent_ids
        for child in children
    )
    assert all(
        child.chunk_metadata["chunk_role"] == "child"
        for child in children
    )
    assert all(
        parent.chunk_metadata["chunk_role"] == "parent"
        for parent in parents
    )



def test_process_markdown_persists_structure_metadata(
    db: Session,
    tmp_path,
    document_processing_service: DocumentProcessingService,
) -> None:
    """验证 Markdown 解析结构能够跨解析阶段持久化。"""

    storage_service = StorageService(str(tmp_path))
    source = (
        "# 部署指南\n"
        "先检查运行环境。\n\n"
        "```bash\n"
        "docker compose up -d\n"
        "```\n\n"
        "## PostgreSQL\n"
        "确认数据库连接正常。\n\n"
        "| 参数 | 值 |\n"
        "| --- | --- |\n"
        "| port | 5432 |"
    )
    stored_result = storage_service.save(
        "guide.md",
        source.encode("utf-8"),
    )

    document = Document(
        filename="guide.md",
        stored_name=stored_result.stored_name,
        path=stored_result.path,
        size=stored_result.size,
        status=DocumentStatus.UPLOADED.value,
    )
    db.add(document)
    db.commit()
    db.refresh(document)

    document_processing_service.process_document(
        db=db,
        document_id=document.id,
    )

    document_content = (
        db.query(DocumentContent)
        .filter(DocumentContent.document_id == document.id)
        .one()
    )

    assert document_content.parser_type == "markdown"
    assert document_content.structure_metadata is not None
    assert document_content.structure_metadata["source_format"] == "markdown"
    assert [
        section["title"]
        for section in document_content.structure_metadata["sections"]
    ] == [
        "部署指南",
        "PostgreSQL",
    ]
    assert document_content.structure_metadata["version"] == "1.2"
    assert [
        block["block_type"]
        for block in document_content.structure_metadata["blocks"]
    ] == [
        "code",
        "table",
    ]
