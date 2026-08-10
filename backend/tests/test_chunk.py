import fitz
import pytest

from app.schemas.chunk import ChunkResult
from app.services.chunking import (
    RecursiveCharacterChunkStrategy,
)
from app.services.chunk_service import (
    ChunkService,
)
from app.services.chunking.factory import (
    ChunkStrategyFactory,
)
from app.constants.document_status import DocumentStatus
from app.models.database.document import Document
from app.models.database.document_chunk import DocumentChunk
from app.models.database.document_content import DocumentContent
from app.repositories.document_chunk_repository import (
    DocumentChunkRepository,
)
from app.repositories.document_content_repository import (
    DocumentContentRepository,
)
from app.repositories.document_repository import DocumentRepository
from app.services.document_processing_service import (
    DocumentProcessingService,
)
from app.services.parser_service import ParserService
from app.services.storage_service import StorageService



# ============================================================
# ChunkResult
# ============================================================


def test_create_chunk_result():
    chunk = ChunkResult(
        content="员工请假需要提前提交申请。",
        chunk_index=0,
        start_offset=0,
        end_offset=14,
        token_count=10,
        metadata={
            "page": 1,
            "section": "请假制度",
        },
    )

    assert chunk.content == "员工请假需要提前提交申请。"
    assert chunk.chunk_index == 0
    assert chunk.start_offset == 0
    assert chunk.end_offset == 14
    assert chunk.token_count == 10
    assert chunk.metadata == {
        "page": 1,
        "section": "请假制度",
    }
    assert chunk.parent_chunk_index is None


def test_chunk_result_rejects_empty_content():
    with pytest.raises(
        ValueError,
        match="Chunk content cannot be empty",
    ):
        ChunkResult(
            content="   ",
            chunk_index=0,
            start_offset=0,
            end_offset=3,
        )


def test_chunk_result_rejects_negative_index():
    with pytest.raises(
        ValueError,
        match="Chunk index cannot be negative",
    ):
        ChunkResult(
            content="hello",
            chunk_index=-1,
            start_offset=0,
            end_offset=5,
        )


def test_chunk_result_rejects_invalid_offsets():
    with pytest.raises(
        ValueError,
        match="Chunk end offset must be greater than start offset",
    ):
        ChunkResult(
            content="hello",
            chunk_index=0,
            start_offset=5,
            end_offset=5,
        )


def test_chunk_result_rejects_negative_token_count():
    with pytest.raises(
        ValueError,
        match="Chunk token count cannot be negative",
    ):
        ChunkResult(
            content="hello",
            chunk_index=0,
            start_offset=0,
            end_offset=5,
            token_count=-1,
        )


def test_chunk_result_metadata_is_not_shared():
    first = ChunkResult(
        content="first",
        chunk_index=0,
        start_offset=0,
        end_offset=5,
    )

    second = ChunkResult(
        content="second",
        chunk_index=1,
        start_offset=5,
        end_offset=11,
    )

    first.metadata["page"] = 1

    assert second.metadata == {}


# ============================================================
# RecursiveCharacterChunkStrategy
# ============================================================


def test_recursive_strategy_name():
    strategy = RecursiveCharacterChunkStrategy()

    assert strategy.strategy_name == "recursive_character"


@pytest.mark.parametrize(
    ("chunk_size", "chunk_overlap", "error_message"),
    [
        (
            0,
            0,
            "Chunk size must be greater than zero",
        ),
        (
            10,
            -1,
            "Chunk overlap cannot be negative",
        ),
        (
            10,
            10,
            "Chunk overlap must be less than chunk size",
        ),
        (
            10,
            11,
            "Chunk overlap must be less than chunk size",
        ),
    ],
)
def test_recursive_strategy_rejects_invalid_config(
    chunk_size,
    chunk_overlap,
    error_message,
):
    with pytest.raises(
        ValueError,
        match=error_message,
    ):
        RecursiveCharacterChunkStrategy(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )


def test_recursive_strategy_rejects_empty_content():
    strategy = RecursiveCharacterChunkStrategy(
        chunk_size=10,
        chunk_overlap=2,
    )

    with pytest.raises(
        ValueError,
        match="Content cannot be empty",
    ):
        strategy.split("   ")


def test_recursive_strategy_returns_single_chunk_for_short_text():
    content = "员工请假需要提前申请。"

    strategy = RecursiveCharacterChunkStrategy(
        chunk_size=100,
        chunk_overlap=10,
    )

    chunks = strategy.split(
        content=content,
        metadata={
            "filename": "handbook.txt",
        },
    )

    assert len(chunks) == 1

    assert chunks[0].content == content
    assert chunks[0].chunk_index == 0
    assert chunks[0].start_offset == 0
    assert chunks[0].end_offset == len(content)
    assert chunks[0].metadata == {
        "filename": "handbook.txt",
    }


def test_recursive_strategy_uses_hard_split_as_fallback():
    content = "abcdefghij"

    strategy = RecursiveCharacterChunkStrategy(
        chunk_size=6,
        chunk_overlap=2,
        separators=[""],
    )

    chunks = strategy.split(content)

    assert [chunk.content for chunk in chunks] == [
        "abcdef",
        "efghij",
    ]

    assert [
        (
            chunk.start_offset,
            chunk.end_offset,
        )
        for chunk in chunks
    ] == [
        (0, 6),
        (4, 10),
    ]


def test_recursive_strategy_prefers_sentence_boundary():
    content = (
        "员工请假需要提前申请。"
        "主管收到申请后进行审批。"
        "审批通过后请假方可生效。"
    )

    strategy = RecursiveCharacterChunkStrategy(
        chunk_size=20,
        chunk_overlap=0,
    )

    chunks = strategy.split(content)

    assert len(chunks) >= 2

    assert chunks[0].content.endswith("。")

    assert all(
        len(chunk.content) <= 20
        for chunk in chunks
    )


def test_recursive_strategy_preserves_original_offsets():
    content = (
        "第一段内容用于测试。\n\n"
        "第二段内容继续测试。\n\n"
        "第三段内容结束测试。"
    )

    strategy = RecursiveCharacterChunkStrategy(
        chunk_size=15,
        chunk_overlap=3,
    )

    chunks = strategy.split(content)

    for index, chunk in enumerate(chunks):
        assert chunk.chunk_index == index

        assert (
            content[
                chunk.start_offset:chunk.end_offset
            ]
            == chunk.content
        )


def test_recursive_strategy_copies_metadata_for_each_chunk():
    strategy = RecursiveCharacterChunkStrategy(
        chunk_size=6,
        chunk_overlap=2,
        separators=[""],
    )

    chunks = strategy.split(
        content="abcdefghij",
        metadata={
            "filename": "test.txt",
        },
    )

    assert len(chunks) == 2

    chunks[0].metadata["page"] = 1

    assert chunks[1].metadata == {
        "filename": "test.txt",
    }

def test_chunk_service_split():
    """
    测试 ChunkService 能正常调用切片策略。
    """

    service = ChunkService()

    content = (
        "第一段内容。"
        "第二段内容。"
        "第三段内容。"
    )

    chunks = service.split(
        content=content,
        strategy_name="recursive_character",
    )

    assert len(chunks) > 0

    assert chunks[0].content

    assert chunks[0].start_offset == 0


def test_chunk_metadata_isolated():
    """
    测试不同 Chunk 的 metadata 不共享引用。
    """

    service = ChunkService()

    metadata = {
        "document_id": 1
    }

    chunks = service.split(
        content=(
            "第一段内容。"
            "第二段内容。"
        ),
        strategy_name="recursive_character",
        metadata=metadata,
    )


    assert len(chunks) > 0


    if len(chunks) > 1:
        assert (
            chunks[0].metadata
            is not
            chunks[1].metadata
        )

def test_chunk_strategy_factory_uses_settings(
    monkeypatch,
) -> None:
    """
    验证切片策略工厂使用应用配置。
    """

    class FakeSettings:
        """
        测试使用的切片配置。
        """

        chunk_size = 600
        chunk_overlap = 100

    monkeypatch.setattr(
        "app.services.chunking.factory"
        ".get_settings",
        lambda: FakeSettings(),
    )

    strategy = ChunkStrategyFactory.create(
        strategy_name="recursive_character",
    )

    assert isinstance(
        strategy,
        RecursiveCharacterChunkStrategy,
    )

    assert strategy.chunk_size == 600
    assert strategy.chunk_overlap == 100

def test_split_aligns_overlap_start_to_sentence_boundary():
    """
    验证Overlap起点优先对齐完整句子。
    """

    content = (
        "甲" * 12
        + "。"
        + "乙" * 12
        + "。"
        + "丙" * 12
        + "。"
    )

    strategy = RecursiveCharacterChunkStrategy(
        chunk_size=26,
        chunk_overlap=8,
    )

    chunks = strategy.split(content)

    assert len(chunks) >= 2

    expected_start = content.index("乙")

    assert chunks[1].start_offset == expected_start
    assert chunks[1].content.startswith("乙")

def test_split_keeps_content_and_offsets_consistent():
    """
    验证Chunk内容始终对应原文Offset。
    """

    content = (
        "第一段介绍部署要求。"
        "第二段介绍网络端口。"
        "第三段介绍日志归档。"
    )

    strategy = RecursiveCharacterChunkStrategy(
        chunk_size=20,
        chunk_overlap=6,
    )

    chunks = strategy.split(content)

    for chunk in chunks:
        assert chunk.content == content[
            chunk.start_offset:chunk.end_offset
        ]

def test_split_covers_all_non_whitespace_characters():
    """
    验证切片结果不会遗漏原文中的非空白字符。
    """

    content = (
        "第一段介绍部署要求。\n\n"
        "第二段介绍网络端口。\n"
        "第三段介绍日志归档。"
    )

    strategy = RecursiveCharacterChunkStrategy(
        chunk_size=20,
        chunk_overlap=6,
    )

    chunks = strategy.split(content)

    covered = [False] * len(content)

    for chunk in chunks:
        for index in range(
            chunk.start_offset,
            chunk.end_offset,
        ):
            covered[index] = True

    assert all(
        character.isspace() or covered[index]
        for index, character in enumerate(content)
    )

def test_split_does_not_create_duplicate_tail_chunk():
    """
    验证文档末尾空白不会产生额外重复Chunk。
    """

    content = "最后一段完整内容。\n\n"

    strategy = RecursiveCharacterChunkStrategy(
        chunk_size=100,
        chunk_overlap=20,
    )

    chunks = strategy.split(content)

    assert len(chunks) == 1
    assert chunks[0].content == "最后一段完整内容。"
    assert not content[chunks[0].end_offset:].strip()

def test_split_start_offsets_strictly_increase():
    """
    验证每个Chunk起点严格递增。
    """

    content = (
        "第一段内容。" * 20
    )

    strategy = RecursiveCharacterChunkStrategy(
        chunk_size=30,
        chunk_overlap=10,
    )

    chunks = strategy.split(content)

    assert all(
        current.start_offset
        < following.start_offset
        for current, following in zip(
            chunks,
            chunks[1:],
        )
    )

    assert all(
        following.start_offset
        <= current.end_offset
        for current, following in zip(
            chunks,
            chunks[1:],
        )
    )

# ============================================================
# Structure-aware Parent Chunk
# ============================================================


def test_structure_aware_parent_respects_section_boundaries() -> None:
    """验证短 Section 不会被合并进同一个 Parent。"""

    content = (
        "# 部署\n"
        "准备运行环境。\n\n"
        "## PostgreSQL\n"
        "确认数据库连接。"
    )
    second_start = content.index("## PostgreSQL")
    structure_metadata = {
        "version": "1.0",
        "source_format": "markdown",
        "sections": [
            {
                "section_index": 0,
                "title": "部署",
                "level": 1,
                "heading_path": ["部署"],
                "start_offset": 0,
                "end_offset": second_start,
            },
            {
                "section_index": 1,
                "title": "PostgreSQL",
                "level": 2,
                "heading_path": ["部署", "PostgreSQL"],
                "start_offset": second_start,
                "end_offset": len(content),
            },
        ],
    }

    chunks = ChunkService().split_parent_by_structure(
        content=content,
        strategy_name="recursive_character",
        structure_metadata=structure_metadata,
        metadata={"chunk_role": "parent"},
        chunk_size=600,
        chunk_overlap=100,
    )

    assert len(chunks) == 2
    assert chunks[0].content.strip().startswith("# 部署")
    assert "PostgreSQL" not in chunks[0].content
    assert chunks[1].content.startswith("## PostgreSQL")
    assert chunks[1].metadata["heading_path"] == [
        "部署",
        "PostgreSQL",
    ]
    assert all(
        chunk.metadata["structure_aware"] is True
        for chunk in chunks
    )


def test_structure_aware_long_section_splits_inside_section() -> None:
    """验证超长 Section 只在自身边界内递归切分，并保留全文 offset。"""

    first_body = "甲" * 90
    content = (
        "# 第一章\n"
        + first_body
        + "\n\n## 第二章\n"
        + "乙" * 10
    )
    second_start = content.index("## 第二章")
    structure_metadata = {
        "version": "1.0",
        "source_format": "markdown",
        "sections": [
            {
                "section_index": 0,
                "title": "第一章",
                "level": 1,
                "heading_path": ["第一章"],
                "start_offset": 0,
                "end_offset": second_start,
            },
            {
                "section_index": 1,
                "title": "第二章",
                "level": 2,
                "heading_path": ["第一章", "第二章"],
                "start_offset": second_start,
                "end_offset": len(content),
            },
        ],
    }

    chunks = ChunkService().split_parent_by_structure(
        content=content,
        strategy_name="recursive_character",
        structure_metadata=structure_metadata,
        chunk_size=40,
        chunk_overlap=5,
    )

    first_section_chunks = [
        chunk
        for chunk in chunks
        if chunk.metadata["section_index"] == 0
    ]

    assert len(first_section_chunks) >= 2
    assert all(
        chunk.end_offset <= second_start
        for chunk in first_section_chunks
    )
    assert all(
        content[chunk.start_offset:chunk.end_offset]
        == chunk.content
        for chunk in chunks
    )
    assert [
        chunk.metadata["section_part_index"]
        for chunk in first_section_chunks
    ] == list(range(len(first_section_chunks)))


def test_structure_aware_rejects_partial_structure_metadata() -> None:
    """验证结构未覆盖非空正文时不进行部分结构切片。"""

    content = "# 标题\n正文一\n\n## 子标题\n正文二"
    second_start = content.index("## 子标题")
    structure_metadata = {
        "version": "1.0",
        "source_format": "markdown",
        "sections": [
            {
                "section_index": 0,
                "title": "标题",
                "level": 1,
                "heading_path": ["标题"],
                "start_offset": 0,
                "end_offset": second_start,
            }
        ],
    }

    chunks = ChunkService().split_parent_by_structure(
        content=content,
        strategy_name="recursive_character",
        structure_metadata=structure_metadata,
        chunk_size=600,
        chunk_overlap=100,
    )

    assert chunks == []


def test_markdown_processing_builds_section_aware_parent_child_chunks(
    db,
    tmp_path,
) -> None:
    """验证文档处理链路真正使用 Section Parent，并向 Child 传播章节元数据。"""

    storage_service = StorageService(str(tmp_path))
    source = (
        "# API认证\n"
        + "认证说明。" * 10
        + "\n\n## 错误处理\n"
        + "错误处理说明。" * 10
    )
    stored_result = storage_service.save(
        "structured-guide.md",
        source.encode("utf-8"),
    )
    document = Document(
        filename="structured-guide.md",
        storage_key=stored_result.storage_key,
        size=stored_result.size,
        status=DocumentStatus.UPLOADED.value,
    )
    db.add(document)
    db.commit()
    db.refresh(document)

    service = DocumentProcessingService(
        storage_service=storage_service,
        document_repository=DocumentRepository(),
        document_content_repository=DocumentContentRepository(),
        parser_service=ParserService(),
        chunk_service=ChunkService(),
        document_chunk_repository=DocumentChunkRepository(),
    )
    service.process_document(
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

    # 全文不足600字符，普通全文切片只会产生1个Parent；
    # 结构感知模式应按两个Section生成2个Parent。
    assert len(source) < 600
    assert len(parents) == 2
    assert [
        parent.chunk_metadata["section_title"]
        for parent in parents
    ] == [
        "API认证",
        "错误处理",
    ]
    assert all(
        parent.chunk_metadata["structure_aware"] is True
        for parent in parents
    )
    assert children
    assert all(
        child.chunk_metadata["section_title"]
        in {"API认证", "错误处理"}
        for child in children
    )
    assert all(
        isinstance(
            child.chunk_metadata.get("document_start_offset"),
            int,
        )
        for child in children
    )


def test_pdf_processing_propagates_page_ranges_to_parent_and_child(
    db,
    tmp_path,
) -> None:
    """验证 PDF Page offset 会映射到 Parent / Child 的准确页码。"""

    pdf = fitz.open()
    first_page = pdf.new_page()
    first_page.insert_text(
        (72, 72),
        "Deployment Guide",
        fontsize=20,
    )
    first_page.insert_text(
        (72, 110),
        "Deployment prerequisites and environment checks.",
        fontsize=11,
    )
    second_page = pdf.new_page()
    second_page.insert_text(
        (72, 72),
        "Redis",
        fontsize=16,
    )
    second_page.insert_text(
        (72, 105),
        "Configure Redis for Celery broker operations.",
        fontsize=11,
    )
    pdf_content = pdf.tobytes()
    pdf.close()

    storage_service = StorageService(str(tmp_path))
    stored_result = storage_service.save(
        "deployment.pdf",
        pdf_content,
    )
    document = Document(
        filename="deployment.pdf",
        storage_key=stored_result.storage_key,
        size=stored_result.size,
        status=DocumentStatus.UPLOADED.value,
    )
    db.add(document)
    db.commit()
    db.refresh(document)

    service = DocumentProcessingService(
        storage_service=storage_service,
        document_repository=DocumentRepository(),
        document_content_repository=DocumentContentRepository(),
        parser_service=ParserService(),
        chunk_service=ChunkService(),
        document_chunk_repository=DocumentChunkRepository(),
    )
    service.process_document(
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

    assert document_content.structure_metadata is not None
    assert [
        page["page_number"]
        for page in document_content.structure_metadata["pages"]
    ] == [1, 2]
    assert [
        parent.chunk_metadata["page_numbers"]
        for parent in parents
    ] == [[1], [2]]
    assert [
        parent.chunk_metadata["section_title"]
        for parent in parents
    ] == [
        "Deployment Guide",
        "Redis",
    ]
    assert children
    assert all(
        child.chunk_metadata["page_numbers"] in ([1], [2])
        for child in children
    )


def test_pdf_page_only_structure_keeps_document_chunking_fallback(
    db,
    tmp_path,
) -> None:
    """验证无可靠 Heading 的 PDF 不按页切 Parent，但仍保留页码定位。"""

    pdf = fitz.open()
    first_page = pdf.new_page()
    first_page.insert_text(
        (72, 72),
        "Normal first page paragraph.",
        fontsize=11,
    )
    second_page = pdf.new_page()
    second_page.insert_text(
        (72, 72),
        "Normal second page paragraph.",
        fontsize=11,
    )
    pdf_content = pdf.tobytes()
    pdf.close()

    storage_service = StorageService(str(tmp_path))
    stored_result = storage_service.save(
        "plain-pages.pdf",
        pdf_content,
    )
    document = Document(
        filename="plain-pages.pdf",
        storage_key=stored_result.storage_key,
        size=stored_result.size,
        status=DocumentStatus.UPLOADED.value,
    )
    db.add(document)
    db.commit()
    db.refresh(document)

    service = DocumentProcessingService(
        storage_service=storage_service,
        document_repository=DocumentRepository(),
        document_content_repository=DocumentContentRepository(),
        parser_service=ParserService(),
        chunk_service=ChunkService(),
        document_chunk_repository=DocumentChunkRepository(),
    )
    service.process_document(
        db=db,
        document_id=document.id,
    )

    document_content = (
        db.query(DocumentContent)
        .filter(DocumentContent.document_id == document.id)
        .one()
    )
    parents = (
        db.query(DocumentChunk)
        .filter(
            DocumentChunk.document_content_id
            == document_content.id,
            DocumentChunk.parent_chunk_id.is_(None),
        )
        .order_by(DocumentChunk.chunk_index.asc())
        .all()
    )

    assert document_content.structure_metadata is not None
    assert document_content.structure_metadata["sections"] == []
    assert len(parents) == 1
    assert parents[0].chunk_metadata["structure_aware"] is False
    assert parents[0].chunk_metadata["page_numbers"] == [1, 2]

