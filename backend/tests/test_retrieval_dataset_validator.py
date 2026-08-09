import hashlib
from datetime import datetime

import pytest
from sqlalchemy.orm import Session

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
from app.repositories.document_repository import (
    DocumentRepository,
)
from app.schemas.retrieval_evaluation import (
    RetrievalCaseCategory,
    RetrievalCaseDifficulty,
    RetrievalEvaluationCase,
    RetrievalEvaluationDataset,
    RetrievalEvaluationDocumentReference,
)
from app.services.evaluation.retrieval_dataset_validator import (
    RetrievalDatasetValidator,
)


def build_validator() -> RetrievalDatasetValidator:
    """创建评估数据集数据库校验器。"""

    return RetrievalDatasetValidator(
        document_repository=DocumentRepository(),
        document_content_repository=(
            DocumentContentRepository()
        ),
        document_chunk_repository=(
            DocumentChunkRepository()
        ),
    )


def create_corpus_document(
    db: Session,
    document_id: int,
    filename: str,
    content: str,
    chunk_contents: list[str],
) -> tuple[Document, list[DocumentChunk]]:
    """创建已完成文档、解析内容和Chunk。"""

    document = Document(
        id=document_id,
        filename=filename,
        storage_key=f"stored-{document_id}.txt",
        stored_name=f"stored-{document_id}.txt",
        path=f"tests/uploads/stored-{document_id}.txt",
        size=len(content.encode("utf-8")),
        status=DocumentStatus.COMPLETED.value,
    )
    db.add(document)
    db.flush()

    document_content = DocumentContent(
        document_id=document.id,
        content=content,
        parser_type="text",
        parser_version="1.0",
    )
    db.add(document_content)
    db.flush()

    chunks = [
        DocumentChunk(
            document_content_id=document_content.id,
            chunk_index=index,
            content=chunk_content,
            token_count=None,
            chunk_strategy="recursive_character",
        )
        for index, chunk_content in enumerate(
            chunk_contents
        )
    ]

    db.add_all(chunks)
    db.commit()

    return document, chunks


def build_dataset(
    document: Document,
    content: str,
    chunk: DocumentChunk,
    strict_corpus: bool = True,
) -> RetrievalEvaluationDataset:
    """创建与数据库语料匹配的数据集。"""

    return RetrievalEvaluationDataset(
        schema_version="1.0",
        dataset_id="validator-test",
        dataset_version="1.0.0",
        description="数据库引用校验测试",
        strict_corpus=strict_corpus,
        corpus_documents=[
            RetrievalEvaluationDocumentReference(
                document_id=document.id,
                filename=document.filename,
                content_sha256=hashlib.sha256(
                    content.encode("utf-8")
                ).hexdigest(),
            )
        ],
        cases=[
            RetrievalEvaluationCase(
                case_id="case-001",
                question="测试问题",
                category=(
                    RetrievalCaseCategory.EXACT_TERM
                ),
                difficulty=(
                    RetrievalCaseDifficulty.EASY
                ),
                should_retrieve=True,
                expected_document_ids=[
                    document.id
                ],
                expected_chunk_ids=[chunk.id],
            )
        ],
    )


def test_validate_accepts_matching_corpus(
    db: Session,
) -> None:
    """验证文档、内容哈希和Chunk均匹配时通过。"""

    content = "固定评估文档全文"
    document, chunks = create_corpus_document(
        db=db,
        document_id=101,
        filename="evaluation.txt",
        content=content,
        chunk_contents=["固定评估文档切片"],
    )

    result = build_validator().validate(
        db=db,
        dataset=build_dataset(
            document=document,
            content=content,
            chunk=chunks[0],
        ),
    )

    assert result.corpus_document_count == 1
    assert result.referenced_chunk_count == 1
    assert result.case_count == 1


def test_validate_rejects_content_hash_mismatch(
    db: Session,
) -> None:
    """验证解析全文变化后数据集快速失败。"""

    content = "真实文档内容"
    document, chunks = create_corpus_document(
        db=db,
        document_id=102,
        filename="hash.txt",
        content=content,
        chunk_contents=["真实切片"],
    )

    dataset = build_dataset(
        document=document,
        content="错误文档内容",
        chunk=chunks[0],
    )

    with pytest.raises(
        ValueError,
        match="content hash mismatch",
    ):
        build_validator().validate(
            db=db,
            dataset=dataset,
        )


def test_validate_rejects_extra_document_in_strict_corpus(
    db: Session,
) -> None:
    """验证严格语料模式拒绝数据库中的额外文档。"""

    content = "目标文档"
    document, chunks = create_corpus_document(
        db=db,
        document_id=103,
        filename="target.txt",
        content=content,
        chunk_contents=["目标切片"],
    )

    create_corpus_document(
        db=db,
        document_id=104,
        filename="extra.txt",
        content="额外文档",
        chunk_contents=["额外切片"],
    )

    with pytest.raises(
        ValueError,
        match="outside the strict evaluation corpus",
    ):
        build_validator().validate(
            db=db,
            dataset=build_dataset(
                document=document,
                content=content,
                chunk=chunks[0],
            ),
        )


def test_validate_rejects_chunk_from_unexpected_document(
    db: Session,
) -> None:
    """验证目标Chunk必须属于该用例标注的文档。"""

    content = "目标文档"
    target_document, target_chunks = (
        create_corpus_document(
            db=db,
            document_id=105,
            filename="target.txt",
            content=content,
            chunk_contents=["目标切片"],
        )
    )

    other_document, other_chunks = (
        create_corpus_document(
            db=db,
            document_id=106,
            filename="other.txt",
            content="其他文档",
            chunk_contents=["其他切片"],
        )
    )

    dataset = build_dataset(
        document=target_document,
        content=content,
        chunk=target_chunks[0],
        strict_corpus=False,
    )
    dataset.cases[0].expected_chunk_ids = [
        other_chunks[0].id
    ]

    with pytest.raises(
        ValueError,
        match="unexpected document",
    ):
        build_validator().validate(
            db=db,
            dataset=dataset,
        )
