from dataclasses import dataclass
import hashlib

from sqlalchemy.orm import Session

from app.constants.document_status import DocumentStatus
from app.repositories.document_chunk_repository import DocumentChunkRepository
from app.repositories.document_content_repository import DocumentContentRepository
from app.repositories.document_repository import DocumentRepository
from app.schemas.retrieval_evaluation import RetrievalEvaluationDataset


@dataclass(frozen=True)
class RetrievalDatasetValidationResult:
    """评估数据集与当前数据库的校验结果。"""

    corpus_document_count: int
    referenced_chunk_count: int
    case_count: int


class RetrievalDatasetValidator:
    """
    检索评估数据集数据库引用校验器。

    负责：
    - 校验固定语料文档是否存在且已处理完成
    - 校验文档文件名和解析全文哈希
    - 校验预期Chunk是否存在并属于正确文档
    - 在strict_corpus模式下拒绝额外文档

    不负责：
    - 修改文档、Chunk或评估数据
    - 执行检索
    - 提交数据库事务
    """

    def __init__(
        self,
        document_repository: DocumentRepository,
        document_content_repository: DocumentContentRepository,
        document_chunk_repository: DocumentChunkRepository,
    ) -> None:
        self.document_repository = document_repository
        self.document_content_repository = (
            document_content_repository
        )
        self.document_chunk_repository = (
            document_chunk_repository
        )

    def validate(
        self,
        db: Session,
        dataset: RetrievalEvaluationDataset,
    ) -> RetrievalDatasetValidationResult:
        """校验评估数据集是否匹配当前数据库语料。"""

        actual_documents = (
            self.document_repository.find_all(db=db)
        )
        actual_documents_by_id = {
            document.id: document
            for document in actual_documents
        }

        expected_documents_by_id = {
            document.document_id: document
            for document in dataset.corpus_documents
        }

        expected_document_ids = set(
            expected_documents_by_id
        )
        actual_document_ids = set(
            actual_documents_by_id
        )

        missing_document_ids = (
            expected_document_ids
            - actual_document_ids
        )

        if missing_document_ids:
            raise ValueError(
                "evaluation corpus documents are missing: "
                f"{sorted(missing_document_ids)}"
            )

        if dataset.strict_corpus:
            unexpected_document_ids = (
                actual_document_ids
                - expected_document_ids
            )

            if unexpected_document_ids:
                raise ValueError(
                    "database contains documents outside "
                    "the strict evaluation corpus: "
                    f"{sorted(unexpected_document_ids)}"
                )

        contents_by_document_id = (
            self.document_content_repository
            .find_by_document_ids(
                db=db,
                document_ids=sorted(
                    expected_document_ids
                ),
            )
        )

        for document_id, reference in (
            expected_documents_by_id.items()
        ):
            document = actual_documents_by_id[
                document_id
            ]

            if document.filename != reference.filename:
                raise ValueError(
                    "evaluation corpus filename mismatch: "
                    f"document_id={document_id}, "
                    f"expected={reference.filename}, "
                    f"actual={document.filename}"
                )

            if (
                DocumentStatus(document.status)
                != DocumentStatus.COMPLETED
            ):
                raise ValueError(
                    "evaluation corpus document is not "
                    "completed: "
                    f"document_id={document_id}, "
                    f"status={document.status}"
                )

            document_content = (
                contents_by_document_id.get(
                    document_id
                )
            )

            if document_content is None:
                raise ValueError(
                    "evaluation corpus document content "
                    "is missing: "
                    f"document_id={document_id}"
                )

            actual_content_sha256 = hashlib.sha256(
                document_content.content.encode(
                    "utf-8"
                )
            ).hexdigest()

            if (
                actual_content_sha256
                != reference.content_sha256
            ):
                raise ValueError(
                    "evaluation corpus content hash "
                    "mismatch: "
                    f"document_id={document_id}"
                )

        expected_chunk_ids = sorted({
            chunk_id
            for case in dataset.cases
            for chunk_id in case.expected_chunk_ids
        })

        chunk_document_ids = (
            self.document_chunk_repository
            .find_document_ids_by_chunk_ids(
                db=db,
                chunk_ids=expected_chunk_ids,
            )
        )

        missing_chunk_ids = (
            set(expected_chunk_ids)
            - set(chunk_document_ids)
        )

        if missing_chunk_ids:
            raise ValueError(
                "evaluation expected chunks are missing: "
                f"{sorted(missing_chunk_ids)}"
            )

        for case in dataset.cases:
            expected_case_document_ids = set(
                case.expected_document_ids
            )

            for chunk_id in case.expected_chunk_ids:
                chunk_document_id = (
                    chunk_document_ids[chunk_id]
                )

                if (
                    chunk_document_id
                    not in expected_case_document_ids
                ):
                    raise ValueError(
                        "evaluation expected chunk belongs "
                        "to an unexpected document: "
                        f"case_id={case.case_id}, "
                        f"chunk_id={chunk_id}, "
                        f"document_id={chunk_document_id}"
                    )

                if (
                    case.document_id is not None
                    and chunk_document_id
                    != case.document_id
                ):
                    raise ValueError(
                        "evaluation expected chunk violates "
                        "document filter: "
                        f"case_id={case.case_id}, "
                        f"chunk_id={chunk_id}"
                    )

        return RetrievalDatasetValidationResult(
            corpus_document_count=(
                len(expected_document_ids)
            ),
            referenced_chunk_count=(
                len(expected_chunk_ids)
            ),
            case_count=len(dataset.cases),
        )
