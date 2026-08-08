from __future__ import annotations

import argparse
from pathlib import Path

from sqlalchemy.orm import Session

from app.constants.embedding_status import EmbeddingStatus
from app.core.config import get_settings
from app.core.database import SessionLocal
from app.models.database.chunk_embedding import ChunkEmbedding
from app.models.database.document_chunk import DocumentChunk
from app.repositories.chunk_embedding_repository import ChunkEmbeddingRepository
from app.repositories.document_chunk_repository import DocumentChunkRepository
from app.repositories.document_content_repository import DocumentContentRepository
from app.repositories.document_repository import DocumentRepository
from app.services.chunk_service import ChunkService
from app.services.embedding.factory import EmbeddingFactory
from app.services.evaluation.retrieval_case_loader import RetrievalCaseLoader
from app.services.evaluation.retrieval_dataset_validator import RetrievalDatasetValidator
from app.services.status_machine import StatusMachine
from app.services.vector_index_service import VectorIndexService
from app.services.vector_store.factory import get_vector_store_components


DEFAULT_CASES_PATH = Path("evaluation/retrieval_cases_v2.json")
settings = get_settings()


def parse_args() -> argparse.Namespace:
    """解析 Parent-Child 评估语料补建参数。"""

    parser = argparse.ArgumentParser(
        description=(
            "Backfill Child chunks for the frozen v0.12 evaluation corpus "
            "without replacing existing Parent chunk IDs."
        )
    )
    parser.add_argument(
        "--cases",
        type=Path,
        default=DEFAULT_CASES_PATH,
        help="Versioned evaluation dataset used to select corpus documents.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Embedding batch size for newly created Child chunks.",
    )
    args = parser.parse_args()

    if args.batch_size <= 0:
        parser.error("--batch-size must be greater than zero")

    return args


class EvaluationParentChildBackfill:
    """
    为冻结评估语料补建 Child Chunk。

    关键约束：
    - 不删除、不重建已有 Parent，保留 v0.12 expected_chunk_ids；
    - 只为尚无 Child 的 Parent 创建 Child；
    - 只向量化新 Child；
    - 外部向量索引启用时，最后重建该文档全部 Point Payload。
    """

    def __init__(self) -> None:
        self.document_repository = DocumentRepository()
        self.document_chunk_repository = DocumentChunkRepository()
        self.chunk_embedding_repository = ChunkEmbeddingRepository()
        self.chunk_service = ChunkService()
        self.embedding_provider = EmbeddingFactory.create()

        vector_index = get_vector_store_components().vector_index
        self.vector_index_service = (
            VectorIndexService(
                document_repository=self.document_repository,
                chunk_embedding_repository=self.chunk_embedding_repository,
                vector_index=vector_index,
            )
            if vector_index is not None
            else None
        )

    def run(
        self,
        db: Session,
        document_ids: list[int],
        batch_size: int,
    ) -> tuple[int, int]:
        """补建全部目标文档，返回新增 Child 数和完成文档数。"""

        if not settings.parent_child_enabled:
            raise RuntimeError(
                "PARENT_CHILD_ENABLED must be true before backfill"
            )

        total_children = 0
        completed_documents = 0

        for document_id in document_ids:
            created_children = self._backfill_document(
                db=db,
                document_id=document_id,
                batch_size=batch_size,
            )
            total_children += created_children
            completed_documents += 1
            print(
                f"[done] document_id={document_id}: "
                f"new_children={created_children}"
            )

        return total_children, completed_documents

    def _backfill_document(
        self,
        db: Session,
        document_id: int,
        batch_size: int,
    ) -> int:
        """为单文档缺失的 Parent 补 Child，并同步 Embedding/索引。"""

        document = self.document_repository.find_by_id(
            db=db,
            document_id=document_id,
        )
        if document is None:
            raise ValueError(f"document not found: {document_id}")

        chunks = self.document_chunk_repository.find_by_document_id(
            db=db,
            document_id=document_id,
        )
        parents = [
            chunk
            for chunk in chunks
            if chunk.parent_chunk_id is None
        ]
        if not parents:
            raise RuntimeError(
                f"document has no Parent chunks: {document_id}"
            )

        parent_ids_with_children = {
            chunk.parent_chunk_id
            for chunk in chunks
            if chunk.parent_chunk_id is not None
        }
        next_chunk_index = (
            max(chunk.chunk_index for chunk in chunks) + 1
            if chunks
            else 0
        )

        new_children: list[DocumentChunk] = []

        for parent in parents:
            if parent.id in parent_ids_with_children:
                continue

            child_results = self.chunk_service.split(
                content=parent.content,
                strategy_name=settings.chunk_strategy,
                chunk_size=settings.parent_child_child_size,
                chunk_overlap=settings.parent_child_child_overlap,
                metadata={
                    "document_content_id": parent.document_content_id,
                    "chunk_strategy": settings.chunk_strategy,
                    "chunk_role": "child",
                    "parent_chunk_id": parent.id,
                    "parent_chunk_index": parent.chunk_index,
                    "evaluation_backfill": True,
                },
            )

            for child_result in child_results:
                new_children.append(
                    DocumentChunk(
                        document_content_id=parent.document_content_id,
                        chunk_index=next_chunk_index,
                        content=child_result.content,
                        token_count=child_result.token_count,
                        chunk_strategy=settings.chunk_strategy,
                        chunk_metadata=child_result.metadata,
                        parent_chunk_id=parent.id,
                    )
                )
                next_chunk_index += 1

        if new_children:
            self.document_chunk_repository.save_all(
                db=db,
                chunks=new_children,
            )
            db.commit()

            self._embed_new_children(
                db=db,
                children=new_children,
                batch_size=batch_size,
            )

        # 即使本次没有新增 Child，也重新 upsert 一次，确保旧 Parent Point
        # 获得 v0.14 所需的 chunk_role / parent_chunk_id Payload。
        if self.vector_index_service is not None:
            self.vector_index_service.index_document(
                db=db,
                document_id=document_id,
            )

        return len(new_children)

    def _embed_new_children(
        self,
        db: Session,
        children: list[DocumentChunk],
        batch_size: int,
    ) -> None:
        """只向量化本次补建的 Child，不改变 Document 终态。"""

        child_ids = [child.id for child in children]

        for offset in range(0, len(child_ids), batch_size):
            batch_ids = child_ids[offset:offset + batch_size]
            batch = self.document_chunk_repository.find_by_ids(
                db=db,
                chunk_ids=batch_ids,
            )

            try:
                for chunk in batch:
                    StatusMachine.transition_embedding(
                        chunk=chunk,
                        target_status=EmbeddingStatus.PROCESSING,
                    )
                db.commit()

                vectors = self.embedding_provider.embed_documents(
                    [chunk.content for chunk in batch]
                )
                if len(vectors) != len(batch):
                    raise RuntimeError(
                        "embedding result count does not match Child count"
                    )

                for chunk, vector in zip(batch, vectors, strict=True):
                    self.chunk_embedding_repository.save_or_update(
                        db=db,
                        embedding=ChunkEmbedding(
                            document_chunk_id=chunk.id,
                            vector=vector,
                            embedding_model=self.embedding_provider.model_name,
                            embedding_dimension=len(vector),
                            embedding_metadata={
                                "source": "evaluation_parent_child_backfill"
                            },
                        ),
                    )
                    StatusMachine.transition_embedding(
                        chunk=chunk,
                        target_status=EmbeddingStatus.COMPLETED,
                    )

                db.commit()

            except Exception:
                db.rollback()
                failed_chunks = self.document_chunk_repository.find_by_ids(
                    db=db,
                    chunk_ids=batch_ids,
                )
                for chunk in failed_chunks:
                    if (
                        EmbeddingStatus(chunk.embedding_status)
                        == EmbeddingStatus.PROCESSING
                    ):
                        StatusMachine.transition_embedding(
                            chunk=chunk,
                            target_status=EmbeddingStatus.FAILED,
                        )
                db.commit()
                raise


def build_dataset_validator() -> RetrievalDatasetValidator:
    """构建与正式评估相同的数据集校验器。"""

    return RetrievalDatasetValidator(
        document_repository=DocumentRepository(),
        document_content_repository=DocumentContentRepository(),
        document_chunk_repository=DocumentChunkRepository(),
    )


def main() -> None:
    """校验冻结语料后补建 Child Chunk。"""

    args = parse_args()
    dataset = RetrievalCaseLoader.load(args.cases)

    with SessionLocal() as db:
        validation = build_dataset_validator().validate(
            db=db,
            dataset=dataset,
        )
        print(
            "Dataset validation passed before backfill: "
            f"documents={validation.corpus_document_count}, "
            f"chunks={validation.referenced_chunk_count}, "
            f"cases={validation.case_count}"
        )

        document_ids = [
            item.document_id
            for item in dataset.corpus_documents
        ]
        child_count, document_count = EvaluationParentChildBackfill().run(
            db=db,
            document_ids=document_ids,
            batch_size=args.batch_size,
        )

    print(
        "Parent-Child evaluation backfill completed: "
        f"documents={document_count}, new_children={child_count}"
    )


if __name__ == "__main__":
    main()
