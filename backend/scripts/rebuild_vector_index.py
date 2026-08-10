from __future__ import annotations

import argparse

from app.constants.document_status import DocumentStatus
from app.core.config import get_settings
from app.core.database import SessionLocal
from app.repositories.chunk_embedding_repository import ChunkEmbeddingRepository
from app.repositories.document_repository import DocumentRepository
from app.services.vector_index_service import VectorIndexService
from app.services.vector_store.factory import VectorStoreFactory


def parse_args() -> argparse.Namespace:
    """解析向量索引重建命令参数。"""
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild the external vector index from completed "
            "ChunkEmbedding records stored in SQL."
        ),
    )
    parser.add_argument(
        "--document-id",
        type=int,
        default=None,
        help=(
            "Only rebuild one completed document. "
            "When omitted, rebuild all completed documents."
        ),
    )
    return parser.parse_args()


def is_completed(document) -> bool:
    """判断文档是否已经完成处理，可用于外部向量索引重建。"""
    return (
        DocumentStatus(document.status)
        == DocumentStatus.COMPLETED
    )


def load_documents(
    document_repository: DocumentRepository,
    db,
    document_id: int | None,
):
    """加载本次需要重建索引的已完成文档。"""
    if document_id is not None:
        if document_id <= 0:
            raise ValueError(
                "document_id must be greater than zero"
            )

        document = document_repository.find_by_id(
            db=db,
            document_id=document_id,
        )

        if document is None:
            raise ValueError("document not found")

        if not is_completed(document):
            raise ValueError(
                "document is not completed and cannot be reindexed: "
                f"document_id={document.id}, status={document.status}"
            )

        return [document], 0

    documents = document_repository.find_all(db=db)
    completed_documents = [
        document
        for document in documents
        if is_completed(document)
    ]
    skipped_count = len(documents) - len(completed_documents)

    return completed_documents, skipped_count


def main() -> int:
    """从 SQL 中已持久化的 ChunkEmbedding 重建外部向量索引。"""
    args = parse_args()
    settings = get_settings()

    if settings.vector_store_backend != "qdrant":
        print(
            "[ERROR] vector_store_backend must be 'qdrant' "
            f"for this rebuild, got: {settings.vector_store_backend}"
        )
        return 2

    components = VectorStoreFactory.create(settings=settings)
    vector_index = components.vector_index

    if vector_index is None:
        print(
            "[ERROR] current vector store backend does not "
            "provide an external VectorIndex"
        )
        return 2

    document_repository = DocumentRepository()
    chunk_embedding_repository = ChunkEmbeddingRepository()
    service = VectorIndexService(
        document_repository=document_repository,
        chunk_embedding_repository=chunk_embedding_repository,
        vector_index=vector_index,
    )

    db = SessionLocal()

    try:
        documents, skipped_count = load_documents(
            document_repository=document_repository,
            db=db,
            document_id=args.document_id,
        )

        print(
            "[INFO] vector backend=qdrant "
            f"collection={settings.qdrant_collection_name}"
        )
        print(
            "[INFO] documents_to_rebuild="
            f"{len(documents)} skipped_not_completed={skipped_count}"
        )

        # 先显式确保 Collection 存在。Qdrant 数据完全丢失时，
        # 该步骤负责重新创建 Collection，并校验向量维度配置。
        vector_index.ensure_collection()

        success_count = 0
        failed_count = 0
        indexed_point_count = 0

        for document in documents:
            try:
                point_count = service.index_document(
                    db=db,
                    document_id=document.id,
                )
            except Exception as exc:
                db.rollback()
                failed_count += 1
                print(
                    "[FAILED] "
                    f"document_id={document.id} "
                    f"filename={document.filename!r} "
                    f"error={exc}"
                )
                continue

            success_count += 1
            indexed_point_count += point_count
            print(
                "[OK] "
                f"document_id={document.id} "
                f"filename={document.filename!r} "
                f"points={point_count}"
            )

        print("\n=== Vector Index Rebuild Summary ===")
        print(f"documents_total={len(documents)}")
        print(f"documents_success={success_count}")
        print(f"documents_failed={failed_count}")
        print(f"documents_skipped={skipped_count}")
        print(f"points_upserted={indexed_point_count}")

        if failed_count > 0:
            return 1

        return 0

    except Exception as exc:
        db.rollback()
        print(f"[ERROR] rebuild failed: {exc}")
        return 1

    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
