"""将现有 LocalStorage 对象幂等复制到 MinIO。

默认不删除本地文件，也不修改数据库；documents.storage_key 保持不变，
因此完成复制后只需切换 STORAGE_BACKEND/DOCKER_STORAGE_BACKEND 即可。
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.database import SessionLocal
from app.models.database.document import Document
from app.services.storage.base import StorageProvider
from app.services.storage.local import LocalStorageProvider
from app.services.storage.minio import MinioStorageProvider


@dataclass(frozen=True)
class MigrationSummary:
    total: int
    copied: int
    skipped: int
    missing_local: int


def migrate_documents(
    db: Session,
    source: StorageProvider,
    target: StorageProvider,
    dry_run: bool = False,
) -> MigrationSummary:
    """按 Document.storage_key 幂等复制对象。"""
    documents = db.query(Document).order_by(Document.id.asc()).all()
    copied = 0
    skipped = 0
    missing_local = 0

    for document in documents:
        storage_key = document.storage_key

        if target.exists(storage_key):
            skipped += 1
            continue

        if not source.exists(storage_key):
            missing_local += 1
            print(
                f"MISSING document_id={document.id} storage_key={storage_key}"
            )
            continue

        if dry_run:
            print(
                f"COPY document_id={document.id} storage_key={storage_key}"
            )
            copied += 1
            continue

        target.save(storage_key, source.read(storage_key))
        copied += 1
        print(
            f"COPIED document_id={document.id} storage_key={storage_key}"
        )

    return MigrationSummary(
        total=len(documents),
        copied=copied,
        skipped=skipped,
        missing_local=missing_local,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只检查和打印迁移计划，不写入 MinIO。",
    )
    args = parser.parse_args()

    settings = get_settings()

    if not settings.minio_access_key or not settings.minio_secret_key:
        raise RuntimeError(
            "MINIO_ACCESS_KEY and MINIO_SECRET_KEY are required"
        )

    source = LocalStorageProvider(settings.local_storage_dir)
    target = MinioStorageProvider(
        endpoint=settings.minio_endpoint,
        access_key=settings.minio_access_key,
        secret_key=settings.minio_secret_key,
        bucket_name=settings.minio_bucket,
        secure=settings.minio_secure,
    )

    db = SessionLocal()
    try:
        summary = migrate_documents(
            db=db,
            source=source,
            target=target,
            dry_run=args.dry_run,
        )
    finally:
        db.close()

    print(
        "SUMMARY "
        f"total={summary.total} "
        f"copied={summary.copied} "
        f"skipped={summary.skipped} "
        f"missing_local={summary.missing_local} "
        f"dry_run={args.dry_run}"
    )


if __name__ == "__main__":
    main()
