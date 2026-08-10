from functools import lru_cache

from app.core.config import Settings, get_settings
from app.services.storage.local import LocalStorageProvider
from app.services.storage.minio import MinioStorageProvider
from app.services.storage_service import StorageService


def create_storage_service(
    settings: Settings | None = None,
) -> StorageService:
    """根据应用配置创建统一 StorageService。"""
    resolved_settings = settings or get_settings()

    if resolved_settings.storage_backend == "local":
        return StorageService(
            provider=LocalStorageProvider(
                storage_dir=resolved_settings.local_storage_dir,
            )
        )

    if resolved_settings.storage_backend == "minio":
        if not resolved_settings.minio_access_key:
            raise ValueError("MINIO_ACCESS_KEY is required for minio storage")
        if not resolved_settings.minio_secret_key:
            raise ValueError("MINIO_SECRET_KEY is required for minio storage")

        return StorageService(
            provider=MinioStorageProvider(
                endpoint=resolved_settings.minio_endpoint,
                access_key=resolved_settings.minio_access_key,
                secret_key=resolved_settings.minio_secret_key,
                bucket_name=resolved_settings.minio_bucket,
                secure=resolved_settings.minio_secure,
            )
        )

    raise ValueError(
        f"unsupported storage backend: {resolved_settings.storage_backend}"
    )


@lru_cache
def get_storage_service() -> StorageService:
    """获取当前进程共享的 StorageService。"""
    return create_storage_service()
