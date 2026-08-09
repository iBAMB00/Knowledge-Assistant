from io import BytesIO
from typing import Any

from app.services.storage.base import StorageProvider, normalize_storage_key


class MinioStorageProvider(StorageProvider):
    """基于 MinIO Python SDK 的对象存储 Provider。"""

    def __init__(
        self,
        endpoint: str,
        access_key: str,
        secret_key: str,
        bucket_name: str,
        secure: bool = False,
        client: Any | None = None,
    ) -> None:
        if not endpoint.strip():
            raise ValueError("minio endpoint cannot be empty")
        if not access_key.strip():
            raise ValueError("minio access key cannot be empty")
        if not secret_key.strip():
            raise ValueError("minio secret key cannot be empty")
        if not bucket_name.strip():
            raise ValueError("minio bucket cannot be empty")

        self.bucket_name = bucket_name.strip()

        if client is None:
            from minio import Minio

            client = Minio(
                endpoint=endpoint.strip(),
                access_key=access_key.strip(),
                secret_key=secret_key.strip(),
                secure=secure,
            )

        self.client = client
        self._bucket_initialized = False

    def save(self, storage_key: str, content: bytes) -> None:
        self._ensure_bucket()
        normalized_key = normalize_storage_key(storage_key)
        self.client.put_object(
            self.bucket_name,
            normalized_key,
            BytesIO(content),
            length=len(content),
        )

    def read(self, storage_key: str) -> bytes:
        normalized_key = normalize_storage_key(storage_key)
        response = self.client.get_object(
            self.bucket_name,
            normalized_key,
        )

        try:
            return response.read()
        finally:
            close = getattr(response, "close", None)
            if callable(close):
                close()

            release_conn = getattr(response, "release_conn", None)
            if callable(release_conn):
                release_conn()

    def delete(self, storage_key: str) -> None:
        normalized_key = normalize_storage_key(storage_key)
        self.client.remove_object(
            self.bucket_name,
            normalized_key,
        )

    def exists(self, storage_key: str) -> bool:
        normalized_key = normalize_storage_key(storage_key)

        try:
            self.client.stat_object(
                self.bucket_name,
                normalized_key,
            )
            return True
        except Exception as exc:
            error_code = getattr(exc, "code", None)

            if error_code in {
                "NoSuchKey",
                "NoSuchObject",
                "NoSuchBucket",
            }:
                return False

            raise

    def _ensure_bucket(self) -> None:
        if self._bucket_initialized:
            return

        if not self.client.bucket_exists(self.bucket_name):
            try:
                self.client.make_bucket(self.bucket_name)
            except Exception as exc:
                error_code = getattr(exc, "code", None)
                if error_code not in {
                    "BucketAlreadyOwnedByYou",
                    "BucketAlreadyExists",
                }:
                    raise

        self._bucket_initialized = True
