from __future__ import annotations

import re
import unicodedata
from pathlib import Path


class DocumentUploadValidationError(ValueError):
    """上传文档不满足安全策略。"""


class DocumentUploadTooLargeError(DocumentUploadValidationError):
    """上传文档超过允许大小。"""


class DocumentUploadMediaTypeError(DocumentUploadValidationError):
    """上传文档类型不受支持或声明类型不匹配。"""


class DocumentUploadPolicy:
    """文档上传安全策略。

    负责文件名、大小、扩展名、声明 MIME 与基础内容特征校验。
    不负责 HTTP、数据库或具体 Storage Provider。
    """

    _ALLOWED_MIME_TYPES: dict[str, frozenset[str]] = {
        ".pdf": frozenset({"application/pdf", "application/octet-stream"}),
        ".txt": frozenset({"text/plain", "application/octet-stream"}),
        ".md": frozenset({
            "text/markdown",
            "text/plain",
            "text/x-markdown",
            "application/octet-stream",
        }),
        ".markdown": frozenset({
            "text/markdown",
            "text/plain",
            "text/x-markdown",
            "application/octet-stream",
        }),
        ".html": frozenset({
            "text/html",
            "application/xhtml+xml",
            "application/octet-stream",
        }),
        ".htm": frozenset({
            "text/html",
            "application/xhtml+xml",
            "application/octet-stream",
        }),
    }

    _WINDOWS_RESERVED_NAMES = frozenset({
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{index}" for index in range(1, 10)),
        *(f"LPT{index}" for index in range(1, 10)),
    })

    _CONTROL_CHARACTER_PATTERN = re.compile(r"[\x00-\x1f\x7f]")

    def __init__(
        self,
        max_file_size_bytes: int = 20 * 1024 * 1024,
        max_filename_length: int = 255,
    ) -> None:
        if max_file_size_bytes <= 0:
            raise ValueError("max_file_size_bytes must be greater than 0")
        if max_filename_length <= 0:
            raise ValueError("max_filename_length must be greater than 0")

        self.max_file_size_bytes = max_file_size_bytes
        self.max_filename_length = max_filename_length

    def validate(
        self,
        filename: str,
        content: bytes,
        content_type: str | None = None,
    ) -> str:
        """校验上传内容并返回可安全保存/展示的文件名。"""
        safe_filename = self.sanitize_filename(filename)
        self.validate_size(content)

        extension = Path(safe_filename).suffix.lower()
        self._validate_extension(extension)
        self._validate_declared_content_type(extension, content_type)
        self._validate_content(extension, content)

        return safe_filename

    def validate_size(self, content: bytes) -> None:
        """校验文件大小，供 Router 限流和 Service 二次校验复用。"""
        if not content:
            raise DocumentUploadValidationError("file content cannot be empty")

        if len(content) > self.max_file_size_bytes:
            raise DocumentUploadTooLargeError(
                "file size exceeds configured upload limit"
            )

    def sanitize_filename(self, filename: str) -> str:
        """去除客户端路径信息并拒绝危险文件名。"""
        normalized = unicodedata.normalize("NFC", filename or "")
        normalized = normalized.replace("\\", "/")
        safe_filename = normalized.rsplit("/", 1)[-1].strip().rstrip(". ")

        if not safe_filename or safe_filename in {".", ".."}:
            raise DocumentUploadValidationError("filename cannot be empty")

        if len(safe_filename) > self.max_filename_length:
            raise DocumentUploadValidationError("filename is too long")

        if self._CONTROL_CHARACTER_PATTERN.search(safe_filename):
            raise DocumentUploadValidationError(
                "filename contains control characters"
            )

        stem = Path(safe_filename).stem.rstrip(". ").upper()
        if stem in self._WINDOWS_RESERVED_NAMES:
            raise DocumentUploadValidationError("filename is reserved")

        return safe_filename

    def _validate_extension(self, extension: str) -> None:
        if extension not in self._ALLOWED_MIME_TYPES:
            raise DocumentUploadMediaTypeError(
                "unsupported document file type"
            )

    def _validate_declared_content_type(
        self,
        extension: str,
        content_type: str | None,
    ) -> None:
        if not content_type:
            return

        normalized_content_type = (
            content_type.split(";", 1)[0].strip().lower()
        )
        if normalized_content_type not in self._ALLOWED_MIME_TYPES[extension]:
            raise DocumentUploadMediaTypeError(
                "declared content type does not match document type"
            )

    @staticmethod
    def _validate_content(extension: str, content: bytes) -> None:
        if extension == ".pdf":
            if not content.startswith(b"%PDF-"):
                raise DocumentUploadMediaTypeError(
                    "PDF signature is invalid"
                )
            return

        try:
            decoded = content.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise DocumentUploadMediaTypeError(
                "text document must be UTF-8 encoded"
            ) from exc

        if "\x00" in decoded:
            raise DocumentUploadMediaTypeError(
                "text document contains binary null bytes"
            )
