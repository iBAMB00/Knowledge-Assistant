from fastapi import UploadFile

from app.services.document_upload_policy import DocumentUploadPolicy


async def read_upload_with_limit(
    file: UploadFile,
    policy: DocumentUploadPolicy,
) -> bytes:
    """最多读取配置上限 + 1 字节，并复用上传策略校验大小。"""
    content = await file.read(policy.max_file_size_bytes + 1)
    policy.validate_size(content)
    return content
