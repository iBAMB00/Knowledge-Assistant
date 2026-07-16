from fastapi import APIRouter, File, HTTPException, UploadFile
from app.models.document import DocumentInfo

from app.services.document_service import DocumentService

router = APIRouter(prefix="/documents", tags=["Documents"])

document_service = DocumentService()


@router.post("/")
async def upload_document(file: UploadFile = File(...)) -> DocumentInfo:
    """
    上传并保存文档。

    Args:
        file: 通过 multipart/form-data 上传的文件。

    Returns:
        上传成功后的文档基础信息。

    Raises:
        HTTPException: 文件不合法或保存失败时抛出。
    """
    try:
        content = await file.read()

        document = document_service.upload_document(
            filename=file.filename or "",
            content=content,
        )

        return document

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="文档上传失败",
        ) from exc
    
    finally:
        await file.close()