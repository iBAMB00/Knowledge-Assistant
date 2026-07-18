from fastapi import APIRouter, File, HTTPException, UploadFile, Depends
from app.schemas.document_info import DocumentInfo

from app.services.document_service import DocumentService
from app.services.storage_service import StorageService
from app.repositories.document_repository import DocumentRepository
from app.schemas.document_response import DocumentResponse

from sqlalchemy.orm import Session
from app.core.database import get_db

router = APIRouter(prefix="/documents", tags=["Documents"])

storage_service = StorageService()
document_repository = DocumentRepository()
document_service = DocumentService(storage_service, document_repository)


@router.post("/",
    response_model=DocumentInfo,
)
async def upload_document(
        file: UploadFile = File(...),
        db: Session = Depends(get_db),
    ):
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
        # API 层负责读取 HTTP 上传文件。
        # Service 层只接收普通 bytes，不依赖 FastAPI。
        content = await file.read()

        document = document_service.upload_document(
            db=db,
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

@router.get(
    "/",
    response_model=list[DocumentResponse],
)
def list_documents(
    db: Session = Depends(get_db),
) -> list[DocumentResponse]:
    """
    查询所有文档。
    """
    return document_service.list_documents(
        db=db,
    )


@router.delete(
    "/{document_id}",
    response_model=None,
)
def delete_document(
    document_id: int,
    db: Session = Depends(get_db),
) -> None:
    """
    删除文档记录。
    """
    document_service.delete_document(
        db=db,
        document_id=document_id,
    )