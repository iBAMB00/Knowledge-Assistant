from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.repositories.document_repository import DocumentRepository
from app.repositories.document_content_repository import DocumentContentRepository
from app.schemas.document_info import DocumentInfo
from app.schemas.document_response import DocumentResponse
from app.services.document_service import DocumentService
from app.services.parser_service import ParserService
from app.services.storage_service import StorageService
from app.services.document_processing_service import DocumentProcessingService
from app.services.chunk_service import ChunkService
from app.repositories.document_chunk_repository import DocumentChunkRepository
from app.schemas.chunk_response import ChunkResponse



router = APIRouter(
    prefix="/documents",
    tags=["Documents"],
)


storage_service = StorageService()
document_repository = DocumentRepository()
document_content_repository = DocumentContentRepository()
parser_service = ParserService()
chunk_service = ChunkService()
document_chunk_repository = DocumentChunkRepository()

document_service = DocumentService(
    storage_service=storage_service,
    document_repository=document_repository,
    document_content_repository=document_content_repository,
    document_chunk_repository=document_chunk_repository,
)

document_processing_service = DocumentProcessingService(
    storage_service=storage_service,
    document_repository=document_repository,
    document_content_repository=document_content_repository,
    parser_service=parser_service,
    chunk_service=chunk_service,
    document_chunk_repository=document_chunk_repository,
)



@router.post(
    "/",
    response_model=DocumentInfo,
)
async def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> DocumentInfo:
    """
    上传并保存原始文档。
    """

    try:
        content = await file.read()

        return document_service.upload_document(
            db=db,
            filename=file.filename or "",
            content=content,
        )

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
    查询文档列表。
    """

    return document_service.list_documents(
        db=db,
    )


@router.delete(
    "/{document_id}",
    status_code=204,
)
def delete_document(
    document_id: int,
    db: Session = Depends(get_db),
) -> None:
    """
    删除指定文档。
    """

    try:
        document_service.delete_document(
            db=db,
            document_id=document_id,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="文档删除失败",
        ) from exc


@router.post(
    "/{document_id}/process",
    response_model=DocumentResponse,
)
def process_document(
    document_id: int,
    db: Session = Depends(get_db),
) -> DocumentResponse:
    """
    同步解析文档并保存解析全文。
    """

    try:
        return document_processing_service.process_document(
            db=db,
            document_id=document_id,
        )

    except ValueError as exc:
        detail = str(exc)

        if detail == "document not found":
            status_code = 404
        elif detail == "invalid document status":
            status_code = 409
        else:
            status_code = 400

        raise HTTPException(
            status_code=status_code,
            detail=detail,
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="文档处理失败",
        ) from exc

@router.get(
    "/{document_id}/content",
    response_model=str,
)
def get_document_content(
    document_id: int,
    db: Session = Depends(get_db),
) -> str:
    """
    获取指定文档的解析全文。
    """
    try:
        return document_service.get_document_content(
            db=db,
            document_id=document_id,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="文档内容获取失败",
        ) from exc

@router.get(
    "/documents/{document_id}/chunks",
    response_model=list[ChunkResponse],
)
def get_document_chunks(
    document_id: int,
    db: Session = Depends(get_db),
):
    """
    查询文档切片。
    """
    try:
        return document_service.get_document_chunks(
            db=db,
            document_id=document_id,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="文档切片获取失败",
        ) from exc