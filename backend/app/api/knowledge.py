import logging

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from sqlalchemy.orm import Session

from app.api.dependencies.auth import get_current_user
from app.core.database import get_db
from app.models.database.user import User
from app.repositories.document_chunk_repository import DocumentChunkRepository
from app.repositories.document_content_repository import DocumentContentRepository
from app.repositories.document_repository import DocumentRepository
from app.repositories.processing_job_repository import ProcessingJobRepository
from app.repositories.knowledge_base_repository import KnowledgeBaseRepository
from app.schemas.chunk_response import ChunkResponse
from app.schemas.chunk_summary_response import ChunkSummaryResponse
from app.schemas.document_info import DocumentInfo
from app.schemas.document_list_item_response import DocumentListItemResponse
from app.schemas.document_response import DocumentResponse
from app.schemas.embedding_process_response import EmbeddingProcessResponse
from app.schemas.processing_job_create_request import ProcessingJobCreateRequest
from app.schemas.processing_job_response import ProcessingJobResponse
from app.services.document_operation_policy import (
    DocumentOperationConflictError,
    DocumentOperationPolicy,
)
from app.services.document_service import DocumentService
from app.services.processing_job_dispatcher import (
    ProcessingJobDispatcher,
    ProcessingJobDispatchError,
)
from app.services.processing_job_runtime import get_processing_job_executor
from app.services.processing_job_service import (
    ActiveProcessingJobError,
    InvalidProcessingJobError,
    ProcessingJobService,
)
from app.services.storage_service import StorageService
from app.services.knowledge_base_access_policy import (
    KnowledgeBaseAccessPolicy,
    ResourceAccessNotFoundError,
)
from app.services.vector_store.factory import get_vector_store_components
from app.tasks.processing_job import execute_processing_job


logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/documents",
    tags=["Documents"],
)


storage_service = StorageService()
document_repository = DocumentRepository()
document_content_repository = DocumentContentRepository()
document_chunk_repository = DocumentChunkRepository()
processing_job_repository = ProcessingJobRepository()
knowledge_base_repository = KnowledgeBaseRepository()
knowledge_base_access_policy = KnowledgeBaseAccessPolicy(
    knowledge_base_repository=knowledge_base_repository,
    document_repository=document_repository,
)
vector_store_components = get_vector_store_components()

document_operation_policy = DocumentOperationPolicy(
    processing_job_repository=processing_job_repository,
)

document_service = DocumentService(
    storage_service=storage_service,
    document_repository=document_repository,
    document_content_repository=document_content_repository,
    document_chunk_repository=document_chunk_repository,
    processing_job_repository=processing_job_repository,
    document_operation_policy=document_operation_policy,
    vector_index=vector_store_components.vector_index,
)

processing_job_service = ProcessingJobService(
    document_repository=document_repository,
    processing_job_repository=processing_job_repository,
)
processing_job_executor = get_processing_job_executor()
processing_job_dispatcher = ProcessingJobDispatcher(task=execute_processing_job)



def _require_knowledge_base_access(
    db: Session,
    knowledge_base_id: int,
    current_user: User,
) -> None:
    try:
        knowledge_base_access_policy.get_accessible_knowledge_base(
            db=db,
            knowledge_base_id=knowledge_base_id,
            user=current_user,
        )
    except ResourceAccessNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def _require_document_access(
    db: Session,
    document_id: int,
    current_user: User,
) -> None:
    try:
        knowledge_base_access_policy.get_accessible_document(
            db=db,
            document_id=document_id,
            user=current_user,
        )
    except ResourceAccessNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def get_processing_job_dispatcher() -> ProcessingJobDispatcher:
    """获取 ProcessingJob 的 Celery 派发器，便于 API 测试替换。"""
    return processing_job_dispatcher


@router.post(
    "/",
    response_model=DocumentInfo,
)
async def upload_document(
    file: UploadFile = File(...),
    knowledge_base_id: int = Form(..., gt=0),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> DocumentInfo:
    """
    上传并保存原始文档。
    """

    try:
        _require_knowledge_base_access(db, knowledge_base_id, current_user)
        content = await file.read()

        return document_service.upload_document(
            db=db,
            filename=file.filename or "",
            content=content,
            knowledge_base_id=knowledge_base_id,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="文档上传失败",
        ) from exc

    finally:
        await file.close()

@router.get(
    "/{document_id}",
    response_model=DocumentResponse,
)
def get_document_by_id(
    document_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> DocumentResponse:
    """
    获取指定文档的详细信息。
    """
    try:
        _require_document_access(db, document_id, current_user)
        return document_service.get_document_by_id(
            db=db,
            document_id=document_id,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="文档查询失败",
        ) from exc



@router.get(
    "/",
    response_model=list[DocumentListItemResponse],
)
def list_documents(
    knowledge_base_id: int = Query(..., gt=0),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> list[DocumentListItemResponse]:
    """
    查询文档列表。
    """

    _require_knowledge_base_access(db, knowledge_base_id, current_user)
    return document_service.list_documents(
        db=db,
        knowledge_base_id=knowledge_base_id,
    )


@router.delete(
    "/{document_id}",
    status_code=204,
)
def delete_document(
    document_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> None:
    """
    删除指定文档。
    """

    try:
        _require_document_access(db, document_id, current_user)
        document_service.delete_document(
            db=db,
            document_id=document_id,
        )

    except DocumentOperationConflictError as exc:
        raise HTTPException(
            status_code=409,
            detail=str(exc),
        ) from exc

    except ValueError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="文档删除失败",
        ) from exc




def _mark_dispatch_failed(db: Session, job_id: int) -> None:
    """派发失败时释放活动任务约束，并保留失败记录。"""
    try:
        processing_job_service.fail_job(
            db=db,
            job_id=job_id,
            error_message="处理任务派发失败",
        )
    except Exception:
        db.rollback()
        logger.exception(
            "failed to mark processing job dispatch failure: job_id=%s",
            job_id,
        )


@router.post(
    "/{document_id}/processing-jobs",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=ProcessingJobResponse,
)
def create_document_processing_job(
    document_id: int,
    request: ProcessingJobCreateRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    dispatcher: ProcessingJobDispatcher = Depends(get_processing_job_dispatcher),
) -> ProcessingJobResponse:
    """创建持久化任务并将 job_id 派发到 Celery Worker。"""
    try:
        _require_document_access(db, document_id, current_user)
        job = processing_job_service.create_job(
            db=db,
            document_id=document_id,
            job_type=request.job_type,
        )
    except ActiveProcessingJobError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except InvalidProcessingJobError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        detail = str(exc)
        status_code = 404 if detail == "document not found" else 400
        raise HTTPException(status_code=status_code, detail=detail) from exc
    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(status_code=500, detail="处理任务创建失败") from exc

    try:
        dispatcher.dispatch(job.id)
    except ProcessingJobDispatchError as exc:
        logger.exception("processing job dispatch failed: job_id=%s", job.id)
        _mark_dispatch_failed(db=db, job_id=job.id)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="处理任务派发失败",
        ) from exc

    logger.info(
        "processing job dispatched: job_id=%s document_id=%s job_type=%s",
        job.id,
        document_id,
        request.job_type.value,
    )

    return job

@router.post(
    "/{document_id}/process",
    response_model=DocumentResponse,
)
def process_document(
    document_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> DocumentResponse:
    """
    同步完成文档解析与切片。
    """

    try:
        _require_document_access(db, document_id, current_user)
        return processing_job_executor.process_document(
            db=db,
            document_id=document_id,
        )

    except ActiveProcessingJobError as exc:
        raise HTTPException(
            status_code=409,
            detail=str(exc),
        ) from exc

    except InvalidProcessingJobError as exc:
        raise HTTPException(
            status_code=409,
            detail=str(exc),
        ) from exc

    except ValueError as exc:
        detail = str(exc)

        if detail == "document not found":
            status_code = 404

        elif (
            detail
            == "document is already being processed"
            or detail == "document content not found"
            or detail.startswith(
                "invalid document status:"
            )
        ):
            status_code = 409

        else:
            status_code = 400

        raise HTTPException(
            status_code=status_code,
            detail=detail,
        ) from exc

    except HTTPException:
        raise

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
    current_user: User = Depends(get_current_user),
) -> str:
    """
    获取指定文档的解析全文。
    """
    try:
        _require_document_access(db, document_id, current_user)
        return document_service.get_document_content(
            db=db,
            document_id=document_id,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc
    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="文档内容获取失败",
        ) from exc

@router.get(
    "/{document_id}/chunks",
    response_model=list[ChunkResponse],
)
def get_document_chunks(
    document_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    查询文档切片。
    """
    try:
        _require_document_access(db, document_id, current_user)
        return document_service.get_document_chunks(
            db=db,
            document_id=document_id,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="文档切片获取失败",
        ) from exc

@router.get(
    "/{document_id}/chunk-summary",
    response_model=ChunkSummaryResponse,
)
def get_chunk_summary(
    document_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    查询文档切片统计。
    """

    try:
        _require_document_access(db, document_id, current_user)
        return document_service.get_chunk_summary(
            db=db,
            document_id=document_id,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="切片统计获取失败",
        ) from exc

@router.post(
    "/{document_id}/embeddings",
    response_model=EmbeddingProcessResponse,
)
def create_document_embeddings(
    document_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> EmbeddingProcessResponse:
    """
    为指定文档的Chunk生成向量。

    文档必须已经完成解析和切片。
    """

    try:
        _require_document_access(db, document_id, current_user)
        processed_count = (
            processing_job_executor.embed_document(
                db=db,
                document_id=document_id,
            )
        )

        return EmbeddingProcessResponse(
            document_id=document_id,
            processed_count=processed_count,
        )

    except ActiveProcessingJobError as exc:
        raise HTTPException(
            status_code=409,
            detail=str(exc),
        ) from exc

    except InvalidProcessingJobError as exc:
        raise HTTPException(
            status_code=409,
            detail=str(exc),
        ) from exc

    except ValueError as exc:
        detail = str(exc)

        if detail == "document not found":
            status_code = 404
        else:
            status_code = 409

        raise HTTPException(
            status_code=status_code,
            detail=detail,
        ) from exc

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="文档向量化失败",
        ) from exc
