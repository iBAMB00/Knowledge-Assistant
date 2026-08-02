from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    HTTPException,
    UploadFile,
    status,
)
from sqlalchemy.orm import Session

from app.core.database import SessionLocal, get_db
from app.repositories.chunk_embedding_repository import ChunkEmbeddingRepository
from app.repositories.document_chunk_repository import DocumentChunkRepository
from app.repositories.document_content_repository import DocumentContentRepository
from app.repositories.document_repository import DocumentRepository
from app.repositories.processing_job_repository import ProcessingJobRepository
from app.schemas.chunk_response import ChunkResponse
from app.schemas.chunk_summary_response import ChunkSummaryResponse
from app.schemas.document_info import DocumentInfo
from app.schemas.document_response import DocumentResponse
from app.schemas.embedding_process_response import EmbeddingProcessResponse
from app.schemas.processing_job_create_request import ProcessingJobCreateRequest
from app.schemas.processing_job_response import ProcessingJobResponse
from app.services.chunk_service import ChunkService
from app.services.document_processing_service import DocumentProcessingService
from app.services.document_service import DocumentService
from app.services.embedding.factory import EmbeddingFactory
from app.services.embedding_service import EmbeddingService
from app.services.parser_service import ParserService
from app.services.processing_job_executor import ProcessingJobExecutor
from app.services.processing_job_runner import ProcessingJobRunner
from app.services.processing_job_service import (
    ActiveProcessingJobError,
    InvalidProcessingJobError,
    ProcessingJobService,
)
from app.services.storage_service import StorageService

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
chunk_embedding_repository = ChunkEmbeddingRepository()
processing_job_repository = ProcessingJobRepository()

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

embedding_provider = EmbeddingFactory.create()

embedding_service = EmbeddingService(
    document_repository=document_repository,
    document_chunk_repository=document_chunk_repository,
    chunk_embedding_repository=chunk_embedding_repository,
    embedding_provider=embedding_provider,
)

processing_job_service = ProcessingJobService(
    document_repository=document_repository,
    processing_job_repository=(
        processing_job_repository
    ),
)

processing_job_executor = ProcessingJobExecutor(
    document_repository=document_repository,
    processing_job_service=(
        processing_job_service
    ),
    document_processing_service=(
        document_processing_service
    ),
    embedding_service=embedding_service,
)

processing_job_runner = ProcessingJobRunner(
    session_factory=SessionLocal,
    executor=processing_job_executor,
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
    "/{document_id}",
    response_model=DocumentResponse,
)
def get_document_by_id(
    document_id: int,
    db: Session = Depends(get_db),
) -> DocumentResponse:
    """
    获取指定文档的详细信息。
    """
    try:
        return document_service.get_document_by_id(
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
            detail="文档查询失败",
        ) from exc



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
    同步完成文档解析与切片。
    """

    try:
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
    "/{document_id}/chunks",
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

@router.get(
    "/{document_id}/chunk-summary",
    response_model=ChunkSummaryResponse,
)
def get_chunk_summary(
    document_id: int,
    db: Session = Depends(get_db),
):
    """
    查询文档切片统计。
    """

    try:
        return document_service.get_chunk_summary(
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
            detail="切片统计获取失败",
        ) from exc

@router.post(
    "/{document_id}/embeddings",
    response_model=EmbeddingProcessResponse,
)
def create_document_embeddings(
    document_id: int,
    db: Session = Depends(get_db),
) -> EmbeddingProcessResponse:
    """
    为指定文档的Chunk生成向量。

    文档必须已经完成解析和切片。
    """

    try:
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

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="文档向量化失败",
        ) from exc


def get_processing_job_runner(
) -> ProcessingJobRunner:
    """
    获取后台任务Runner。

    单独提供依赖函数，便于API测试替换为Fake Runner。
    """

    return processing_job_runner

@router.post(
    "/{document_id}/processing-jobs",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=ProcessingJobResponse,
)
def create_document_processing_job(
    document_id: int,
    request: ProcessingJobCreateRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    runner: ProcessingJobRunner = Depends(
        get_processing_job_runner
    ),
) -> ProcessingJobResponse:
    """
    创建文档后台处理任务。

    接口只创建任务并返回，不等待实际处理完成。
    """

    try:
        job = processing_job_service.create_job(
            db=db,
            document_id=document_id,
            job_type=request.job_type,
        )

        background_tasks.add_task(
            runner.run,
            job.id,
        )

        return job

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
            status_code = 400

        raise HTTPException(
            status_code=status_code,
            detail=detail,
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="处理任务创建失败",
        ) from exc